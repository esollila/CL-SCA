import numpy as np
from ..base import CLobject
import numbers
from scipy.linalg import get_lapack_funcs, LinAlgError


class CLHybridSchedule(CLobject):
    """
    CL-Hyb: switches from the CL-SCA update to CL-FP update after a
    fixed number of iterations. The method uses fixed  step-size schedule, 
    ('fss'): eta is held at its initial value for all iterations, but are different for 
    FP and SCA updates. See arguments eta_sca and eta_fp.

    Phase 1 (k <= switch_iter):  SCA proposal
        hat_gamma_n = [gamma_n + sqrt(e_n/d_n^3) - 1/d_n]_+ ,  step eta_sca
    Phase 2 (k >  switch_iter):  FP proposal
        hat_gamma_n = [gamma_n + e_n/d_n^2 - 1/d_n]_+ ,        step eta_fp

    Rationale: the two proposals move every coordinate in the same
    direction, with the FP displacement equal to (1 + sqrt(e_n/d_n))
    times the SCA displacement. Far from a solution this factor is large,
    so the sqrt-tamed SCA step tolerates a larger damping factor and
    handles the transient; near a KKT point the factor tends to 2, and the
    FP map with step eta_fp behaves locally like SCA with step 2*eta_fp,
    lifting SCA's effective-step cap of 1/2 in the endgame. Stability of
    phase 2 requires eta_fp < 2/lambda_max(D * Hessian) at the solution;
    eta_fp ~ 0.5 is a safe practical default, larger values can diverge.

    Both proposals satisfy the same descent inequality with curvature
    d_n^2, so the sufficient-decrease/stationarity-rate guarantees apply
    to each phase with the same constants.

    Implementation. With Sigma = L L^H and SCM = S_h S_h^H (the latter computed once),
        G = L^{-1} A,   T = L^{-1} S_h,
        d_n = ||g_n||^2,   e_n = ||T^H g_n||^2,
        L(gamma) = tr(Sigma^{-1} SCM) + log|Sigma| = ||T||_F^2 + 2 sum_i log L_ii.
    L^{-1} (M x M, cheap) is formed explicitly so that G, T and T^H G are matrix
    products, which BLAS executes faster than triangular solves; this replaces
    cho_solve (two triangular solves with A) and SCM @ B, and the loss comes for
    free. Columns with negligible gamma_n are left out of Sigma (see prunetol).
    The iterates agree (up to round-off) with those of the cho_factor/cho_solve
    version.

    Parameters
    ----------
    sig : float, optional
        Noise variance (sigma^2). Default is 1.0.
    max_iter : int, optional
        Maximum number of iterations. Default is 50.
    errtol : float, optional
        Tolerance on the projected KKT residual
            || [gamma - grad L(gamma)]_+ - gamma ||_inf ,
        evaluated at the current iterate; the same criterion is used by all compared
        methods. Default is 1e-3.
    switch_iter : int, optional
        Number of initial SCA iterations before switching to FP updates.
        Default is 4.
    eta_sca : float, optional
        Fixed step size for the SCA phase. Default is 0.8.
    eta_fp : float, optional
        Fixed step size for the FP phase. Default is 0.5.
    gam_init : array-like, optional
        Initial gammas of shape (N,). If None, initialized to all zeros.
    print_flag : bool, optional
        If True, prints convergence information. Default is False.
    calculate_loss : bool, optional
        If True, stores the loss and the projected KKT residual at each iteration.
        Default is False.
    prunetol : float, optional
        Columns with gamma_n <= prunetol * max(gamma) are left out when Sigma_gamma
        is formed (the gammas themselves are still updated). Default 1e-12 changes
        Sigma only at round-off level; set 0 to use all nonzero gammas.
    """

    def __init__(self, sig=1.0, max_iter=50, errtol=1e-3, switch_iter=4,
                 eta_sca=0.8, eta_fp=0.5, gam_init=None,
                 print_flag=False, calculate_loss=False, prunetol=1e-12):

        if not isinstance(sig, numbers.Real) or sig <= 0:
            raise ValueError("'sig' must be a positive real scalar")
        self.sig = float(sig)
        self.max_iter = int(max_iter)
        self.errtol = float(errtol)
        self.switch_iter = int(switch_iter)
        self.eta_sca = float(eta_sca)
        self.eta_fp = float(eta_fp)
        self.prunetol = float(prunetol)
        self.print_flag = bool(print_flag)
        self.calculate_loss = bool(calculate_loss)
        self.gam_init = gam_init  # default: all zeros

        if self.max_iter < 1:
            raise ValueError("'max_iter' must be at least 1")

    def fit(self, A, SCM):
        """
        Run CL-Hybrid for complex-valued data.

        Parameters
        ----------
        A : (M,N) complex ndarray
        SCM : (M,M) complex ndarray, the Sample Covariance Matrix of the measurement vectors Y.

        Returns
        -------
        gammas : (N,) float array
        iters : int
        err : float
        """
        A = np.asarray(A, dtype=np.complex128)
        SCM = np.asarray(SCM, dtype=np.complex128)
        M, N = A.shape

        if self.gam_init is not None:
            gamma_old = np.array(self.gam_init, dtype=float).reshape(-1)
            if gamma_old.shape[0] != N:
                raise ValueError("gam_init must have shape (N,)")
        else:
            # it is recommended to start with all zeros as below. Only use gam_init if you
            # know gam_init is close to the optimum
            gamma_old = np.zeros(N, dtype=float)

        # ---- hoist everything loop-invariant out of the loop -------------------
        sig = self.sig                # known noise variance
        prune = self.prunetol         # relative pruning threshold for Sigma
        errtol = self.errtol
        calc_loss = self.calculate_loss
        switch_iter = self.switch_iter

        # SCM = S_h S_h^H, computed once (eigh is robust also for rank-deficient SCM)
        lam, U = np.linalg.eigh(SCM)
        keep = lam > max(lam[-1], 0.0) * 1e-12
        S_h = U[:, keep] * np.sqrt(lam[keep])

        potrf, trtri = get_lapack_funcs(('potrf', 'trtri'), (A,))

        eps = 1e-14  # small positive floor; prevents division blow-up if d_n
        # rounds to ~0 due to floating-point error in near-singular cases

        def statistics(gamma):
            """d_n, e_n at gamma, and L, T for the loss."""
            # only columns with gamma_n > prune * max(gamma) enter Sigma; damped
            # iterates never reach exactly zero, so this recovers the sparsity
            idx = np.flatnonzero(gamma > prune * gamma.max())
            W = A[:, idx] * np.sqrt(gamma[idx])
            SigmaY = W @ W.conj().T
            SigmaY.flat[::M + 1] += sig
            L, info = potrf(SigmaY, lower=1, clean=1, overwrite_a=1)
            if info != 0:
                raise LinAlgError("Sigma_gamma is not positive definite")
            Linv, _ = trtri(L, lower=1)                 # L^{-1}, lower triangular
            G = Linv @ A                                # L^{-1} A
            T = Linv @ S_h                              # L^{-1} S_h
            H = T.conj().T @ G                          # S_h^H Sigma^{-1} A
            d_n = (G.real ** 2 + G.imag ** 2).sum(axis=0)    # a_n^H Sigma^{-1} a_n
            e_n = (H.real ** 2 + H.imag ** 2).sum(axis=0)    # a_n^H Sigma^-1 SCM Sigma^-1 a_n
            # d_n is theoretically > 0 (SigmaY is PD); the clamp only guards against
            # floating-point round-off near singularity
            np.maximum(d_n, eps, out=d_n)
            return d_n, e_n, L, T

        loss, KKTloss = [], []
        err = np.inf
        converged = False
        for iters in range(1, self.max_iter + 1):

            d_n, e_n, L, T = statistics(gamma_old)

            # convergence check: projected KKT residual
            delta = np.maximum(gamma_old - (d_n - e_n), 0.0) - gamma_old
            err = np.linalg.norm(delta, ord=np.inf)

            if calc_loss:
                loss.append((T.real ** 2).sum() + (T.imag ** 2).sum()
                            + 2.0 * np.log(L.diagonal().real).sum())
                KKTloss.append(err)

            if err < errtol:
                converged = True
                if self.print_flag:
                    print(f"Projected KKT residual criterion, convergence at it {iters}, err={err:.3e}")
                break

            multiplier = 1.0 / d_n
            r_n = e_n * multiplier
            if iters <= switch_iter:
                residual = np.sqrt(r_n) - 1.0              # SCA proposal
                eta = self.eta_sca
            else:
                residual = r_n - 1.0                       # FP proposal
                eta = self.eta_fp
            gamma_new = gamma_old + multiplier * residual  # update gammas
            np.maximum(gamma_new, 0.0, out=gamma_new)      # the [.]_+ operator

            # Then damping: gamma_old + eta * (gamma_new - gamma_old), in place
            gamma_new -= gamma_old
            gamma_new *= eta
            gamma_new += gamma_old
            np.maximum(gamma_new, 0.0, out=gamma_new)      # round-off only (eta <= 1)
            gamma_old = gamma_new

        if not converged:
            # all iterations used: residual at the returned point, with the same norm
            d_n, e_n, _, _ = statistics(gamma_old)
            err = np.linalg.norm(np.maximum(gamma_old - (d_n - e_n), 0.0) - gamma_old,
                                 ord=np.inf)
            if self.print_flag:
                print(f"Solution not converged. Error: {err:.6f}")

        self.gammas = gamma_old
        self.iters = iters
        self.err = err
        self.loss = loss
        self.KKTloss = KKTloss

        return gamma_old, iters, err
