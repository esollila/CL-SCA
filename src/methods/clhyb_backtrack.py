import numpy as np
from ..base import CLobject
import numbers
from collections import deque
from scipy.linalg import get_lapack_funcs, LinAlgError


LOSS_RTOL = 1e-14   # relative resolution assumed for the computed loss


class CLHybridBacktrack(CLobject):
    """
    CL-Hyb-BT: hybrid SCA/FP iteration with backtracking step size.

    Phase 1 (k <= switch_iter): SCA proposal
        hat_gamma_n = [gamma_n + sqrt(e_n/d_n^3) - 1/d_n]_+
    Phase 2 (k >  switch_iter): FP proposal
        hat_gamma_n = [gamma_n + e_n/d_n^2 - 1/d_n]_+

    In both phases gamma^{k+1} = gamma^k + eta_k p^k with p^k = hat_gamma - gamma^k,
    and eta_k is found by backtracking: starting from a trial step eta_k <= eta, the
    step is multiplied by beta until the sufficient-decrease condition

        L(gamma^k + eta_k p^k) <= L(gamma^k) - (dmin_k^2 / 2) * eta_k * ||p^k||^2

    is satisfied, where dmin_k = min_{s <= k} min_n d_n(gamma^s) is the running minimum
    of the curvature statistics. Both proposals satisfy
    grad L^T p <= -sum_n d_n^2 p_n^2, so the test is passed for every
    eta_k <= d_min^2 / L_grad in either phase, every accepted step satisfies the
    sufficient-decrease inequality of the stationarity-rate theorem, and no step-size
    constant has to be tuned for either phase.

    Near a KKT point the FP displacement is about twice the SCA displacement, so FP
    with step eta acts locally like SCA with step 2*eta. At the switch the certified
    step size is therefore halved, so the first FP trial starts at the same effective
    step as the last accepted SCA step.

    Close to stationarity the predicted change of the loss, eta_k |grad L^T p^k|,
    drops below the resolution of the computed loss (~1e-14 relative), and the test
    is decided by round-off. In that
    regime the step size is frozen at the smallest of the last ten certified step
    sizes, and halved (at most four times) whenever ||p^k|| grows.

    Cost. The Cholesky factor of Sigma_gamma computed to evaluate the loss at the
    accepted trial point is reused by the update of the next iteration, so an
    iteration without backtracking costs one Cholesky factorization, as in CLSCA.
    Each rejected trial adds one factorization (no product with A).

    Implementation. With Sigma = L L^H and SCM = S_h S_h^H (the latter computed once),
        G = L^{-1} A,   T = L^{-1} S_h,
        d_n = ||g_n||^2,   e_n = ||T^H g_n||^2,
        L(gamma) = tr(Sigma^{-1} SCM) + log|Sigma| = ||T||_F^2 + 2 sum_i log L_ii.
    L^{-1} (M x M, cheap) is formed explicitly so that G and T are matrix products,
    which BLAS executes faster than triangular solves. A trial point costs one
    Cholesky factorization and the M x M products for T and the loss; the product
    with A is only formed for the accepted point. Columns with negligible gamma_n
    are left out of Sigma (see prunetol).

    Parameters
    ----------
    sig : float, optional
        Noise variance (sigma^2). Default is 1.0.
    max_iter : int, optional
        Maximum number of iterations. Default is 50.
    errtol : float, optional
        Tolerance on the projected KKT residual
            || [gamma - grad L(gamma)]_+ - gamma ||_inf ,
        the same stopping criterion as in CLSCA. Default is 1e-3.
    switch_iter : int, optional
        Number of initial SCA iterations before switching to FP proposals.
        Default is 4.
    eta : float, optional
        Largest trial step, in (0, 1]. Default is 1.0 (full step).
    beta : float, optional
        Backtracking contraction factor, eta <- beta * eta, in (0, 1). Default is 0.5.
    max_backtracks : int, optional
        Maximum number of contractions per iteration; if exceeded, the smallest trial
        step is accepted. Default is 30.
    warm_start : bool, optional
        If True, an iteration starts from the last certified step eta_cert, and tries
        the larger step min(eta, eta_cert / beta) only after two consecutive
        iterations whose first trial was accepted. This avoids repeating the same
        rejection in every iteration. If False, every iteration starts from eta.
        Default is True.
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

    Attributes (after fit)
    ----------------------
    gammas, iters, err, loss, KKTloss : as in CLSCA.
    etas : list of accepted step sizes eta_k.
    n_trials : total number of trial points evaluated (= Cholesky factorizations
        inside the line search), a measure of the backtracking overhead.
    dmin : running minimum dmin_k at termination.
    """

    def __init__(self, sig=1.0, max_iter=50, errtol=1e-3, switch_iter=4, eta=1.0, beta=0.5,
                 max_backtracks=30, warm_start=True, gam_init=None,
                 print_flag=False, calculate_loss=False, prunetol=1e-12):

        if not isinstance(sig, numbers.Real) or sig <= 0:
            raise ValueError("'sig' must be a positive real scalar")
        if not (0.0 < eta <= 1.0):
            raise ValueError("'eta' must be in (0, 1]")
        if not (0.0 < beta < 1.0):
            raise ValueError("'beta' must be in (0, 1)")
        self.sig = float(sig)
        self.max_iter = int(max_iter)
        self.errtol = float(errtol)
        self.switch_iter = int(switch_iter)
        self.eta = float(eta)
        self.beta = float(beta)
        self.max_backtracks = int(max_backtracks)
        self.warm_start = bool(warm_start)
        self.prunetol = float(prunetol)
        self.print_flag = bool(print_flag)
        self.calculate_loss = bool(calculate_loss)
        self.gam_init = gam_init  # default: all zeros

        if self.max_iter < 1:
            raise ValueError("'max_iter' must be at least 1")

    def fit(self, A, SCM):
        """
        Run CL-Hyb-BT for complex-valued data.

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
            np.maximum(gamma_old, 0.0, out=gamma_old)
        else:
            # it is recommended to start with all zeros as below. Only use gam_init if you
            # know gam_init is close to the optimum
            gamma_old = np.zeros(N, dtype=float)

        # ---- hoist everything loop-invariant out of the loop -------------------
        sig = self.sig
        prune = self.prunetol         # relative pruning threshold for Sigma
        errtol = self.errtol
        calc_loss = self.calculate_loss
        eta_max, beta = self.eta, self.beta
        max_bt = self.max_backtracks
        warm_start = self.warm_start
        switch_iter = self.switch_iter

        # SCM = S_h S_h^H, computed once (eigh is robust also for rank-deficient SCM)
        lam, U = np.linalg.eigh(SCM)
        keep = lam > max(lam[-1], 0.0) * 1e-12
        S_h = U[:, keep] * np.sqrt(lam[keep])

        potrf, trtri = get_lapack_funcs(('potrf', 'trtri'), (A,))

        def factorize(gamma):
            """L^{-1} for Sigma_gamma = L L^H, T = L^{-1} S_h, and the loss."""
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
            T = Linv @ S_h                              # L^{-1} S_h
            loss_val = ((T.real ** 2).sum() + (T.imag ** 2).sum()
                        + 2.0 * np.log(L.diagonal().real).sum())
            return Linv, T, loss_val

        def statistics(Linv, T):
            """d_n = a_n^H Sigma^{-1} a_n and e_n = a_n^H Sigma^{-1} SCM Sigma^{-1} a_n."""
            G = Linv @ A                                # L^{-1} A
            H = T.conj().T @ G                          # S_h^H Sigma^{-1} A
            d_n = (G.real ** 2 + G.imag ** 2).sum(axis=0)
            e_n = (H.real ** 2 + H.imag ** 2).sum(axis=0)
            return d_n, e_n

        loss, KKTloss, etas = [], [], []
        err = np.inf
        eps = 1e-14  # small positive floor; prevents division blow-up if d_n
        # rounds to ~0 due to floating-point error in near-singular cases
        n_trials = 0
        dmin_run = np.inf                    # running minimum of min_n d_n(gamma^s)
        eta_cert = eta_max                   # last step size certified by the test
        recent_cert = deque([eta_max], maxlen=10)   # recently certified step sizes
        eta_frozen = None                    # step size while the test is uninformative
        pnorm_old = np.inf
        streak = 2                           # consecutive first-trial acceptances

        Linv, T, loss_old = factorize(gamma_old)
        converged = False
        for iters in range(1, self.max_iter + 1):

            # statistics at gamma^k, from the factor of the accepted trial point
            d_n, e_n = statistics(Linv, T)

            # d_n is theoretically > 0 (SigmaY is PD); the clamp only guards against
            # floating-point round-off near singularity
            np.maximum(d_n, eps, out=d_n)
            dmin_run = min(dmin_run, d_n.min())

            # convergence check: projected KKT residual
            delta = np.maximum(gamma_old - (d_n - e_n), 0.0) - gamma_old
            err = np.linalg.norm(delta, ord=np.inf)

            if calc_loss:
                loss.append(loss_old)
                KKTloss.append(err)

            if err < errtol:
                converged = True
                if self.print_flag:
                    print(f"Projected KKT residual criterion, convergence at it {iters}, err={err:.3e}")
                break

            # undamped proposal and search direction
            multiplier = 1.0 / d_n
            r_n = e_n * multiplier
            if iters <= switch_iter:
                residual = np.sqrt(r_n) - 1.0              # SCA proposal
            else:
                residual = r_n - 1.0                       # FP proposal
                if iters == switch_iter + 1:
                    # FP step ~ 2 x SCA step near a KKT point: keep the effective step
                    eta_cert *= 0.5
                    recent_cert = deque([e * 0.5 for e in recent_cert], maxlen=10)
            gamma_new = gamma_old + multiplier * residual  # update gammas
            np.maximum(gamma_new, 0.0, out=gamma_new)      # the [.]_+ operator
            p = gamma_new - gamma_old
            pp = p @ p
            slope = (d_n - e_n) @ p                        # grad(loss)^T p  (< 0)

            # ---- backtracking line search ------------------------------------------
            # required decrease per unit step:  L(gamma + eta p) <= L(gamma) - eta * dec
            dec = 0.5 * dmin_run ** 2 * pp
            # The test is informative only if the predicted change of the loss,
            # eta * |grad(loss)^T p|, exceeds the resolution ftol of the computed loss.
            # Otherwise (close to stationarity) the step size is frozen at the
            # smallest recently certified value, and halved (at most four times)
            # whenever ||p|| grows.
            ftol = LOSS_RTOL * (1.0 + abs(loss_old))
            pnorm = np.sqrt(pp)
            if not warm_start:
                eta = eta_max
            elif streak >= 2:
                eta = min(eta_max, eta_cert / beta)     # try a larger step
            else:
                eta = eta_cert
            informative = -eta * slope > ftol
            if not informative:
                if eta_frozen is None:
                    eta_frozen = min(recent_cert)
                    eta_floor = eta_frozen * beta ** 4   # at most four halvings
                elif pnorm > pnorm_old:
                    eta_frozen = max(eta_frozen * beta, eta_floor)
                eta = eta_frozen
            else:
                eta_frozen = None
            pnorm_old = pnorm

            for nb in range(max_bt + 1):
                if eta == 1.0:
                    gamma_trial = gamma_new
                else:
                    gamma_trial = gamma_old + eta * p
                    np.maximum(gamma_trial, 0.0, out=gamma_trial)  # round-off only
                Linv, T, loss_trial = factorize(gamma_trial)
                n_trials += 1
                if not informative or nb == max_bt:
                    break
                if loss_trial <= loss_old - eta * dec:      # sufficient decrease
                    eta_cert = eta
                    recent_cert.append(eta)
                    break
                eta *= beta
                # once the predicted change is below ftol, the next trial is accepted
                informative = -eta * slope > ftol

            streak = streak + 1 if nb == 0 else 0
            etas.append(eta)
            gamma_old, loss_old = gamma_trial, loss_trial

        if not converged:
            # all iterations used: residual at the returned point, from the cached factor
            d_n, e_n = statistics(Linv, T)
            np.maximum(d_n, eps, out=d_n)
            err = np.linalg.norm(np.maximum(gamma_old - (d_n - e_n), 0.0) - gamma_old,
                                 ord=np.inf)
            if self.print_flag:
                print(f"Solution not converged. Error: {err:.6f}")

        self.gammas = gamma_old
        self.iters = iters
        self.err = err
        self.loss = loss
        self.KKTloss = KKTloss
        self.etas = etas
        self.n_trials = n_trials
        self.dmin = dmin_run

        return gamma_old, iters, err
