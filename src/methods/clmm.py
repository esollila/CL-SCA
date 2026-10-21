import numpy as np
from ..base import CLobject
import numbers
from scipy.linalg import get_lapack_funcs, LinAlgError

class CLMM(CLobject):
    """
    CLMM: Covariance-learning MM algorithm for sparse signal recovery.

    Update (projected onto gamma >= 0):
        gamma_n <- [gamma_n + (gamma_n + offset) * (sqrt(e_n/d_n) - 1)]_+
    where offset is either  (sig/||A||_F^2) of (sig/||A|^2_2), where the former is 
    obtained using argument offset="loose" and the latter using offset = "tight". 
    
    Implementation. With Sigma = L L^H and SCM = S_h S_h^H (the latter computed once),
        G = L^{-1} A,   T = L^{-1} S_h,
        d_n = ||g_n||^2,   e_n = ||T^H g_n||^2,
        L(gamma) = tr(Sigma^{-1} SCM) + log|Sigma| = ||T||_F^2 + 2 sum_i log L_ii.
    L^{-1} (M x M, cheap) is formed explicitly so that G, T and T^H G are matrix
    products, which BLAS executes faster than triangular solves; this replaces
    cho_solve (two triangular solves with A) and SCM @ B, and the loss comes for
    free. Columns with negligible gamma_n are left out of Sigma (see prunetol).
    The iterates agree (up to round-off) with those of the cho_factor/cho_solve
    version, and the linear algebra is identical to that of CLSCA, CLHybrid and the
    backtracking variants, so runtimes are directly comparable.

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
    offset : {'loose', 'tight'}, optional
        'loose' uses offset= (sig/||A||_F^2) and tight uses offset= (sig/||A|^2_2). 
        Default is 'tight'.
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

    def __init__(self, sig=1.0, max_iter=50, errtol=1e-3, gam_init=None,
                 print_flag=False, calculate_loss=False, prunetol=1e-12,offset = "tight"):

        if not isinstance(sig, numbers.Real) or sig <= 0:
            raise ValueError("'sig' must be a positive real scalar")
        if offset not in ("loose", "tight"):
            raise ValueError("'offset' must be 'loose' or 'tight'")
        self.sig = float(sig)
        self.max_iter = int(max_iter)
        self.errtol = float(errtol)
        self.prunetol = float(prunetol)
        self.print_flag = bool(print_flag)
        self.calculate_loss = bool(calculate_loss)
        self.gam_init = gam_init  # default: all zeros
        

        self.offset = offset
        if self.max_iter < 1:
            raise ValueError("'max_iter' must be at least 1")

    def fit(self, A, SCM):
        """
        Run CLMM for complex-valued data.

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
        if self.offset == "loose":
            offset = sig / np.sum(A.real ** 2 + A.imag ** 2)   # sig / ||A||_F^2
        else:
            # else use the tightest value 
            lam_max = np.linalg.eigvalsh(A @ A.conj().T)[-1]   # eigvalsh sorts ascending
            offset = sig / lam_max
        
        # SCM = S_h S_h^H, computed once (eigh is robust also for rank-deficient SCM)
        lam, U = np.linalg.eigh(SCM)
        keep = lam > max(lam[-1], 0.0) * 1e-12
        S_h = U[:, keep] * np.sqrt(lam[keep])

        potrf, trtri = get_lapack_funcs(('potrf', 'trtri'), (A,))

        eps = 1e-14  # small positive floor; prevents division blow-up if d_n
        # rounds to ~0 due to floating-point error in near-singular cases

        def statistics(gamma):
            """d_n, e_n at gamma, and L, T for the loss."""
            # only columns with gamma_n > prune * max(gamma) enter Sigma; the
            # iterates rarely reach exactly zero, so this recovers the sparsity
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

            residual = np.sqrt(e_n / d_n) - 1.0            # sqrt(r_n) - 1
            gamma_new = gamma_old + (gamma_old + offset) * residual   # MM update
            np.maximum(gamma_new, 0.0, out=gamma_new)      # the [.]_+ operator
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
