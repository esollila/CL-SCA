import numpy as np
from ..base import CLobject
import numbers
from scipy.linalg import get_lapack_funcs, LinAlgError


class CWO(CLobject):
    """
    CWO: coordinate-wise optimization for the covariance-learning problem,
    using Sherman-Morrison rank-one updates within a sweep.

    One iteration = one full sweep over the N coordinates. At the start of
    every sweep Sigma^{-1} is rebuilt from a Cholesky factorization; this
    (i) supplies the statistics needed for the projected KKT residual, and
    (ii) removes the drift that repeated rank-one updates would otherwise
    accumulate.

    Implementation. With Sigma = L L^H and SCM = S_h S_h^H (the latter computed once),
        G = L^{-1} A,   T = L^{-1} S_h,
        d_n = ||g_n||^2,   e_n = ||T^H g_n||^2,
        L(gamma) = tr(Sigma^{-1} SCM) + log|Sigma| = ||T||_F^2 + 2 sum_i log L_ii.
    The once-per-sweep statistics use the same linear algebra as the CL methods:
    L^{-1} is formed explicitly, G, T and T^H G are matrix products, the loss comes
    for free, and Sigma^{-1} = L^{-H} L^{-1} seeds the sweep. 
    Columns with negligible gamma_n are left out of Sigma (see prunetol).
    The iterates agree (up to round-off) with those of the cho_factor/cho_solve
    version. The sweep itself is inherently sequential; its per-coordinate work
    (two M x M matrix-vector products and one rank-one update) is executed by
    NumPy, so for small M a substantial part of its runtime is interpreter
    overhead.

    Parameters
    ----------
    sig : float, positive
        Noise variance (sigma^2).
    max_iter : int, positive
        Maximum number of sweeps.
    errtol : float, positive
        Tolerance on the projected KKT residual
            || [gamma - grad ell(gamma)]_+ - gamma ||_inf ,
        evaluated at the start of each sweep. The same criterion is used by
        every compared method, so the iteration counts are comparable.
    print_flag : bool
        If True, print convergence progress.
    gam_init : array-like, optional
        Initial gammas of shape (N,). If None, initialized to all zeros.
    calculate_loss : bool
        If True, record the loss and the KKT residual at each sweep.
    randomize_order : bool
        If True, draw a fresh coordinate order for every sweep.
    seed : int or None
        Seed of the internal generator used for the coordinate order.
    prunetol : float, optional
        Columns with gamma_n <= prunetol * max(gamma) are left out when Sigma_gamma
        is formed (the gammas themselves are still updated). Default 1e-12 changes
        Sigma only at round-off level; set 0 to use all nonzero gammas.
    """

    def __init__(self, sig=1.0, max_iter=50, errtol=1e-3, print_flag=False,
                 gam_init=None, calculate_loss=False, randomize_order=True,
                 seed=None, prunetol=1e-12):
        if not isinstance(sig, numbers.Real) or sig <= 0:
            raise ValueError("'sig' must be a positive real scalar")
        if not isinstance(max_iter, numbers.Integral) or max_iter <= 0:
            raise ValueError("'max_iter' must be a positive integer")
        if not isinstance(errtol, numbers.Real) or errtol < 0:
            raise ValueError("'errtol' must be a nonnegative real scalar")
        self.sig = float(sig)
        self.max_iter = int(max_iter)
        self.errtol = float(errtol)
        self.print_flag = bool(print_flag)
        self.gam_init = gam_init
        self.calculate_loss = bool(calculate_loss)
        self.randomize_order = bool(randomize_order)
        self.seed = seed
        self.prunetol = float(prunetol)

    def fit(self, A, SCM):
        """
        Parameters
        ----------
        A : (M,N) complex ndarray
            Dictionary/pilot matrix; M is the measurement dimension.
        SCM : (M,M) complex ndarray
            Sample covariance matrix.

        Returns
        -------
        gammas : (N,) float array
        iters : int
            Number of sweeps performed (= number of residual evaluations).
        err : float
            Projected KKT residual at the returned gammas.
        """
        A = np.asarray(A, dtype=np.complex128)
        SCM = np.asarray(SCM, dtype=np.complex128)
        M, N = A.shape

        if self.gam_init is not None:
            gamma = np.array(self.gam_init, dtype=float).reshape(-1)
            if gamma.shape[0] != N:
                raise ValueError("gam_init must have shape (N,)")
        else:
            gamma = np.zeros(N, dtype=float)

        # ---- hoist everything loop-invariant out of the loop -------------------
        rng = np.random.default_rng(self.seed)
        sig = self.sig
        prune = self.prunetol
        errtol = self.errtol
        calc_loss = self.calculate_loss
        A_rows = np.ascontiguousarray(A.T)   # A_rows[n] = a_n, contiguous
        vdot, outer = np.vdot, np.outer      # local names: cheaper lookups in the sweep

        # SCM = S_h S_h^H, computed once (eigh is robust also for rank-deficient SCM)
        lam, U = np.linalg.eigh(SCM)
        keep = lam > max(lam[-1], 0.0) * 1e-12
        S_h = U[:, keep] * np.sqrt(lam[keep])

        potrf, trtri = get_lapack_funcs(('potrf', 'trtri'), (A,))

        eps = 1e-14  # small positive floor; prevents division blow-up if d_n
        # rounds to ~0 due to floating-point error in near-singular cases

        def statistics(gamma):
            """d_n, e_n at gamma, and L, T, L^{-1} for the loss and the sweep."""
            # only columns with gamma_n > prune * max(gamma) enter Sigma
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
            return d_n, e_n, L, T, Linv

        loss, KKTloss = [], []
        err = np.inf
        converged = False
        for iters in range(1, self.max_iter + 1):

            # ---- statistics and residual at `gamma` -----------------------------
            d_all, e_all, L, T, Linv = statistics(gamma)
            delta = np.maximum(gamma - (d_all - e_all), 0.0) - gamma
            err = np.linalg.norm(delta, ord=np.inf)

            if calc_loss:
                loss.append((T.real ** 2).sum() + (T.imag ** 2).sum()
                            + 2.0 * np.log(L.diagonal().real).sum())
                KKTloss.append(err)

            if err < errtol:
                converged = True
                if self.print_flag:
                    print(f"Projected KKT residual criterion, convergence at "
                          f"sweep {iters}, err={err:.3e}")
                break

            # Sigma^{-1} = L^{-H} L^{-1} for the sweep, from the same factorization
            SigmaYinv = Linv.conj().T @ Linv
            SigmaYinv = (SigmaYinv + SigmaYinv.conj().T) / 2

            # ---- one sweep of exact coordinate minimizations --------------------
            order = rng.permutation(N) if self.randomize_order else range(N)
            g = gamma.tolist()               # Python floats: cheap scalar access
            for n in order:
                a = A_rows[n]
                b = SigmaYinv @ a
                d_n = vdot(a, b).real
                if d_n < eps:
                    d_n = eps
                e_n = vdot(b, SCM @ b).real

                step = (e_n - d_n) / (d_n * d_n)
                g_n = g[n]
                if step < -g_n:              # keep gamma_n >= 0
                    step = -g_n
                if step != 0.0:
                    g[n] = g_n + step
                    # Sherman-Morrison: Sigma <- Sigma + step * a a^H
                    SigmaYinv -= outer(b * (step / (1.0 + step * d_n)), b.conj())
            gamma = np.array(g)

        if not converged:
            # all sweeps used: residual at the final iterate
            d_all, e_all, _, _, _ = statistics(gamma)
            err = np.linalg.norm(
                np.maximum(gamma - (d_all - e_all), 0.0) - gamma, ord=np.inf)
            if self.print_flag:
                print(f"Solution not converged. Error: {err:.6f}")

        self.gammas = gamma
        self.iters = iters
        self.err = err
        self.loss = loss
        self.KKTloss = KKTloss

        return gamma, iters, err
