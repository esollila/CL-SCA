import numpy as np
import numbers
from scipy.linalg import cho_factor, cho_solve


class CLMP:
    """
    CLMP:  Covariance Learning Matching Pursuit (CL-MP) for sparse support recovery.

    See reference:
        Marata, Ollila and Alves: "Activity Detection for Massive Random Access
        Using Covariance-Based Matching Pursuit", IEEE Transactions on Vehicular Technology,
        vol. 74, pp-17292-17303,  2025.

    Implementation. Both B = Sigma^{-1} A and C = SCM B are maintained by rank-one
    (Sherman-Morrison) updates, so that after a single initial product SCM @ A each
    greedy step costs O(M N) operations instead of the O(M^2 N) of recomputing
    SCM @ B. 

    Parameters:
    ----------
        K : int
            Number of non-zero sources to select (must be positive).
        sig : float
            Known noise variance (positive real scalar).
    """

    def __init__(self, K=None, sig=1.0):

        if not isinstance(sig, numbers.Real) or sig <= 0:
            raise ValueError("'sig' must be a positive real scalar")

        self.K = K
        self.sig = float(sig)

    def fit(self, A, SCM):
        """
        Parameters
        ----------
        A : np.ndarray (complex), shape (M, N)
            Dictionary (sensing) matrix.
        SCM : np.ndarray (complex), shape (M, M)
            Sample covariance matrix (SCM).

        Returns
        -------
        Ilocs : np.ndarray of int, shape (K,)
            Estimated locations of support indices of the non-zero signal powers.
        """
        A = np.asarray(A, dtype=np.complex128)
        SCM = np.asarray(SCM, dtype=np.complex128)
        M, N = A.shape
        K = self.K

        if K is None:
            raise ValueError("positional argument 'K' missing")
        if not isinstance(K, numbers.Integral) or K <= 0:
            raise ValueError("'K' must be a positive integer")
        if K > N:
            raise ValueError(f"'K'={K} cannot exceed the number of atoms N={N}")

        # Initialize: at gamma = 0, Sigma = sig*I, so B = Sigma^{-1} A = A / sig exactly
        Ilocs = np.full(K, -1, dtype=int)
        gam = np.zeros(K, dtype=float)
        B = A / self.sig                     # Sigma^{-1} A           (M, N)
        C = SCM @ B                          # SCM Sigma^{-1} A       (M, N)

        eps = 1e-14  # small positive floor; prevents division blow-up if gamma_den
        # rounds to ~0 due to floating-point error in near-singular cases

        available = np.ones(N, dtype=bool)
        # Greedy selection loop
        for k in range(K):

            # 1) gamma and objective fk for all atoms
            # gamma_num[n] = Re(b_n^H SCM b_n),  gamma_den[n] = Re(a_n^H b_n)
            gamma_num = (B.real * C.real + B.imag * C.imag).sum(axis=0)
            gamma_den = (A.real * B.real + A.imag * B.imag).sum(axis=0)
            # gamma_den is theoretically > 0 (Sigma is PD); the clamp only guards
            # against floating-point round-off near singularity
            np.maximum(gamma_den, eps, out=gamma_den)
            gamma = np.maximum(gamma_num / gamma_den ** 2 - 1.0 / gamma_den, 0.0)
            tmp = gamma * gamma_den
            fk = np.log1p(tmp) - tmp           # log(1 + tmp) - tmp

            # 2) select the index minimizing fk among the not-yet-selected atoms
            fk[~available] = np.inf
            indx = int(np.argmin(fk))
            Ilocs[k] = indx
            available[indx] = False

            # 3) store gamma at the selected index
            gam[k] = gamma[indx]

            # 4) rank-one (Sherman-Morrison) updates of B and C = SCM B
            if k < K - 1 and gam[k] > 0:
                a = A[:, indx]
                b = B[:, indx].copy()           # copies: B and C are updated in place
                s = C[:, indx].copy()           # = SCM b
                denom = max(1.0 + gam[k] * gamma_den[indx], eps)   # 1 + gam a^H b
                aH_B = a.conj() @ B             # (N,)
                factor = gam[k] / denom
                B -= np.outer(factor * b, aH_B)
                C -= np.outer(factor * s, aH_B)

        self.supp = Ilocs
        self.gam = gam
        self.loss = []
        return Ilocs

    def support(self, K):
        if K > len(self.supp):
            raise ValueError(f"Requested K={K} exceeds the fitted support size {len(self.supp)}")
        return self.supp[:K]

    def estimate_sources(self, A, Y):

        if not hasattr(self, "supp"):
            raise RuntimeError("Call fit() before estimate_sources()")

        M, N = A.shape
        Am = A[:, self.supp]
        gam0 = self.gam

        Am_scaled = Am * gam0[np.newaxis, :]        # scales each column of Am by gam0[k], O(M*K)
        SigmaY = Am_scaled @ Am.conj().T
        SigmaY[np.diag_indices_from(SigmaY)] += self.sig   # add sig*I without building np.eye(M)

        # compute B = SigmaY^{-1} Am without explicitly inverting SigmaY
        cF = cho_factor(SigmaY, lower=True, check_finite=False, overwrite_a=False)
        B = cho_solve(cF, Am, check_finite=False)
        W = B * gam0

        self.signal_estimates_ = np.zeros((N, Y.shape[1]), dtype=Y.dtype)
        self.signal_estimates_[self.supp, :] = W.conj().T @ Y
        self.weights_ = W
