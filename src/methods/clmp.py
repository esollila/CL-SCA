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

    Parameters:
    ----------
        K : int
            Number of non-zero sources to select (must be positive).
        sig : float
            Known noise variance (positive real scalar).
    """
    
    def __init__(self,K=None,sig=1.0):
        
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
        # Validate inputs
        A = np.asarray(A, dtype=np.complex128)
        SCM = np.asarray(SCM, dtype=np.complex128)
        M, N = A.shape
                
        if self.K == None:
            raise ValueError("positional argument 'K' missing")
            
        if not isinstance(self.K, numbers.Integral) or self.K <= 0:
            raise ValueError("'K' must be a positive integer")

        if self.K > N:
            raise ValueError(f"'K'={self.K} cannot exceed the number of atoms N={N}")
        # Initialize
        Ilocs = np.full(self.K, -1, dtype=int)
        B = (1.0 / float(self.sig)) * A  # B approximates Sigma^{-1} A initially (Sigma = sig * I)
        gam = np.zeros(self.K, dtype=float)

        eps = 1e-14  # small positive floor; prevents division blow-up if gam_den 
        # rounds to ~0 due to floating-point error in near-singular cases
        
        available = np.ones(N, dtype=bool)
        # Greedy selection loop
        for k in range(self.K):
            
            # 1) Compute gamma and objective fk for all atoms
            # gamma_num[m] = real(b_m^H SCM b_m)
            SCM_B = SCM @ B                      # (M, N)
            gamma_num = np.real(np.sum(np.conj(B) * SCM_B, axis=0))  # (N,)

            # gamma_denum[m] = real(a_m^H b_m) with b_m = column m of B
            gamma_den = np.real(np.sum(np.conj(A) * B, axis=0))   # (N,)
            # gam_den = Re(a_n^H SigmaY^-1 a_n) is theoretically > 0 (SigmaY is PD);
            # clamp only guards against floating-point round-off near singularity
            np.maximum(gamma_den, eps, out=gamma_den)
            gamma = np.maximum(gamma_num / (gamma_den**2) - 1.0 / gamma_den, 0.0)  
            tmp = gamma * gamma_den
            fk = np.log1p(tmp) - tmp  # log(1+tmp) - tmp

            # # 2) Select index minimizing fk among not-yet-selected
            # selected = set(Ilocs[:k].tolist()) - {-1}
            # candidates = [m for m in range(N) if m not in selected]
            # if len(candidates) == 0:
            #     # Should not happen if K <= N
            #     break
            
            # # Argmin over candidates
            # cand_fk = fk[candidates]
            # min_idx_local = int(np.argmin(cand_fk))
            # indx = candidates[min_idx_local]
            # Ilocs[k] = indx
            
            fk_masked = np.where(available, fk, np.inf)
            indx = int(np.argmin(fk_masked))
            Ilocs[k] = indx
            available[indx] = False

            # 3) Store gamma at selected index
            gam[k] = float(gamma[indx])

            # 4) Update B = Sigma^{-1} A using Sherman–Morrison rank-1 update
            if k < self.K - 1 and gam[k] > 0:
                a = A[:, indx]       # (L,)
                b = B[:, indx]       # (L,)
                denom = 1.0 + gam[k] * float(np.real(np.vdot(a, b)))  # vdot = a^H b
                denom = max(denom, eps) 

                # a^H B -> shape (N,)
                aH_B = a.conj().T @ B
                # Rank-1 update: B <- B - (gam/(1+gam*Re(a^H b))) * b (a^H B)
                factor = gam[k] / denom
                B = B - factor * (b[:, None] @ aH_B[None, :])

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
        Am = A[:,self.supp]
        gam0 = self.gam        

        Am_scaled = Am * gam0[np.newaxis, :]        # scales each column of Am by gam0[k], O(M*K)
        SigmaY = Am_scaled @ Am.conj().T
        SigmaY[np.diag_indices_from(SigmaY)] += self.sig   # add sig*I without building np.eye(M)
        
        # compute B = SigmaY^{-1} A without explicitly inverting SigmaY
        cF = cho_factor(SigmaY, lower=True, check_finite=False, overwrite_a=False)
        B = cho_solve(cF, Am, check_finite=False)
        W = B*gam0
        
        self.signal_estimates_ = np.zeros((N, Y.shape[1]), dtype=Y.dtype)
        self.signal_estimates_[self.supp,:] = W.conj().T @ Y
        self.weights_ = W
 