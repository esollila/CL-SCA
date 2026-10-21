import numpy as np
import numbers
from scipy.linalg import cho_factor, cho_solve

class CLobject():
    # def __init__(self):
    #     self.verbose = False
    
    def support(self, K):
        """
        Get the indices of the K largest entries in the estimated gammas.

        Parameters
        ----------
        K : int
            Number of largest entries to select.

        Returns
        -------
        Ilocs : np.ndarray
            Indices of the K largest entries in gammas, sorted ascending.
        """
        if not hasattr(self, 'gammas'):
            raise ValueError("fit method must be called before support extraction.")
        
        Ilocs = np.argsort(-self.gammas)[:K]
        Ilocs = np.sort(Ilocs)
        self.supp = Ilocs
        return Ilocs
        
    def compute_loss(self,SigmaY,RY):
        """ 
        Function to compute the negative log-likelihood loss for given SigmaY and RY.
        
            SigmaY: (M, M) positive definite
            RY:     (M, M) positive semidefinite
        """
        
        # X = np.linalg.solve(SigmaY, RY)
        # f_tmp  = np.real(np.trace(X)) # should be the same as f
        # g_tmp = np.log(np.linalg.det(SigmaY)) # should be the same as g

        c, lower = cho_factor(SigmaY, overwrite_a=False, check_finite=False)
        assert np.all(np.diag(c) > 0)
        X = cho_solve((c, lower), RY)
        f  = np.real(np.trace(X))
        g = 2.0 * np.sum(np.real(np.log(np.diag(c))))
        loss = f+g
        self.loss_ = loss.item()
        
        return self.loss_
    
    def estimate_sources(self, A, Y,supp=None):
       
        N, M = A.shape 
        
        #if self.supp is None:
        #    raise ValueError("Support indices not set. Call support() method first.")

        if supp is None:
            supp = np.arange(M) # use all atoms if support not provided
            
        Am = A[:,supp]
        gam0 = self.gammas[supp]
            
        SigmaY = self.sig * np.eye(N) + Am @ np.diag(gam0) @ Am.conj().T # covariance matrix

        # compute B = SigmaY^{-1} A without explicitly inverting SigmaY
        cF = cho_factor(SigmaY, lower=True, check_finite=False, overwrite_a=False)
        B = cho_solve(cF, Am, check_finite=False)

        W = B*gam0
        if len(supp) < M:
            self.signal_estimates_ = np.zeros((M, Y.shape[1]), dtype=Y.dtype)
            self.signal_estimates_[supp,:] = W.conj().T @ Y
        else:   
            self.signal_estimates_ = W.conj().T @ Y
            
        self.weights_ = W
        
        return self.signal_estimates_
    
    def compute_gamma_init(self,A, RY):
        """
        Python implementation of CWOpt (coordinate-wise optimization with
        Sherman–Morrison rank-1 updates) using 1 cycle of coordinate descent. Used for initialization. 

        Parameters
        ----------
        A : np.ndarray (complex)
            Dictionary matrix of shape (N, M).
        RY : np.ndarray (complex)
            Sample covariance matrix of shape (N, N).

        Returns
        -------
        gamma : np.ndarray (float)
            Length-M vector of estimated source powers.
        """
        # Ensure complex dtype for matrices; gamma stays real
        N,M = A.shape
        gamma = np.zeros(M, dtype=float)
        SigmaYinv = (1.0 / self.sig) * np.eye(N, dtype=np.complex128)
        for indx in range(M):
         
            a = A[:, indx]                 # (N,)
            b = SigmaYinv @ a              # (N,)
            # c = real(a^H b)
            c = float(np.real(np.vdot(a, b)))  # vdot does conj(a) @ b
            if c < 1e-12:
                c = 1e-12
            # tmp = (real(b^H RY b) - c) / c^2
            Ry_b = RY @ b
            bh_Ry_b = float(np.real(np.vdot(b, Ry_b)))
            tmp = (bh_Ry_b - c) / (c * c)

            d = max(tmp,0.0)  # ensure nonnegativity
            gamma[indx] = d

            # Sherman–Morrison rank-1 update when d != 0:
            # SigmaYinv <- SigmaYinv - (d / (1 + d*c)) * b b^H
            if d != 0.0:
                factor = d / (1.0 + d * c)
                SigmaYinv -= factor * np.outer(b, b.conj())

        return gamma     

    def compute_KKT_loss(self, A, gam, SCM):
        
        M, N = A.shape
        sig = self.sig
        W = A * np.sqrt(gam)
        SigmaY = W @ W.conj().T
        SigmaY.flat[::M + 1] += sig
        
        # Solve SigmaY B = A for B (M, N) without forming the inverse.
        # SigmaY is rebuilt every iteration, so it is safe to overwrite it in place.
        cF = cho_factor(SigmaY, lower=True, check_finite=False, overwrite_a=True)
        B = cho_solve(cF, A, check_finite=False)

        # Numerator and denominator for the gamma update
        SCM_B = SCM @ B
        e_n = np.einsum('ij,ij->j', B.conj(), SCM_B).real  # a' SigmaY^-1 RY SigmaY^-1 a
        d_n = np.einsum('ij,ij->j', A.conj(), B).real      # Re(a_n^H SigmaY^-1 a_n)
            
        gradient = d_n - e_n
        delta = np.maximum(gam - gradient, 0.0) - gam
        err = np.linalg.norm(delta,ord=np.inf)
        
        return gradient, err.item()
        
