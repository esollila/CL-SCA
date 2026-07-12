import numpy as np
import numbers
from ..base import CLobject
from scipy.linalg import cho_factor, cho_solve

class FastMSBL(CLobject):
    """
    sig : float
            Known noise variance (sigma^2).
    max_iter  : int
            Maximum number of EM it ations.
    dmu : float
            Max change in mu between the last two iterations for termination
    """

    def __init__(self, sig=1.0, gam_init=None, max_iter =50, errtol=1e-3, print_flag=False, calculate_loss=False):

        if not isinstance(sig, numbers.Real) or sig <= 0:
            raise ValueError("'sig' must be a positive real scalar")
        
        self.sig = float(sig)
        self.max_iter  = int(max_iter )
        self.errtol = float(errtol)
        self.print_flag = bool(print_flag)
        self.gam_init = gam_init
        self.calculate_loss = bool(calculate_loss)
        
    def fit(self,A,SCM):
        """
        EM algorithm for M-SBL (Multiple Sparse Bayesian Learning). Uses the fast EM implementation described
    
        Parameters
        ----------
        A : (M, N) complex ndarray
        SCM : (M, M) complex ndarray

        Returns
        -------
        gammas : (N,) float array
        iters : int
        err   : float
        """
        
        A = np.asarray(A, dtype=np.complex128)
        SCM = np.asarray(SCM, dtype=np.complex128)
        M, N = A.shape

        # Initialization
        if self.gam_init is None or len(np.atleast_1d(self.gam_init)) == 0:
            gamma_old = np.ones(N, dtype=np.float64)
        else:
            gamma_old = np.asarray(self.gam_init, dtype=float).reshape(-1)
            if gamma_old.shape[0] != N:
                raise ValueError("gam_init must have shape (N,)")

        loss, KKTloss = [], [] 
        eps = 1e-14  # small positive floor; prevents division blow-up if gam_den 
        
        for iters in range(1, self.max_iter  + 1):
            
            W = A * np.sqrt(gamma_old)
            SigmaY = W @ W.conj().T
            SigmaY.flat[::SigmaY.shape[0]+1] += self.sig

            if self.calculate_loss:
                loss.append(self.compute_loss(SigmaY,SCM))
                _, KKT_err = self.compute_KKT_loss(A, gamma_old, SCM)
                KKTloss.append(KKT_err)
            
            # Solve SigmaY B = A for B (N, M) without forming inverse
            cF = cho_factor(SigmaY, lower=True, check_finite=False, overwrite_a=False)
            B = cho_solve(cF, A, check_finite=False)

            # Numerator and denominator for gamma update
            # gam_num[m] = Re(b_m^H RY b_m)
            SCM_B = SCM @ B
            gam_num = np.einsum('ij,ij->j', B.conj(), SCM_B).real  # (N,)
            gam_den = np.einsum('ij,ij->j', A.conj(), B).real
        
            # Proposed update (elementwise nonnegativity)
            grad = gam_num - gam_den
            gamma_new = gamma_old + gamma_old**2 * grad

            den = max(np.linalg.norm(gamma_old, ord=np.inf), 1e-14)
            err = np.linalg.norm(gamma_new - gamma_old, ord=np.inf) / den
        
            if err < self.errtol:
                if self.print_flag:
                    print(f"Solution converged at it  {iters}, err={err:.3e}")
                break
            
            gamma_old = gamma_new.copy()

            
        # If not converged within max_iter 
        if self.print_flag and err >= self.errtol:
            print(f"Solution not converged. Error: {err:.6f}")
        
        self.gammas = gamma_new  
        self.iters = iters
        self.err = err
        self.loss = loss
        self.KKTloss = KKTloss
        
        return gamma_new, iters, err
