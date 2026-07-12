import numpy as np
from ..base import CLobject
import numbers
from scipy.linalg import cho_factor, cho_solve

class CLSCA(CLobject):
    """
    CLSCA: Covariance-learning Successive Convex Approximation algorithm for sparse signal recovery.
    
    Parameters
    ----------
    sig : float, optional
        Noise variance (sigma^2). Default is 1.0.
    max_iter  : int, optional
        Maximum number of iterations. Default is 50.
    errtol : float, optional
        Convergence tolerance. Default is 1e-3.
    epsval : float, optional
        step-size sequence parameter (default is 0.05)
    eta : float, optional
        step-size sequence parameter (default is 0.99)
    prunetol : float, optional
        threshold for pruning small gammas when updatingg SigmaY  (default is 0, i.e., no pruning). Rationale: 
        If gammas are very small, the corresponding atoms contribute negligibly to SigmaY, and pruning them can improve 
        numerical stability and reduce computation.
    gamma_init : array-like, optional
        Initial gammas. If None, initilized to all zeros. If provided, must be of shape (N,).
    print_flag : bool, optional
        If True, prints convergence information. Default is False. 
    calculate_loss : bool, optional
        If True, computes the loss and Karusk-Kuhn-Tucer loss at each iteration. Default is False.
    """
    def __init__(self, sig=1.0, max_iter =50, errtol=1e-3, epsval=0.05, eta = 0.99, prunetol = 0, gam_init=None, print_flag=False, calculate_loss=False):

        if not isinstance(sig, numbers.Real) or sig <= 0:
            raise ValueError("'sig' must be a positive real scalar")
        self.sig = float(sig)
        self.max_iter  = int(max_iter )
        self.errtol = float(errtol)
        self.eta = float(eta) 
        self.epsval = float(epsval)
        self.prunetol = float(prunetol)
        self.print_flag = bool(print_flag)
        self.calculate_loss = bool(calculate_loss)
        self.gam_init = gam_init  # default: all zeros
        
    def fit(self,A, SCM):
        """
        Faster CLSCA for complex-valued data.

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
            self.gam_init = np.asarray(self.gam_init, dtype=float).reshape(-1)
            if self.gam_init.shape[0] != N:
                raise ValueError("gam_init must have shape (N,)")
            gamma_old = self.gam_init
        else:
            # it is recommended to start with all zeros as below. Only use gam_init if you know 
            # gam_init is close to the optimum
            gamma_old = np.zeros(N, dtype=float)  # 
        
        eta = float(self.eta) # parameter of step-size sequence 
        sig = self.sig # known noise variance 

        # Pre-allocate identity once
        indx = [True]*N # default: all N atoms are active (not pruned)
        loss, KKTloss = [], []
        
        eps = 1e-14  # small positive floor; prevents division blow-up if gam_den 
        # rounds to ~0 due to floating-point error in near-singular cases
        
        for iters in range(1, self.max_iter  + 1):
            # Build SigmaY = sig*I + W W^H with W = A * sqrt(gamma)
            # Using sqrt helps numerical stability and reduces flops.
            
            if self.prunetol > 0:
                indx = gamma_old >= self.prunetol * gamma_old.max()
            W = A[:, indx] * np.sqrt(gamma_old[indx])
            SigmaY = W @ W.conj().T
            SigmaY.flat[::SigmaY.shape[0]+1] += sig
            
            if self.calculate_loss:
                loss.append(self.compute_loss(SigmaY,SCM))
                _, kkt_err = self.compute_KKT_loss(A, gamma_old, SCM)
                KKTloss.append(kkt_err)

            # Solve SigmaY B = A for B (M, N) without forming inverse     
            cF = cho_factor(SigmaY, lower=True, check_finite=False, overwrite_a=False)
            B = cho_solve(cF, A, check_finite=False)
 
            # Numerator and denominator for gamma update
            # gam_num[m] = Re(b_n^H SCM b_n)
            SCM_B = SCM @ B
            gam_num = np.einsum('ij,ij->j', B.conj(), SCM_B).real  # (N,)
            gam_den = np.einsum('ij,ij->j', A.conj(), B).real #  gam_den[n] = Re(a_n^H B_n)

            # gam_den = Re(a_n^H SigmaY^-1 a_n) is theoretically > 0 (SigmaY is PD);
            # clamp only guards against floating-point round-off near singularity
            np.maximum(gam_den, eps, out=gam_den)

            inv_den = 1.0 / gam_den                   
            sqrt_term = np.sqrt(gam_num * inv_den)       # sqrt(gam_num / gam_den)            
            gamma_new = gamma_old + inv_den * (sqrt_term - 1.0)
            np.maximum(gamma_new, 0.0, out=gamma_new)    # the [.]_+ oper            

            # Then damping: 
            d = gamma_new - gamma_old
            gamma_sca = gamma_old + eta*d
            err = np.linalg.norm(d)

            if err < self.errtol:
                if self.print_flag:
                    print(f"Solution converged at it  {iters}, err={err:.3e}")
                break
            
            gamma_old = gamma_sca.copy()
            eta = eta * (1.0 - self.epsval * eta)  # step-size sequence update
            
        # If not converged within max_iter 
        if self.print_flag and err >= self.errtol:
            print(f"Solution not converged. Error: {err:.6f}")
        
        self.gammas = gamma_sca   
        self.iters = iters
        self.err = err
        self.loss = loss
        self.KKTloss = KKTloss
        
        return gamma_sca, iters, err
 