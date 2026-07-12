import numpy as np
from ..base import CLobject
import numbers
from scipy.linalg import cho_factor, cho_solve

class CWO(CLobject):
    
    def __init__(self,sig=1.0, max_iter = 50, errtol=1e-3, print_flag=False, gam_init=None, calculate_loss=False, randomize_order=True, seed=None):
        """
        Parameters
        ----------
        sig : float, positive
            Noise/scale parameter.
        max_iter : int, positive
            Maximum number of iterations.
        errtol : float, positive
            Convergence tolerance on successive iterate error.
        print_flag : bool
            If True, print convergence progress.
        calculate_loss : bool
            If True, track loss value at each iteration.
        """
        if not isinstance(sig, numbers.Real) or sig <= 0:
            raise ValueError("'sig' must be a positive real scalar")
        if not isinstance(max_iter, numbers.Integral) or max_iter <= 0:
            raise ValueError("'max_iter' must be a positive integer")
        if not isinstance(errtol, numbers.Real) or errtol <= 0:
            raise ValueError("'errtol' must be a positive real scalar")
        self.sig = float(sig)
        self.max_iter  = int(max_iter)
        self.errtol = float(errtol)
        self.print_flag = bool(print_flag)
        self.gam_init = gam_init
        self.calculate_loss = bool(calculate_loss)
        self.randomize_order = bool(randomize_order)
        self._rng = np.random.default_rng(seed)  # seed=None -> different every run; seed=42 -> reproducible

    def fit(self,A, SCM):
        """
        Python implementation of CWOpt (coordinate-wise optimization with
        Sherman-Morrison rank-1 updates).

        Parameters
        ----------
        A : np.ndarray (complex)
            Dictionary matrix of shape (N, M).
        SCM : np.ndarray (complex)
            Sample covariance matrix of shape (N, N).

        Returns
        -------
        Ilocs : np.ndarray (int)
            Indices of the K largest entries in gamma (support), sorted ascending.
            Note: zero-based indices (Python convention).
        gamma : np.ndarray (float)
            Length-M vector of estimated source powers.
        j1 : int
            Number of it ations performed (1-based).
        """
        # Ensure complex dtype for matrices; gamma stays real
        A = np.asarray(A, dtype=np.complex128)
        SCM = np.asarray(SCM, dtype=np.complex128)
        M, N = A.shape
        I = np.eye(M, dtype=np.complex128)   # Pre-allocate identity once

        if self.gam_init is not None:
            self.gam_init = np.asarray(self.gam_init, dtype=float).reshape(-1)
            if self.gam_init.shape[0] != N:
                raise ValueError("gam_init must have shape (N,)")
            gamma_old = self.gam_init
            W = A * np.sqrt(gamma_old)
            SigmaY = W @ W.conj().T
            SigmaY.flat[::SigmaY.shape[0]+1] += self.sig
            cF = cho_factor(SigmaY, lower=True, check_finite=False,overwrite_a=False)
            SigmaYinv = cho_solve(cF, I, check_finite=False)
            SigmaYinv = (SigmaYinv + SigmaYinv.conj().T) / 2
        else:
            # it is recommended to start with all zeros as below. Only use gam_init if you know 
            # gam_init is close to the optimum
            gamma_old = np.zeros(N, dtype=float)  # 
            SigmaYinv = (1.0 / self.sig) * np.eye(M, dtype=np.complex128)  
 
        gamma = gamma_old.copy()  # Current iterate of gammas

        loss, KKTloss = [], []
        eps = 1e-14  # small positive floor; prevents division blow-up if c
        # rounds to ~0 due to floating-point error in near-singular cases
        
        for j1 in range(1, self.max_iter  + 1):
            
            if self.calculate_loss:
                SigmaY = self.sig * I + (A * gamma_old) @ A.conj().T
                loss.append(self.compute_loss(SigmaY,SCM))
                _,kkt_err = self.compute_KKT_loss(A, gamma_old, SCM)
                KKTloss.append(kkt_err)
                
            if self.randomize_order:
                indices = self._rng.permutation(N)  # fresh random order each sweep, reproducible if seeded
            else:
                indices = np.arange(N)

            for j2 in indices:
                indx = indices[j2]
                a = A[:, indx]                 # (N,)
                b = SigmaYinv @ a              # (N,)
                # c = real(a^H b)
                c = float(np.real(np.vdot(a, b)))
                c = max(c, eps)

                # tmp = (real(b^H RY b) - c) / c^2
                SCM_b = SCM @ b
                bh_SCM_b = float(np.real(np.vdot(b, SCM_b)))
                tmp = (bh_SCM_b - c) / (c * c)

                d = max(tmp, -gamma_old[indx])
                gamma[indx] = gamma_old[indx] + d

                # Sherman–Morrison rank-1 update when d != 0:
                # SigmaYinv <- SigmaYinv - (d / (1 + d*c)) * b b^H
                if d != 0.0:
                    factor = d / (1.0 + d * c)
                    SigmaYinv -= factor * np.outer(b, b.conj())

            denom = np.linalg.norm(gamma_old, ord=np.inf)
            denom = max(denom, 1e-14)
            err = np.linalg.norm(gamma - gamma_old, ord=np.inf) / denom
                            
            if err < self.errtol:
                if self.print_flag:
                    print(f"Solution converged!\nIteration: {j1:4d}. Error: {err:.7f}")
                break

            gamma_old = gamma.copy()

        # If not converged within max_iter 
        if self.print_flag and err >= self.errtol:
            print(f"Solution not converged. Error: {err:.6f}")
        self.gammas = gamma
        self.iters = j1
        self.err = err if j1 > 1 else None
        self.KKTloss = KKTloss
        self.loss = loss
        
        return gamma, j1, err if j1 > 1 else None     