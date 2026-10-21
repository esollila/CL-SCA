
import numpy as np
from src.methods import CWO, CLMP, CLMM, FastMSBL
from src.methods import CLHybridBacktrack 
from datetime import datetime
from src.plotting.AD_plotting import plot_missdetection, plot_computation_times
from AD_simul import run_AD_monte_carlo_experiment
import time

if __name__ == "__main__": 

    # System set-up    
    N = 300                     # number of MTDs
    L = 20                      # pilot length 20,30, or 50
    Mlist = np.arange(20, 81, 10) # list of number of antennas 
    Klist = np.array([20,30,40])   # sparsity orders 
    SNRlow, SNRhigh = -15, 0 #  gammas Unif(-15,0) in dB


    #-- Algorithm parameters 
    MAX_ITERS = 500             # Max iters for CL-methods 
    ERRTOL = 1e-2
    print("Max iters: ", MAX_ITERS, "stopping tolerance :", ERRTOL)
    SWITCH_IT = 5               # Hybrid: number of SCA iterations before switching to FP updates
    GAM0 = np.zeros(N, dtype=float)   # M-SBL cannot start from the zero vector
    GAM1 = np.ones(N, dtype=float) 
    
    # MC-parameters
    MC_iters = 1000 # number of MC iterations (set to 10000 for results in the paper)

    # ===========================
    # Possible methods 
    # ===========================
    CL = {}
    CL["Hyb-bt"] = CLHybridBacktrack(
        sig=1.0, max_iter=MAX_ITERS, errtol=ERRTOL, eta = 1.0, 
        switch_iter=SWITCH_IT, 
        beta=0.5,            # backtracking contraction factor
        max_backtracks=10,   # max halvings per iteration
        print_flag=False, 
        calculate_loss=False,
        gam_init=GAM0,       # all expcept MSBL start with zero vector
        warm_start= True
    )
    
    CL["CWO"] = CWO(
        sig=1.0, max_iter=MAX_ITERS, errtol=ERRTOL, print_flag=False,
        calculate_loss=False, 
        randomize_order=True,  seed=1,  gam_init=GAM0,
    )
    CL["MSBL"] = FastMSBL(
        sig=1.0, max_iter=MAX_ITERS, errtol=ERRTOL,
        print_flag=False, calculate_loss=False, gam_init=GAM1,
    )
    CL["MM"] = CLMM(
        sig=1.0, max_iter=MAX_ITERS, errtol=ERRTOL, offset="tight",
        print_flag=False, calculate_loss=False, gam_init=GAM0,
    )
    CL["CLMP"] = CLMP(sig=1.0)
    SSR = {} # No
    #-- Choose methods: 
    CLmethods = ["Hyb-bt","CWO","MM","CLMP","MSBL"] 
        
    # ===========================
    # Main Monte-Carlo loop
    # ===========================
    titlestr = f'M={"_".join(map(str,Mlist))}_N={N}_L={L}_MC={MC_iters}_K={"_".join(map(str,Klist))}'
    
    t0 = time.perf_counter()
    Pmd, Per, tme, nmse, config = run_AD_monte_carlo_experiment(
            CL, SSR, CLmethods, titlestr, MC_iters=MC_iters, N=N, L=L, Mlist=Mlist,Klist=Klist, SNRlow=-15, SNRhigh=0)
    t1 = time.perf_counter()
    timespent = (t1 - t0)
    print(f'Time spent {timespent}')

    #%%
    save_fig = False

    methods = CLmethods +  list(SSR.keys()) 
    plot_computation_times(tme, Klist, titlestr, methods,save_fig=save_fig)
    plot_missdetection(Mlist, Pmd, Klist, titlestr, methods=methods,save_fig=save_fig)
    plot_missdetection(Mlist, nmse, Klist, titlestr, ylab="NMSE",methods=methods,save_fig=save_fig)
    plot_missdetection(Mlist, Per, Klist, titlestr, ylab="Probability of exact recovery", methods=methods,save_fig=save_fig)