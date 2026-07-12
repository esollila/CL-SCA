
import numpy as np
from src.methods import CLSCA, CWO, CLMP, FastMSBL
from datetime import datetime
from src.plotting.AD_plotting import plot_missdetection, plot_computation_times
from AD_simul import run_AD_monte_carlo_experiment
import time

if __name__ == "__main__": 
    
    MC_iters = 1000         # number of MC iterations
    N = 300                 # number of MTDs
    L = 20                  # pilot length 20,30, or 40
    Mlist = np.arange(20, 81, 10) # list of number of antennas  
    MAX_ITERS = 50          # Max iters for CL-methods 
    MAX_ITERS_MSBL = 500    # Max iters for MSBL
    SNRlow = -15
    SNRhigh = 0
    Klist = np.array([20,30,40])   

    # ===========================
    # Possible methods 
    # ===========================
    calculate_loss = False # do not calculate for MC simulations
    clsca = CLSCA(
        sig=1.0, 
        max_iter=MAX_ITERS, 
        errtol=1e-3, 
        eta = 0.99,
        epsval = 0.05,
        prunetol = 0,
        print_flag=False,
        calculate_loss=calculate_loss
        )
    cwo = CWO(sig=1.0,
        max_iter=MAX_ITERS, 
        errtol=1e-3, 
        print_flag=False,
        calculate_loss=calculate_loss,
        randomize_order=True,  # randomize the order of atoms in each sweep
        seed=None  # different seed for each run; set to an integer for reproducibility
        )
    msbl = FastMSBL(
        sig=1.0,
        max_iter=MAX_ITERS_MSBL,
        errtol=1e-5,
        print_flag=False,
        calculate_loss=calculate_loss
        )
    clmp = CLMP(sig=1.0)
    CL = {}
    
    #-- Choose methods: 
    # comment out the methods you do not want to include in the simulation  
    CL["SCA"] = clsca
    CL["CWO"] = cwo        
    CL["CLMP"] = clmp
    CL["MSBL"] = msbl

    SSR = {}
    
    # ===========================
    # Main Monte-Carlo loop
    # ===========================
    titlestr = f'M={"_".join(map(str,Mlist))}_N={N}_L={L}_MC={MC_iters}_K={"_".join(map(str,Klist))}'
    
    t0 = time.perf_counter()
    Pmd, Per, tme, nmse, config = run_AD_monte_carlo_experiment(
            CL, SSR, titlestr, MC_iters=MC_iters, N=N, L=L, Mlist=Mlist,Klist=Klist, SNRlow=-15, SNRhigh=0)
    t1 = time.perf_counter()
    timespent = (t1 - t0)
    print(f'Time spent {timespent}')

    #%%
    save_fig = True

    methods = list(CL.keys()) +  list(SSR.keys()) 
    plot_computation_times(tme, Klist, titlestr, methods,save_fig=save_fig)
    plot_missdetection(Mlist, Pmd, Klist, titlestr, methods=methods,save_fig=save_fig)
    plot_missdetection(Mlist, nmse, Klist, titlestr, ylab="NMSE",methods=methods,save_fig=save_fig)
    plot_missdetection(Mlist, Per, Klist, titlestr, ylab="Probability of exact recovery", methods=methods,save_fig=save_fig)
    