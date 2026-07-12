# 
import numpy as np
import time
import json
from src.data.generate_AD_data import ActivityDetetionDataset

from joblib import Parallel, delayed
import subprocess
import copy 

def clone_method(met):
    # returns a deep copy: same attributes, independent state
    return copy.deepcopy(met)
   
#%%
def run_AD_monte_carlo_experiment(
    CL,                             # dictionary of CL methods (like CLSCA, CWO)
    SSR,                            # dictionary of SSR methods (like SOMP etc)
    titlestr,                       
    MC_iters=500,
    N=1000,                         # number of devices
    L=64,                           # pilot length
    Mlist=np.arange(10, 61, 10),    # list of number of snapshots
    Klist=np.arange(10, 51, 10),    # list of active devices K
    SNRlow=-15,
    SNRhigh=0,
):
    
    methods = list(CL.keys()) + list(SSR.keys())
    num_methods = len(methods)

    # ----------------------------
    # BUILD CONFIG DICTIONARY
    # ----------------------------
    config = {
        "MC_iters": int(MC_iters),
        "N": int(N),
        "L": int(L),
        "Mlist": [int(m) for m in Mlist],
        "Klist": [int(k) for k in Klist],
        "SNRlow": float(SNRlow),
        "SNRhigh": float(SNRhigh),
        "methods": methods,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": subprocess.check_output(
            ["python", "--version"]
        ).decode().strip(),
    }

    # Try to log algorithm parameters if available in CL objects
    algo_params = {}
    for m in list(CL.keys()):
        obj = CL[m]
        algo_params[m] = extract_simple_params(obj)
    for m in list(SSR.keys()):
        obj = SSR[m]
        algo_params[m] = extract_simple_params(obj)  

    config["algorithm_parameters"] = algo_params
    
    # ------------------------------------
    # PREALLOCATE RESULT ARRAYS
    # ------------------------------------
    Pmd = np.zeros((len(Mlist), num_methods, len(Klist)))
    Per = np.zeros((len(Mlist), num_methods, len(Klist)))
    tme = np.zeros((len(Mlist), num_methods, len(Klist)))
    nmse = np.zeros((len(Mlist), num_methods, len(Klist)))

    print(f"MC iterations = {MC_iters}, L = {L}, N = {N}")
    print(f"Detected methods: {methods}")

    for k_idx, K in enumerate(Klist):
        
        print(f"\n---- K = {K:3d}")
        base_rng = np.random.default_rng(0)

        if "CLMP" in list(methods):
            CL["CLMP"].K = int(K) 
        if "SNIHT" in list(methods):
            SSR["SNIHT"].K = int(K)
            
        for m_idx, M in enumerate(Mlist):
            print(f"- M = {M:3d} ", end="", flush=True)

            # ---- Dataset
            dataset = ActivityDetetionDataset(
                NRSIM=MC_iters,
                N=N,      # devices
                L=L,      # pilot length
                K=K,      # sparsity
                M=M,      # snapshots
                SNRlow=SNRlow,
                SNRhigh=SNRhigh,
                rng=base_rng,
            )

            # ---- Run evaluation (fast loop) ----
            Pmd_k, Per_k, tme_k, nmse_k = run_ActivityDetection_parallel(
            #Pmd_k, Per_k, tme_k, nmse_k = run_ActivityDetection(
                dataset=dataset,
                K=K,
                CL=CL,
                SSR = SSR
            )
            
            # convert dict -> array in method order
            Pmd[m_idx, :, k_idx] = np.array([Pmd_k[m] for m in methods])
            Per[m_idx, :, k_idx] = np.array([Per_k[m] for m in methods])
            tme[m_idx, :, k_idx] = np.array([tme_k[m] for m in methods])
            nmse[m_idx, :, k_idx] = np.array([nmse_k[m] for m in methods])
            
            print("done.")
            
    with open(f"saved_results/json/{titlestr}_config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("\nSaved:")
    print(f"  - results/json/{titlestr}_config.json")
    print("\nMonte Carlo simulation complete.")
    return Pmd, Per, tme, nmse, config



#%%
def run_ActivityDetection_parallel(dataset, K, CL, SSR, n_jobs=-1):
    """ 
    This function runs the activity detection for a given dataset, K, and methods 
    in CL and SSR. It returns the missdetection rate, probability of exact recovery, 
    computation time, and NMSE for each method.
    
    It runs the evaluation in parallel using joblib. The dataset is expected to be an 
    instance of ActivityDetetionDataset, which generates fresh data for each sample.
    The function uses joblib's Parallel and delayed to distribute the workload across 
    available CPU cores.
    """

    methods = list(CL.keys()) + list(SSR.keys())
    
    cputme = {m: 0.0 for m in methods}
    md = {m: 0.0 for m in methods}
    per = {m: 0.0 for m in methods}
    nmse = {m: 0.0 for m in methods}
    total_samples = len(dataset)

    res_list = Parallel(n_jobs=n_jobs)(
        delayed(process_one_sample)(i, dataset, K, CL,SSR)
        for i in range(total_samples)
    )

    for res in res_list:
        for m in methods:
            cputme[m] += res[m]["time"]
            md[m] += res[m]["md"]
            per[m] += res[m]["per"]
            nmse[m] += res[m]["nmse"]

    for m in methods:
        cputme[m] /= total_samples
        md[m] /= total_samples
        per[m] /= total_samples
        nmse[m] /= total_samples

    return md, per, cputme, nmse


#%%
def process_one_sample(idx, dataset, K, CL,SSR):
    Y, A, sup, S_K = dataset[idx]
    S = np.zeros((A.shape[1], Y.shape[1]), dtype=Y.dtype)
    S[sup,:] = S_K.copy()
    M = Y.shape[1]
    cov_m = (Y @ Y.conj().T) / M

    results = {}
    for metname, met in CL.items():
        
        met_clone = clone_method(met)  # ensure no cross-contamination of state
        t0 = time.perf_counter()
        met_clone.fit(A, cov_m)
        t1 = time.perf_counter()
        met_clone.support(K)
        md_met = len(np.setdiff1d(met_clone.supp, sup))
        met_clone.estimate_sources(A, Y)
        diff = S - met_clone.signal_estimates_
        nmse = np.sum(np.abs(diff)**2) / np.sum(np.abs(S)**2)  
        results[metname] = {
            "time": t1 - t0,
            "md": md_met / K,
            "per": 1 if md_met == 0 else 0,
            "nmse": nmse 
        }
    for metname, met in SSR.items():
        met_clone = clone_method(met)  # ensure no cross-contamination of state
        t0 = time.perf_counter()
        met_clone.fit(Y, A) 
        t1 = time.perf_counter()
        
        md_met = len(np.setdiff1d(met_clone.supp, sup))
        diff = S - met_clone.signal_estimates_
        nmse = np.sum(np.abs(diff)**2) / np.sum(np.abs(S)**2)  
        results[metname] = {
            "time": t1 - t0,
            "md": md_met / K,
            "per": 1 if md_met == 0 else 0,
            "nmse": nmse 
        }
    return results

#%%
def extract_simple_params(obj):
    params = {}

    # Case 1: normal object with __dict__
    if hasattr(obj, "__dict__"):
        items = vars(obj).items()
    else:
        # Case 2: fallback — look at all attributes via dir()
        items = ((k, getattr(obj, k)) for k in dir(obj))

    for k, v in items:
        # skip private/internal stuff
        if k.startswith("_"):
            continue

        # only keep JSON-friendly scalar types
        if isinstance(v, (int, float, bool, str)):
            params[k] = v

    return params


#%%
def run_ActivityDetection(dataset, K, CL,SSR):
    methods = list(CL.keys()) + list(SSR.keys())

    cputme = {m: 0.0 for m in methods}
    md = {m: 0.0 for m in methods}
    per = {m: 0.0 for m in methods}
    nmse = {m: 0.0 for m in methods}

    total_samples = len(dataset)

    for it, (Y, A, sup,S_K) in enumerate(dataset):  

        S = np.zeros((A.shape[1], Y.shape[1]), dtype=Y.dtype)
        S[sup,:] = S_K
        M = Y.shape[1]
        cov_m = (Y @ Y.conj().T) / M

        for metname, met in CL.items():
            t0 = time.perf_counter()
            met.fit(A, cov_m)
            t1 = time.perf_counter()

            cputme[metname] += (t1 - t0)

            met.support(K)
            md_met = len(np.setdiff1d(met.supp, sup))
            md[metname] += md_met / K
            if md_met == 0:
                per[metname] += 1
                
            met.estimate_sources(A, Y)
            diff = S - met.signal_estimates_
            nmse[metname] += np.sum(np.abs(diff)**2) / np.sum(np.abs(S)**2)   
            
        for metname, met in SSR.items():
             
            t0 = time.perf_counter()
            met.fit(Y, A) 
            t1 = time.perf_counter()

            cputme[metname] += (t1 - t0)

            md_met = len(np.setdiff1d(met.supp, sup))
            md[metname] += md_met / K
            if md_met == 0:
                per[metname] += 1
                
            #met.estimate_sources(A, Y)
            diff = S - met.signal_estimates_
            nmse[metname] += np.sum(np.abs(diff)**2) / np.sum(np.abs(S)**2)          
            
        if (it + 1) % 100 == 0:
            print(".", end="", flush=True)

    for m in methods:
        md[m] /= total_samples
        per[m] /= total_samples
        cputme[m] /= total_samples
        nmse[m] /= total_samples

    return md, per, cputme, nmse 