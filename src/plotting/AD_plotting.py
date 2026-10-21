import matplotlib.pyplot as plt
import numpy as np
#import time
from datetime import datetime
#%%

def plot_missdetection(
    Mlist, Pmd, Klist, titlestr, ylim=None, 
    ylab = "Probability of missdetection", methods=None, fignro=1,save_fig=True, 
    tikz=False, log_scale = True,which = None):
    """
    Plot probability of missdetection (Pmd) for multiple methods and K values.

    Args:
        Mlist (1D array): number of antennas (x-axis)
        Pmd (3D array): shape (len(Mlist), num_methods, len(Klist))
        Klist (1D array): different sparsity levels
        methods (list of str, optional): method names. Default: ['Method 1', 'Method 2', ...]
        fignro (int, optional): figure number

    Example:
        plot_missdetection(Mlist, Pmd, Klist, methods=['CWO', 'CL-MP'])
    """
    
    if which is not None: 
        Pmd = Pmd[:,which,:]
    num_methods = Pmd.shape[1]
    num_K = Pmd.shape[2]
    # Define markers and line styles (extend if needed)
    markers = ['o', 's', '^', 'd', 'x', '*', 'v', '+']
    colors = plt.cm.tab10.colors  # up to 10 distinct colors

    if methods is None:
        methods = [f'Method {i+1}' for i in range(num_methods)]
                
    if which is not None:
        methods = [methods[i] for i in which]
        markers = [markers[i] for i in which]
        colors = [colors[i] for i in which]

    if not tikz:
        linestyles = ['-', '--', '-.', ':']
    else: 
        linestyles = ['-', '-', '-', '-']
    

    plt.figure(fignro)
    plt.clf()

    if log_scale:
        pltfnc = plt.semilogy
    else:
        pltfnc = plt.plot

    
    for m_idx in range(num_methods):
        color = colors[m_idx % len(colors)]
        method_name = methods[m_idx]

        for k_idx, K in enumerate(Klist):
            marker = markers[k_idx % len(markers)]
            linestyle = linestyles[k_idx % len(linestyles)]
            label = f'{method_name} $K={K}$' if k_idx == 0 else f'$K={K}$'

            pltfnc(
                Mlist,
                Pmd[:, m_idx, k_idx],
                color=color,
                marker=marker,
                #linestyle=linestyle,
                linewidth=2,
                markersize=8,
                label=label
            )
            if ylim is not None:
                plt.ylim(ylim)

    plt.xlabel('M (number of antennas)')
    plt.ylabel(ylab)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    if not tikz:
        plt.legend(fontsize=12)
        
    plt.tight_layout()
    
    if save_fig:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M") 
        filename = ylab.replace(" ", "_") 
        plt.savefig(f'saved_results/figures/{filename}_{titlestr}_{timestamp}.pdf', bbox_inches="tight")
        plt.close(fignro)   # 🔑 prevents display
    else:
        if tikz:
            import tikzplotlib # Requires matplotlib 3.7.0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M") 
            filename = ylab.replace(" ", "_") 
            tikzplotlib.save(f'saved_results/tikz/{filename}_{titlestr}_{timestamp}.tex')
            #tikzplotlib.save("./tikz/spect.tex")    
        else: 
            plt.show()    
#%%
def plot_computation_times(tme, Klist,titlestr, methods=None, fignro=1,save_fig=True,tikz=False, which = None):
    """
    Plot computation times as a grouped bar chart for multiple methods and K values.

    Args:
        tme (numpy.ndarray): shape (num_M, num_methods, num_K)
        Klist (1D array): list of K values
        methods (list of str, optional): method names. Default: Method 1, Method 2, ...
        fignro (int, optional): figure number
    """
    
    if which is not None: 
       tme = tme[:,which,:]
    num_methods = tme.shape[1]
    num_K = tme.shape[2]

    if methods is None:
        methods = [f"Method {i+1}" for i in range(num_methods)]
    if which is not None:
        methods = [methods[i] for i in which]
        
    # Average over Mlist (like MATLAB mean(tme,1))
    tmp = tme.mean(axis=0)          # shape -> (num_methods, num_K)
    cputme = tmp.T                  # shape -> (num_K, num_methods)

    x = np.arange(num_K)            # positions of K values
    total_width = 0.8               # total width for all bars in a group
    width = total_width / num_methods  # individual bar width

    plt.figure(fignro,figsize=(5,6))
    plt.clf()

    for m_idx, method_name in enumerate(methods):
        # Compute offsets for each method in the group
        offsets = x - total_width/2 + width/2 + m_idx*width
        plt.bar(offsets, cputme[:, m_idx], width=width, label=method_name)

    plt.xticks(x, Klist)
    plt.xlabel("K (number of active devices)")
    plt.ylabel("Running time [s]")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    if not tikz: 
        plt.legend(loc="upper left", fontsize=14)
    plt.tight_layout()
    
    if save_fig:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M") 
        plt.savefig(f'saved_results/figures/CPU_ad_{titlestr}_{timestamp}.pdf', bbox_inches="tight")
        plt.close(fignro)   # 🔑 prevents display
    else:
        if tikz:
            import tikzplotlib # Requires matplotlib 3.7.0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M") 
            tikzplotlib.save(f'saved_results/tikz/CPU_ad_{titlestr}_{timestamp}.tex')
            #tikzplotlib.save("./tikz/spect.tex")    
        else: 
            plt.show()
#%%

