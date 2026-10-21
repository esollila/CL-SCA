"""
convergence_gap.py
==================
Median objective gap  ell(gamma^k) - ell_star  versus iteration k for all
proposed methods together with CL-MM and M-SBL.

For each Monte Carlo (MC) instance, ell_star is the smallest loss reached by a
long reference run of CL-Hyb-bt (REF_ITERS iterations), or by any compared
method if one goes lower. The plot shows the median gap over the MC runs on a
logarithmic axis.

Usage:  python convergence_gap.py      (run from the project root)
"""

import numpy as np
import matplotlib.pyplot as plt

from src.methods import (CLSCASchedule, CLSCABacktrack, CLFPSchedule,
                         CLHybridSchedule, CLHybridBacktrack, CLMM, FastMSBL)
from src.data.generate_AD_data import ActivityDetetionDataset

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N, L, K, M      = 300, 20, 30, 50   # sources, pilot length, active, antennas
SNRlow, SNRhigh = -15, 0            # dB
MC_ITERS        = 20                # Monte Carlo runs
MAX_ITERS       = 250               # iterations of every compared method
REF_ITERS       = 1500              # iterations of the reference run
ETA_SCA, ETA_FP = 0.60, 0.35        # tuned fixed step sizes
SWITCH_IT       = 5                 # CL-Hyb: SCA -> FP switch iteration
XMAX            = 120               # right limit of the iteration axis
FLOOR_REL       = 1e-13             # gaps below FLOOR_REL*|ell_star| are clipped

GAM0 = np.ones(N)                   # common initialization (M-SBL needs gamma > 0)
COMMON = dict(sig=1.0, max_iter=MAX_ITERS, errtol=0.0,   # errtol=0: run all iterations
              calculate_loss=True, gam_init=GAM0)

# ---------------------------------------------------------------------------
# Methods: key -> (estimator, plot style)
# ---------------------------------------------------------------------------
DASH_FSS, DASH_DSS, DASH_BT = (0, (5, 2)), (0, (1, 2)), (0, (3, 1, 1, 1, 1, 1))

METHODS = {
    "CL-Hyb":     (CLHybridSchedule(eta_sca=ETA_SCA, eta_fp=ETA_FP,
                                    switch_iter=SWITCH_IT, **COMMON),
                   dict(color="#0072B2", linestyle="-", lw=2.0)),
    "CL-Hyb-bt":  (CLHybridBacktrack(eta=1.0, switch_iter=SWITCH_IT, max_backtracks=10,
                                     **COMMON),
                   dict(color="#0072B2", linestyle=DASH_BT, lw=2.0)),
    "CL-FP-fss":  (CLFPSchedule(step="fss", eta=ETA_FP, **COMMON),
                   dict(color="#D55E00", linestyle=DASH_FSS)),
    "CL-FP-dss":  (CLFPSchedule(step="dss", eta=0.99, epsval=0.05, **COMMON),
                   dict(color="#D55E00", linestyle=DASH_DSS)),
    "CL-SCA-fss": (CLSCASchedule(step="fss", eta=ETA_SCA, **COMMON),
                   dict(color="#009E73", linestyle=DASH_FSS)),
    "CL-SCA-dss": (CLSCASchedule(step="dss", eta=0.99, epsval=0.05, **COMMON),
                   dict(color="#009E73", linestyle=DASH_DSS)),
    "CL-SCA-bt":  (CLSCABacktrack(eta=1.0, max_backtracks=10, **COMMON),
                   dict(color="#009E73", linestyle=DASH_BT)),
    "CLMM":       (CLMM(offset="loose", **COMMON),
                   dict(color="#AA4499", linestyle=(0, (6, 2, 1, 2)))),
    "CLMM-2":     (CLMM(offset="tight", **COMMON),
                   dict(color="#CC79A7", linestyle=(0, (8, 2)))),
    "M-SBL":      (FastMSBL(**COMMON),
                   dict(color="#595959", linestyle="-.")),
}

REF = CLHybridBacktrack(eta=1.0, switch_iter=SWITCH_IT, max_backtracks=10,
                        **{**COMMON, "max_iter": REF_ITERS})

def pad(traj, n):
    """Pad a loss trajectory to length n by repeating its last value."""
    traj = np.asarray(traj, dtype=float)
    return traj[:n] if len(traj) >= n else np.r_[traj, np.full(n - len(traj), traj[-1])]


# ---------------------------------------------------------------------------
# Monte Carlo loop
# ---------------------------------------------------------------------------
dataset = ActivityDetetionDataset(NRSIM=MC_ITERS, N=N, L=L, K=K, M=M,
                                  SNRlow=SNRlow, SNRhigh=SNRhigh,
                                  rng=np.random.default_rng(1))

losses   = {name: [] for name in METHODS}
ell_star = []

for it, (Y, A, _, _) in enumerate(dataset):
    SCM = Y @ Y.conj().T / Y.shape[1]

    REF.fit(A, SCM)
    es = np.min(REF.loss)
    for name, (alg, _) in METHODS.items():
        alg.fit(A, SCM)
        losses[name].append(np.asarray(alg.loss, dtype=float))
        es = min(es, np.min(alg.loss))   # guard: a method may go below the reference
    ell_star.append(es)
    print(f"MC {it + 1:3d}/{MC_ITERS}:  ell_star = {es:.8f}")

# ---------------------------------------------------------------------------
# Median objective gap and plot
# ---------------------------------------------------------------------------
plt.figure(figsize=(4.2, 3.0))
ax = plt.gca()

print("\nMedian objective gap:")
for name, (_, style) in METHODS.items():
    gaps = [np.maximum(pad(traj, MAX_ITERS) - es, FLOOR_REL * max(1.0, abs(es)))
            for traj, es in zip(losses[name], ell_star)]
    med = np.median(np.vstack(gaps), axis=0)
    print(f"  {name:11s} k=25: {med[25]:.2e}   k=50: {med[50]:.2e}   "
          f"k={MAX_ITERS - 1}: {med[-1]:.2e}")
    ax.semilogy(med, color=style["color"], linestyle=style["linestyle"],
                linewidth=style.get("lw", 1.6), label=name)

ax.set_xlabel(r"iteration $k$")
ax.set_ylabel(r"$\ell(\gamma^k)-\ell^\star$")
ax.set_xlim(0, XMAX)
ax.grid(True, color="0.9", linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left",
          columnspacing=1.0, handlelength=2.4)
plt.tight_layout()
# plt.savefig(f"gap_K={K}_M={M}_N={N}_L={L}_MC={MC_ITERS}.pdf", bbox_inches="tight")
plt.show()
