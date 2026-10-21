"""
convergence_runmin.py
=====================
Running-minimum loss  min_{s<=k} ell(gamma^s) / ell(gamma^0)  for a few
representative Monte Carlo (MC) runs.

Target level: for each MC instance, the loss reached by CWO after CWO_CYCLES
cycles. Each curve stops when it reaches this target, so its end point is the
iteration count to target. The legend shows the average number of iterations
and the average time to reach the target, computed over ALL MC runs:
    time = iterations x per-iteration time,
where the per-iteration time is measured on the first instance with loss
evaluation switched off. A ">" means some runs never reached the target within
MAX_ITERS, so the average is a lower bound.

Usage:  python convergence_runmin.py      (run from the project root)
"""

import copy
import time

import numpy as np
import matplotlib.pyplot as plt

from src.methods import CLFPSchedule, CLHybridBacktrack, CLMM, FastMSBL, CWO
from src.data.generate_AD_data import ActivityDetetionDataset

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
N, L, K, M      = 300, 20, 30, 50   # sources, pilot length, active, antennas
SNRlow, SNRhigh = -15, 0            # dB
MC_ITERS        = 20                # Monte Carlo runs
MAX_ITERS       = 250               # iterations of every compared method
CWO_CYCLES      = 10                # CWO cycles defining the target level
ETA_FP          = 0.35              # tuned fixed step size of CL-FP-fss
SWITCH_IT       = 5                 # CL-Hyb: SCA -> FP switch iteration
N_SHOW          = 3                 # MC runs drawn per method
XMAX            = 60                # right limit of the iteration axis
TIKZ            = False             # True: export a TikZ .tex file (needs tikzplotlib)

GAM0 = np.ones(N)                   # common initialization (M-SBL needs gamma > 0)
COMMON = dict(sig=1.0, max_iter=MAX_ITERS, errtol=0.0,   # errtol=0: run all iterations
              calculate_loss=True, gam_init=GAM0)

# ---------------------------------------------------------------------------
# Methods: label -> (estimator, plot style)
# ---------------------------------------------------------------------------
METHODS = {
    "CL-Hyb-bt": (CLHybridBacktrack(eta=1.0, switch_iter=SWITCH_IT, max_backtracks=10,
                                    **COMMON),
                  dict(color="#0072B2", linestyle=(0, (3, 1, 1, 1, 1, 1)), lw=2.0)),
    "CL-FP-fss": (CLFPSchedule(step="fss", eta=ETA_FP, **COMMON),
                  dict(color="#D55E00", linestyle=(0, (5, 2)))),
    "M-SBL":     (FastMSBL(**COMMON),
                  dict(color="#595959", linestyle="-.")),
    "CLMM-2":    (CLMM(offset="tight", **COMMON),
                  dict(color="#CC79A7", linestyle=(0, (8, 2)))),
}

# The loss is recorded at the START of each iteration, so CWO_CYCLES + 1
# iterations are needed to see the iterate after CWO_CYCLES cycles.
CWO_REF = CWO(sig=1.0, max_iter=CWO_CYCLES + 1, errtol=0.0, calculate_loss=True,
              randomize_order=True, seed=1, gam_init=GAM0)


def pad(traj, n):
    """Pad a loss trajectory to length n by repeating its last value."""
    traj = np.asarray(traj, dtype=float)
    return traj[:n] if len(traj) >= n else np.r_[traj, np.full(n - len(traj), traj[-1])]


def ms_per_iteration(alg, A, SCM, n_rep=3):
    """Per-iteration wall-clock time (ms) of a copy of alg with loss evaluation off."""
    alg = copy.deepcopy(alg)
    alg.calculate_loss = False
    best = np.inf
    for _ in range(n_rep):
        t0 = time.perf_counter()
        alg.fit(A, SCM)
        best = min(best, (time.perf_counter() - t0) / max(alg.iters, 1))
    return 1e3 * best


# ---------------------------------------------------------------------------
# Monte Carlo loop
# ---------------------------------------------------------------------------
dataset = ActivityDetetionDataset(NRSIM=MC_ITERS, N=N, L=L, K=K, M=M,
                                  SNRlow=SNRlow, SNRhigh=SNRhigh,
                                  rng=np.random.default_rng(1))

losses  = {name: [] for name in METHODS}
ell_ref = []                                   # per-run CWO target level

for it, (Y, A, _, _) in enumerate(dataset):
    SCM = Y @ Y.conj().T / Y.shape[1]

    CWO_REF.fit(A, SCM)
    ell_ref.append(np.min(CWO_REF.loss))

    if it == 0:                                # timing, measured once
        ms = {name: ms_per_iteration(alg, A, SCM) for name, (alg, _) in METHODS.items()}
        cwo_long = copy.deepcopy(CWO_REF)
        cwo_long.max_iter = MAX_ITERS          # longer run -> stabler estimate
        ms["CWO"] = ms_per_iteration(cwo_long, A, SCM)

    for name, (alg, _) in METHODS.items():
        alg.fit(A, SCM)
        losses[name].append(pad(alg.loss, MAX_ITERS))
    print(f"MC {it + 1:3d}/{MC_ITERS} done")

# ---------------------------------------------------------------------------
# Iterations to target (over all runs)
# ---------------------------------------------------------------------------
print(f"\nIterations / time to the CWO target ({CWO_CYCLES} cycles):")
legend = {}
for name in METHODS:
    hits, censored = [], 0
    for traj, ref in zip(losses[name], ell_ref):
        idx = np.flatnonzero(np.minimum.accumulate(traj) <= ref)
        hits.append(idx[0] + 1 if len(idx) else MAX_ITERS)
        censored += len(idx) == 0
    mean_it = np.mean(hits)
    pre = ">" if censored else ""
    legend[name] = f"{name} ({pre}{mean_it:.0f} it, {pre}{mean_it * ms[name]:.1f} ms)"
    print(f"  {name:10s} {pre}{mean_it:6.1f} it  {pre}{mean_it * ms[name]:7.1f} ms"
          + (f"   [{censored}/{MC_ITERS} runs did not reach the target]" if censored else ""))
print(f"  {'CWO':10s} {CWO_CYCLES:6.1f} it  {CWO_CYCLES * ms['CWO']:7.1f} ms")

# ---------------------------------------------------------------------------
# Plot N_SHOW randomly chosen runs (the same runs for every method)
# ---------------------------------------------------------------------------
shown = np.sort(np.random.default_rng(0).choice(MC_ITERS, size=min(N_SHOW, MC_ITERS),
                                                replace=False))
plt.figure(figsize=(4.2, 3.0))
ax = plt.gca()

for name, (_, style) in METHODS.items():
    for i, j in enumerate(shown):
        traj = losses[name][j]
        rm = np.minimum.accumulate(traj) / traj[0]
        below = np.flatnonzero(rm <= ell_ref[j] / traj[0])
        if len(below):
            rm = rm[:below[0] + 1]              # stop the curve at the target
        ax.plot(rm, color=style["color"], linestyle=style["linestyle"],
                linewidth=style.get("lw", 1.4), label=legend[name] if i == 0 else None)

ax.set_xlabel(r"iteration $k$")
ax.set_ylabel(r"$\min_{s\leq k}\,\ell(\gamma^s)\,/\,\ell(\gamma^0)$")
ax.set_xlim(0, XMAX)
ax.grid(True, color="0.9", linewidth=0.6)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
plt.tight_layout()
plt.show()
