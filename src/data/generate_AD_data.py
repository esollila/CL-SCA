import numpy as np

#%%
def gaussian_mm(N, M, unit_norm=True, rng=None):
    """
    Create a complex Gaussian dictionary A of shape (N, M).
    If unit_norm=True, columns are L2-normalized (common in dictionary simulations).
    """
    if rng is None:
        rng = np.random.default_rng()

    A = (rng.standard_normal((N, M)) + 1j * rng.standard_normal((N, M))) / np.sqrt(2.0)
    if unit_norm:
        norms = np.linalg.norm(A, axis=0)
        norms = np.where(norms == 0, 1.0, norms)
        A = A / norms
    return A

#%%
def bernoulli_mm(L, N, unit_norm=True, p=0.5, rng=None):
    """
    Generate a complex Bernoulli measurement matrix A of shape (L, N).

    Each entry is (sr + i*si) / sqrt(2), where sr, si ∈ {+1, -1} drawn from
    Bernoulli(p) via 1 - 2*Bernoulli, 

    Parameters
    ----------
    L : int
        Number of rows.
    N : int
        Number of columns.
    unit_norm : bool, optional
        If True, normalize columns to unit L2-norm. Default True.
    p : float, optional
        Probability parameter for Bernoulli(1) before mapping to ±1. Default 0.5.

    Returns
    -------
    A : (L, N) complex128 ndarray
        Complex Bernoulli measurement matrix.
        
    Example:
    --------
    A = bernoulli_mm(16, 256, unit_norm=True, rng=np.random.default_rng(42))
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # Draw signs in {+1, -1}
    sr = 1 - 2 * rng.binomial(1, p, size=(L, N))
    si = 1 - 2 * rng.binomial(1, p, size=(L, N))

    A = (sr + 1j * si) / np.sqrt(2.0)
    A = A.astype(np.complex128, copy=False)

    if unit_norm:
        norms = np.linalg.norm(A, axis=0)
        # Guard against any zero-norm columns (shouldn't occur with this construction)
        norms = np.where(norms == 0, 1.0, norms)
        A = A / norms

    return A


#%%
class ActivityDetetionDataset:
    # For MMV model: Y = A X + E 
    def __init__(
        self,
        NRSIM,   # number of independent simulations in the dataset.
        N,       # number of devices (dictionary size, so A.shape[1])
        M,       # number of antennas (so X.shape[1])
        K,       # sparsity (number of nonzero rows of X)
        L,       # pilot length (dimensinality, so Y.shape[0])
        SNRlow,  # SNR low in dB of device powers
        SNRhigh, # SNR high in dB of device powers
        use_bernoulli=True, # Bernoulli pilots
        p=0.5,           # probability of Bernoulli pilot 
        unit_norm=False, # normalize pilots to unit norm (default = False)
        rng=None,        # random number generator    
    ):
        """
        Dataset that generates NRSIM independent realizations of (Y, A, loc).

        Each call to __getitem__ generates fresh data (i.e., no pre-storage).
        """
        self.NRSIM = NRSIM
        self.N = N
        self.M = M
        self.K = K
        self.L = L
        self.SNRlow = SNRlow
        self.SNRhigh = SNRhigh
        self.use_bernoulli = use_bernoulli
        self.p = p
        self.unit_norm = unit_norm
        
        # Store integer seeds, not Generator objects
        if rng is None:
            rng = np.random.default_rng()
        self.seeds = rng.integers(0, 2**32 - 1, size=NRSIM)

    def __len__(self):
        return self.NRSIM

    def __getitem__(self, idx):
        # create a local RNG inside worker process
        rng_i = np.random.default_rng(self.seeds[idx])

        # generate random SNRs 
        gamma_dB = self.SNRlow + (self.SNRhigh - self.SNRlow) * rng_i.random(self.K)  # shape (K,)
        gamma = 10 ** (gamma_dB / 10)  # linear scale
        
        Y, A, loc, S= generate_AD_dataset(
            N=self.L, # <- A.shape[0], so rows of A
            M=self.N, # <- A.shape[1], so columns of A
            K=self.K, # sparsity
            L=self.M, # number of snapshots
            gam=gamma,
            use_bernoulli=self.use_bernoulli,
            p=self.p,
            unit_norm=self.unit_norm,
            rng=rng_i,   # <-- your RNG passed here
        )
        return Y, A, loc, S  # <-- CHANGE HERE    


def generate_AD_dataset(N, M, K, L, gam=None, *,
                  use_bernoulli=True,
                  p=0.5,
                  unit_norm=True,
                  rng=None,
                  seed=None):
    """
    Generate complex measurements Y = A[:, loc] @ S + E.

    Args:
        N (int): number of rows of A (measurement dimension).
        M (int): number of columns of A (dictionary size).
        K (int): number of active columns (sparsity level).
        L (int): number of snapshots.
        gam (ndarray|None): powers of active sources; shape (K,). If None, all ones are used.
        use_bernoulli (bool): if True, use Bernoulli matrix; else Gaussian.
        p (float): probability of +1 entry for Bernoulli matrix.
        unit_norm (bool): whether to unit-normalize columns of A.
        seed (int|None): seed for internal RNG if rng not provided.
        rng (np.random.Generator|None): external RNG. If None, a new one is created.

    Returns:
        Y (N x L complex ndarray): measurements.
        A (N x M real ndarray): sensing/measurement matrix.
        loc (K-dim ndarray): sorted active column indices in A.
        S (K x L complex ndarray): source matrix.
        
    Example usage:
    Y, A, loc, S, rng = generate_data(N=128, M=512, K=10, L=20, use_bernoulli=True, p=0.5, unit_norm=True, seed=42)
    """
    created_rng = False
    if rng is None:
        rng = np.random.default_rng(seed)
        created_rng = True

    # Source matrix S (complex circularly-symmetric Gaussian), optionally transformed by Lam
    S_base = (rng.standard_normal((K, L)) + 1j * rng.standard_normal((K, L))) / np.sqrt(2.0)
    Lam = np.diag(np.sqrt(gam)) if gam is not None else None
    S = (Lam @ S_base) if Lam is not None else S_base

    # Measurement matrix A
    if use_bernoulli:
        A = bernoulli_mm(N, M, unit_norm=unit_norm, p=p, rng=rng)
    else:
        A = gaussian_mm(N, M, unit_norm=unit_norm, rng=rng)

    # Noise E (complex circularly-symmetric Gaussian)
    E = (rng.standard_normal((N, L)) + 1j * rng.standard_normal((N, L))) / np.sqrt(2.0)

    # Active column locations
    loc = np.sort(rng.choice(M, size=K, replace=False))

    # Measurements
    Y = A[:, loc] @ S + E
    if created_rng:
        return Y, A, loc, S, rng
    else:
        return Y, A, loc, S
