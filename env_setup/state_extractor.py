import numpy as np


def extract_state(obs):
    """
    Receives a Grid2Op Observation object and returns a flat (1D), NORMALIZED
    Numpy vector ready to be fed into a neural network.

    Normalization matters: rho is ~[0, 1] while gen_p / load_p are in MW (tens
    to hundreds). Without scaling, the MW features dominate fc1 and the rho
    safety signal gets buried, making learning very hard.
    """
    # 1. Extract critical metrics
    rho = np.nan_to_num(obs.rho, nan=0.0)          # line loading (already ~[0, 1+])
    gen_p = np.nan_to_num(obs.gen_p, nan=0.0)      # generation (MW)
    load_p = np.nan_to_num(obs.load_p, nan=0.0)    # demand (MW)

    # 2. Normalize the MW-scale features
    gen_pmax = np.asarray(obs.gen_pmax, dtype=np.float32)
    total_cap = float(np.sum(gen_pmax)) + 1e-6

    gen_p_norm = gen_p / (gen_pmax + 1e-6)         # ~[0, 1] per generator
    load_p_norm = load_p / total_cap               # fraction of total capacity

    # 3. Concatenate into a single flat vector (rho left as-is; >1 = overload is informative)
    state_vector = np.concatenate([rho, gen_p_norm, load_p_norm])

    return state_vector.astype(np.float32)
