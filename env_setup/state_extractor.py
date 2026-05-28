import numpy as np

def extract_state(obs):
    """
    Receives a Grid2Op Observation object 
    and returns a flat (1D) Numpy vector ready to be fed into a neural network.
    """
    # 1. Extract critical metrics
    rho = obs.rho              # [Load] Capacity percentage of each power line (0.0 to 1.0+)
    gen_p = obs.gen_p          # [Generation] How many megawatts each power plant is currently producing
    load_p = obs.load_p        # [Demand] How many megawatts consumers are currently demanding
    
    # If there are missing values (NaN) in the environment, convert them to 0 to prevent the network from crashing
    rho = np.nan_to_num(rho, nan=0.0)
    gen_p = np.nan_to_num(gen_p, nan=0.0)
    load_p = np.nan_to_num(load_p, nan=0.0)

    # 2. Concatenate all arrays into a single flat vector
    state_vector = np.concatenate([rho, gen_p, load_p])
    
    # 3. Convert to float32 (the standard for fast computation with PyTorch models)
    return state_vector.astype(np.float32)