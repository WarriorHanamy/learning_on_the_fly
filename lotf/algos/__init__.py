from lotf.algos import bptt, shac


def get_train_fn(window_size: int):
    """Return the appropriate training function based on window_size.

    window_size = 0  → full-trajectory BPTT (bptt.train)
    window_size > 0  → short-horizon actor-critic (shac.train)
    """
    return shac.train if window_size > 0 else bptt.train
