import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
    try:
        return np.array(a).reshape(new_shape).tolist()
    except:
        return []
