from typing import Any, Generator

import jax.numpy as np


def cast_to_tensor(*x: Any) -> Generator[np.ndarray, None, None]:
    return (xt if isinstance(xt, np.ndarray) else np.array(xt) for xt in x)
