import numpy as np

weights = {
    "dummy": np.random.randn(1)
}

np.savez("model_weights.npz", **weights)
print("Dummy weights saved")
