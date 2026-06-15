"""
mie_predictor.py — standalone Mie scattering inference using an ONNX model.
Dependencies: numpy, onnxruntime
"""

import numpy as np
import onnxruntime
import matplotlib.pyplot as plt


# Normalization constants — set these to match how the model was trained
N_MIN, N_MAX = 1.0, 1.5
K_MIN, K_MAX = 0.68, 1.00
X_MIN = np.pi * 50 / 650   # ≈ 3.43
X_MAX = np.pi * 1200 / 450   # ≈ 6.28

ASSUMED_WAVELENGTH = 500

class MiePredictor:
    def __init__(self, onnx_path):
        """Load an ONNX model from disk."""
        self.session = onnxruntime.InferenceSession(
            onnx_path, 
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
    
    def predict(self, x, n, k, angle=None):
        """
        Predict Mie scattering intensity.
        
        Args:
            x: size parameter (dimensionless)
            n: real part of refractive index
            k: imaginary part of refractive index
            angle: array of angles in radians. Defaults to 181 angles from 0 to π.
        
        Returns:
            numpy array of scattering intensities, one per angle.
        """
        if angle is None:
            angle = np.linspace(0, np.pi, 181)
        angle = np.asarray(angle)
        
        # Normalize inputs to [0, 1]
        n_norm = (n - N_MIN) / (N_MAX - N_MIN)
        k_norm = (k - K_MIN) / (K_MAX - K_MIN)
        x_norm = (x - X_MIN) / (X_MAX - X_MIN)
        theta_norm = angle / np.pi
        
        # Build (n_angles, 4) batch — one row per angle, n/k/x repeated
        batch = np.stack([
            np.full_like(theta_norm, n_norm),
            np.full_like(theta_norm, k_norm),
            np.full_like(theta_norm, x_norm),
            theta_norm
        ], axis=1).astype(np.float32)
        
        # Run inference
        log_pred = self.session.run(None, {self.input_name: batch})[0]
        
        # Un-log to get physical intensity
        return 10 ** log_pred.flatten()

x = 5.0

predictor = MiePredictor("NN_5_Derivatives_D(50-1200)_W(450-650)_N(1-1,5)_k(0.68-1).onnx")
curve = predictor.predict(x=x, n=1.2, k=1.1)
print(curve)

angles = np.linspace(0, 180, 181) * np.pi / 180
plt.figure(figsize=(10, 6))

plt.plot(angles, curve, label= f"x = {x}", color='red', linestyle='-')

plt.xlabel("Angle (radians)")
plt.ylabel("Scattering intensity (SU)")
plt.title(f"NN vs PyMieDiff outside of training range (n=1.6, k=1.0, λ=500nm) with derivatives")
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
