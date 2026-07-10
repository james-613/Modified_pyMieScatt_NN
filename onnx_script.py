"""
mie_predictor.py — standalone Mie scattering inference using an ONNX model.
Dependencies: numpy, onnxruntime
"""

import numpy as np
import pymiediff as pmd
import onnxruntime
import matplotlib.pyplot as plt
import torch


class MiePredictor:

    # Normalization constants — set these to match how the model was trained
    N_MIN, N_MAX = 1.0, 1.9
    K_MIN, K_MAX = 0.5, 1.00
    X_MIN = np.pi * 100 / 650 
    X_MAX = np.pi * 1200 / 450 
    ASSUMED_WAVELENGTH = 500    # just for calculation purposes

    def __init__(self, onnx_path):
        # Load an ONNX model
        self.session = onnxruntime.InferenceSession(
            onnx_path, 
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
    
    def predict(self, x, n, k, angle=None):
        """
        Function to predict Mie scattering intensity with onnx file
        
        Args:
            x: size parameter
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
        n_norm = (n - self.N_MIN) / (self.N_MAX - self.N_MIN)
        k_norm = (k - self.K_MIN) / (self.K_MAX - self.K_MIN)
        x_norm = (x - self.X_MIN) / (self.X_MAX - self.X_MIN)
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
        # 181 intensity values for each angle in radians
        return 10 ** log_pred.flatten()

# wrapper function for generating pyMieDiff phase functions
def py_Mie_Diff_Scattering(wavelength, diameter, n, k, environment_value = 1.00):
    # - setup the particle
    wl0 = torch.tensor([wavelength]) # converting into what pyMieDiff requires as input
    k0 = 2 * torch.pi / wl0

    p = pmd.Particle(
        r_layers=[diameter/2],
        mat_layers=[n + k*1j],
        mat_env = 1.00
    )

    theta = torch.arange(0, 181) * torch.pi/181
    
    angle_scattering = p.get_angular_scattering(k0= k0, theta = theta)
    # returns 181 intensity values
    return angle_scattering['i_unpol']

# example of running the code
x = 4.3
n = 1.5
k = 0.99

# change to the name of the onnx file
predictor = MiePredictor("mie_scattering_NN_D(100-1200)_W(450-650)_N(1.0-1.9)_k(0.5-1.0).onnx")
curve_truth = predictor.predict(x=x, n=1.5, k=0.99)

# have to convert x to diameter and wavelength when running pyMieDiff
# assumed wavelength is 500
curve_NN = py_Mie_Diff_Scattering(500, (500 * x)/(np.pi), n, k)
print(curve_NN)

# plotting example
angles = np.linspace(0, 180, 181) * np.pi / 180
plt.figure(figsize=(10, 6))
plt.plot(angles, curve_truth, label= f"NN x = {x}", color='red', linestyle='--')
plt.plot(angles, curve_NN, label = f"PyMieDiff x = {x}", color= 'blue', linestyle = '-')
plt.xlabel("Angle (radians)")
plt.ylabel("Scattering intensity (SU)")
plt.title(f"NN vs PyMieDiff outside of training range (n=1.6, k=1.0, λ=500nm) with derivatives")
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
