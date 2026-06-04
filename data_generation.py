import numpy as np
import matplotlib.pyplot as plt

def determine_x(wavelength, diameter, n_val):
    return (2 * np.pi * n_val * diameter/2)/wavelength

N_PARTICLES = 1000

particle_x = np.zeros(N_PARTICLES)

for p in range(N_PARTICLES):
    wavelength_norm = np.random.uniform(0, 1)
    diameter_norm   = np.random.uniform(0, 1)
    n_norm          = np.random.uniform(0, 1)
    k_norm          = np.random.uniform(0, 1)

    wavelength_val = wavelength_norm * 1900 + 100
    diameter_val   = diameter_norm * 1980 + 20
    n_val          = n_norm * 3 + 0
    k_val          = k_norm * 1.1 + 0

    particle_x[p] = determine_x(wavelength_val, diameter_val, n_val)

plt.figure(figsize=(9, 5))
plt.hist(particle_x, bins=50)
plt.xlabel("Size parameter x")
plt.ylabel("Count")
plt.title("Distribution of x from linear sampling")
plt.grid(True, alpha=0.3)
plt.show()





