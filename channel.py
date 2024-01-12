import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh

# Constants for simulation
E_signal = 1  # Energy of the signal
N0 = 1  # Noise power spectral density
N_MC_Iteration = 100000  # Number of Monte Carlo iterations/samples
L_values = [1, 2, 3, 4]  # Number of multipath components

# Define the bins for the histogram
max_snr = 5
bins = np.linspace(0, max_snr, 1000)  # Creates bins from 0 to max_snr

# Simulation
plt.figure()

for L in L_values:
    # Simulate L independent Rayleigh fading paths and sum their powers
    g = np.sqrt(E_signal / (2 * L)) * (np.random.randn(L, N_MC_Iteration) + 1j * np.random.randn(L, N_MC_Iteration))
    total_power = np.sum(np.abs(g)**2, axis=0)

    # Calculate histogram
    counts, bin_edges = np.histogram(total_power, bins=bins)
    bin_widths = np.diff(bin_edges)
    pdf = counts / (N_MC_Iteration*bin_widths)  # Scale counts to get density

    # Plot the manually scaled histogram (PDF)
    plt.bar(bin_edges[:-1], pdf, width=bin_widths, alpha=0.5, label=f'L={L}')

# Overlay the theoretical Rayleigh PDF for a large L as a reference
beta = 1  # Assuming beta is the total signal energy
rayleigh_x = np.linspace(0, max_snr, 1000)
rayleigh_pdf = rayleigh.pdf(rayleigh_x, scale=np.sqrt(beta / 2))
plt.plot(rayleigh_x, rayleigh_pdf, 'k-', label='Theoretical Rayleigh PDF')

# Final plot formatting
plt.xlabel('|g|', fontsize=14)
plt.ylabel('Probability Mass Function', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
