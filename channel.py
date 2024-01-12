import numpy as np
import matplotlib.pyplot as plt

# Constants for simulation
E_signal = 1  # Energy of the signal
N0 = 1  # Noise power spectral density
N_MC_Iteration = 10000  # Number of Monte Carlo iterations/samples
non_line_of_sight_path_counts = [1, 2, 4]  # Example path counts

# Simulation and plotting
plt.figure(figsize=(12, 7))

for M in non_line_of_sight_path_counts:
    # Simulate the channel
    h = np.sqrt(E_signal / (2 * M)) * (np.random.randn(M, N_MC_Iteration) + 1j * np.random.randn(M, N_MC_Iteration))
    # Simulate the noise
    noise = np.sqrt(N0 / 2) * (np.random.randn(M, N_MC_Iteration) + 1j * np.random.randn(M, N_MC_Iteration))
    # Calculate SNR using MRC
    SNR_MRC = np.sum(np.abs(h)**2, axis=0) / N0
    # Plot histogram
    plt.hist(SNR_MRC, bins=200, density=True, alpha=0.5, label=f'L={M}')

# Overlay the Rayleigh PDF
sigma = np.sqrt(E_signal / 2)  # Scale parameter for Rayleigh distribution
rayleigh_x = np.linspace(0, 5 * sigma, 1000)
rayleigh_pdf = (rayleigh_x / sigma**2) * np.exp(-rayleigh_x**2 / (2 * sigma**2))
plt.plot(rayleigh_x, rayleigh_pdf, 'r-', label='Rayleigh PDF')

# Final plot formatting
plt.xlabel('|g|', fontsize=14)
plt.ylabel('Probability Mass Function', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
