import numpy as np
import matplotlib.pyplot as plt

# Channel parameters
non_line_of_sight_path_counts = [1, 4, 8, 12, 24]  # number of nonLoS pathes to simulate
N_MC_Iteration = 5000            # number of Monte Carlo simulation
E_signal = 1                     # signal energy per channel
N0 = 1                         # noise energy

# Prepare the plot
plt.figure()

# Constants
E_signal = 1  # Example signal energy
N0 = 1  # Example noise power spectral density
N_MC_Iteration = 1000  # Reduced number of Monte Carlo iterations/samples to save memory
non_line_of_sight_path_counts = [1, 2, 4]  # Different numbers of NLOS paths

# Loop over the different numbers of branches for the histogram
for M in non_line_of_sight_path_counts:
    h = np.sqrt(E_signal / (2 * M)) * (np.random.randn(M, N_MC_Iteration) + 1j * np.random.randn(M, N_MC_Iteration))
    a = (2 * np.round(np.random.rand(1, N_MC_Iteration)) - 1) + 1j * (2 * np.round(np.random.rand(1, N_MC_Iteration)) - 1)
    a = a / np.sqrt(2)
    y = h * np.tile(a, (M, 1)) + np.sqrt(N0 / 2) * (np.random.randn(M, N_MC_Iteration) + 1j * np.random.randn(M, N_MC_Iteration))
    H = np.sum(np.abs(h)**2, axis=0)  # Compute channel power without creating a large matrix
    SNR_MRC = H / N0

    # Plot histogram
    plt.hist(SNR_MRC, bins=200, alpha=0.7, label=f'L={M}', density=True)

# Calculate and plot the theoretical Rayleigh distribution
sigma = np.sqrt(E_signal / 2)  # Scale parameter for Rayleigh distribution
rayleigh_x = np.linspace(0, np.max(SNR_MRC), N_MC_Iteration)
rayleigh_pdf = (rayleigh_x / sigma**2) * np.exp(-rayleigh_x**2 / (2 * sigma**2))

# Plot the Rayleigh distribution
plt.plot(rayleigh_x, rayleigh_pdf, 'r-', label='Rayleigh PDF')

# Plot formatting
plt.xlabel('|g|')
plt.ylabel('Probability Density Function')
plt.title('Histogram of SNR with Rayleigh PDF')
plt.legend()
plt.grid(True)
plt.show()

# CDF subplot
plt.figure()

# Loop over the different numbers of branches for the histogram
for M in non_line_of_sight_path_counts:
    h = np.sqrt(E_signal / (2 * M)) * (np.random.randn(M, N_MC_Iteration) + 1j * np.random.randn(M, N_MC_Iteration))
    a = (2 * np.round(np.random.rand(1, N_MC_Iteration)) - 1) + 1j * (2 * np.round(np.random.rand(1, N_MC_Iteration)) - 1)
    a = a / np.sqrt(2)
    y = h * np.tile(a, (M, 1)) + np.sqrt(N0 / 2) * (np.random.randn(M, N_MC_Iteration) + 1j * np.random.randn(M, N_MC_Iteration))
    H = np.diag(np.dot(h.conj().T, h))
    SNR_MRC = np.abs(H)**2 / (np.abs(H) * N0)

    # Plot CDF
    sorted_SNR = np.sort(SNR_MRC.ravel())
    cdf = np.linspace(1/len(sorted_SNR), 1.0, len(sorted_SNR))
    plt.plot(sorted_SNR, cdf, label=f'M={M}')

plt.xlabel('Gain')
plt.ylabel('CDF')
plt.title('CDF of Gain')
plt.legend()
plt.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
