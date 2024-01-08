import numpy as np
import matplotlib.pyplot as plt

# Channel parameters
branch_counts = [1, 4, 8, 12, 24, 50]  # number of branches to simulate
N = 3000                    # number of channel realizations
Es = 1                     # signal energy per channel
N0 = 0.2 
gamma = Es / N0             # average SNR per channel

# Prepare the plot
plt.figure(figsize=(14, 7))

# Histogram subplot
plt.subplot(1, 2, 1)

# Loop over the different numbers of branches for the histogram
for M in branch_counts:
    h = np.sqrt(Es / (2 * M)) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
    a = (2 * np.round(np.random.rand(1, N)) - 1) + 1j * (2 * np.round(np.random.rand(1, N)) - 1)
    a = a / np.sqrt(2)
    y = h * np.tile(a, (M, 1)) + np.sqrt(N0 / 2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
    H = np.diag(np.dot(h.conj().T, h))
    SNR_MRC = np.abs(H)**2 / (np.abs(H) * N0)

    # Plot histogram
    plt.hist(SNR_MRC.ravel(), bins=50, alpha=0.7, label=f'L={M}')

plt.xlabel('Gain')
plt.ylabel('Frequency')
plt.title('Histogram of Gain')
plt.legend()
plt.grid(True)

# CDF subplot
plt.subplot(1, 2, 2)

# Loop over the different numbers of branches for the CDF
for M in branch_counts:
    h = np.sqrt(Es / (2 * M)) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
    a = (2 * np.round(np.random.rand(1, N)) - 1) + 1j * (2 * np.round(np.random.rand(1, N)) - 1)
    a = a / np.sqrt(2)
    y = h * np.tile(a, (M, 1)) + np.sqrt(N0 / 2) * (np.random.randn(M, N) + 1j * np.random.randn(M, N))
    H = np.diag(np.dot(h.conj().T, h))
    SNR_MRC = np.abs(H)**2 / (np.abs(H) * N0)

    # Plot CDF
    sorted_SNR = np.sort(SNR_MRC.ravel())
    cdf = np.linspace(1/len(sorted_SNR), 1.0, len(sorted_SNR))
    plt.plot(sorted_SNR, cdf, label=f'L={M}')

plt.xlabel('Gain')
plt.ylabel('CDF')
plt.title('CDF of Gain')
plt.legend()
plt.grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
