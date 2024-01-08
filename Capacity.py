import numpy as np
from scipy.integrate import quad
from scipy.optimize import newton
import matplotlib.pyplot as plt

# Define SNR values in dBs
SNR_dB = np.arange(1, 31)
# Define sample length
length = 10**6
# Allocate output arrays
csi_rx = np.zeros(len(SNR_dB))
csi_trx = np.zeros(len(SNR_dB))
# Convert to standard deviations for scaling
n_pow = 2./(10.**(SNR_dB/10))
# Calculate AWGN capacity, channel information irrelevant
awgn_c = np.log2(1+(2./n_pow))

# Function to calculate the integral for CSI at RX
def csi_rx_integral(gamma, gamma_bar):
    return np.log2(1 + gamma) * (np.exp(-gamma / gamma_bar) / gamma_bar)

# Function for the constraint to find gamma_0
def constraint(gamma_0, gamma_bar):
    result, _ = quad(lambda gamma: (1./gamma_0 - 1./gamma) * 
                     (np.exp(-gamma / gamma_bar) / gamma_bar), gamma_0, np.inf)
    return result - 1

for j, snr in enumerate(SNR_dB):
    # Generate in-phase and quadrature components of the signal
    i = np.random.randn(length) / np.sqrt(n_pow[j])
    q = np.random.randn(length) / np.sqrt(n_pow[j])
    # Create fading component with unit variance
    h = (i + 1j*q)
    h_pow = np.abs(h)**2
    gamma_bar = np.mean(h_pow)
    
    # CSI at RX only
    csi_rx[j], _ = quad(csi_rx_integral, 0, np.inf, args=(gamma_bar,))
    
    # CSI at both TX and RX
    # Solve for gamma_0 using the constraint function
    gamma_0 = newton(constraint, 1e-100, args=(gamma_bar,))
    # Calculate the capacity given gamma_0
    csi_trx[j], _ = quad(lambda gamma: np.log2(gamma / gamma_0) * 
                         (np.exp(-gamma / gamma_bar) / gamma_bar), gamma_0, np.inf)

# Plot results
plt.figure()
plt.plot(SNR_dB, awgn_c, 'r-.', linewidth=2)
plt.plot(SNR_dB, csi_rx, 'b', linewidth=2)
plt.plot(SNR_dB, csi_trx, 'g', linewidth=2)
plt.xlabel('SNR(dB)')
plt.ylabel('Shannon Capacity (Bits/Sec/Hz)')
plt.title('Channel Capacity vs. SNR | Part II:Q1')
plt.legend(['AWGN Channel', 'Rayleigh Fading Channel w/ CSI@RX', 
            'Rayleigh Fading Channel w/ CSI@TXRX'], loc='upper left')
plt.grid(True)
plt.show()