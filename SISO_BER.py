import numpy as np
import matplotlib.pyplot as plt

# number of bits or symbols
N = 10**6  

# transmitted signal
binary = np.random.rand(1, N) > 0.5  # generating 0,1 with equal probability
# BPSK modulation of the transmitted signal 0 -> -1; 1 -> 1
s = 2 * binary - 1  

SNR_Range = np.arange(-10, 30.1, 0.1)
nErr = np.zeros(len(SNR_Range)) 

for idx, SNR_Value in enumerate(SNR_Range):
    error_count = 0
    SNR_Linear = 10.0**(SNR_Value / 20.0)
    # White gaussian noise, 0dB variance
    n = 1 / np.sqrt(2) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
    # Rayleigh channel
    h = 1 / np.sqrt(2) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))

    # Channel and noise noise addition
    y = h * s + n * SNR_Linear

    # Equalization
    yHat = y / h

    # Receiver - hard decision decoding
    signal_Hat = np.real(yHat) > 0
        
    # Counting the errors using bitwise XOR for boolean arrays
    nErr[idx] = np.count_nonzero(binary ^ signal_Hat)  # Corrected line


    simulated_BER = nErr / N  # simulated ber
    SNR_Lin = 10.0**(SNR_Range / 10.0)
    theory_BER = 0.5 * (1 - np.sqrt(SNR_Lin / (SNR_Lin + 1)))

# plot
plt.figure()
plt.semilogy(SNR_Range, theory_BER, 'bp-', color = 'magenta', linewidth=2, label='Rayleigh-Theory')
plt.semilogy(SNR_Range, simulated_BER, color='blue', linewidth=2, label='Rayleigh-Simulation')
plt.grid(True)
plt.legend()
plt.xlabel('SNR, dB')
plt.ylabel('Bit Error Rate')
plt.title('SISO BER for BPSK modulation in Rayleigh channel')
plt.show()
