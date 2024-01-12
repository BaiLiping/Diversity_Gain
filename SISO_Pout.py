import numpy as np
import matplotlib.pyplot as plt

SNR_Range = np.arange(-10, 30.1, 0.1)
N_MC_Iteration = 100000
R=1
beta=1
P_out = []
P_out_theory = []


def sample_exponential(beta, size=1):
    real=np.random.normal(0,np.sqrt(beta/2), size)
    imag=np.random.normal(0,np.sqrt(beta/2), size)
    # Calculate the magnitude squared
    magnitude_square =real**2 + imag**2
    return magnitude_square
  
for SNR_Value in SNR_Range:
    SNR_Linear = 10.0**(SNR_Value / 10.0)
    # sample g_square
    g_square = sample_exponential(beta, N_MC_Iteration)
    capacity = np.log2(1+SNR_Linear*g_square)
    # compute Pout given SNR
    outage_count = sum(1 for element in capacity if element < R)
    Pout_SNR=outage_count/N_MC_Iteration
    Pout_SNR_theory = 1 - np.exp(-(2**R - 1) / SNR_Linear)
    P_out.append(Pout_SNR)
    P_out_theory.append(Pout_SNR_theory)

# plot
plt.figure()
plt.semilogy(SNR_Range, P_out, linewidth=4, label='Simulation', )
plt.semilogy(SNR_Range, P_out_theory, linewidth=4, label='Theory')
plt.grid(True)
plt.legend(fontsize=14)
plt.xlabel('SNR, dB', fontsize=14)
plt.ylabel('Pout', fontsize=14)
plt.show()
