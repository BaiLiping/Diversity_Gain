import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import factorial

def pdf_chi_squared(x, M, beta):
    # This is the PDF of ||g||^2 where g is a complex Gaussian random variable.
    return (x**(M-1) * np.exp(-x/beta)) / (beta**M * factorial(M-1))

def outage_probability(R, SNR_linear, M, beta):
    # Upper limit of the integral
    upper_limit = (2**R - 1) / SNR_linear
    # Calculate the outage probability by integrating the PDF of ||g||^2
    result, _ = quad(pdf_chi_squared, 0, upper_limit, args=(M, beta))
    return result

def sample_complex_gaussian(beta, size=1):
    real_part =np.random.normal(0,np.sqrt(beta/2), size)
    imag_part =np.random.normal(0,np.sqrt(beta/2), size)
    return real_part + 1j*imag_part

def sample_chi_squared(beta, M, size=1):
    size = (size, M)
    # Generate M complex 
    # samples from a 
    # complex normal distribution
    samples_complex_normal =sample_complex_gaussian(beta, size)

    # Calculate ||g||^2 by summing the squares f the magnitudes of the complex samples
    samples_chi_squared = np.sum(np.abs(samples_complex_normal)**2, axis=1)
    return samples_chi_squared

SNR_Range = np.arange(-10, 20, 0.5)
N_MC_Iteration = 10000000
R = 1
M_values = [1, 2, 4]
beta = 1
plt.figure()
for M in M_values:
    P_out = []
    P_out_theory = []
    for SNR_Value in SNR_Range:
        SNR_Linear = 10.0 ** (SNR_Value / 10.0)
        # sample g_squared
        g_squared = sample_chi_squared(beta, M, size=N_MC_Iteration)
        # compute capacity
        capacity = np.log2(1 + SNR_Linear * g_squared)
        # compute Pout given SNR
        outage_count = sum(1 for element in capacity if element < R)
        Pout_SNR = outage_count / N_MC_Iteration
        Pout_SNR_theory = outage_probability(R, SNR_Linear, M, beta)
        P_out.append(Pout_SNR)
        P_out_theory.append(Pout_SNR_theory)
    # Plot results for this value of M
    plt.semilogy(SNR_Range, P_out, linewidth=2, label=f'Simulation (M={M})')
    plt.semilogy(SNR_Range, P_out_theory, linestyle='--', linewidth=2, label=f'Theory (M={M})')
plt.grid(True)
plt.legend(fontsize=12)
plt.xlabel('SNR, dB', fontsize=14)
plt.ylabel('Pout', fontsize=14)
plt.title('Outage Probability for Different M')
plt.show()
