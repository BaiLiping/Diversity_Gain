import numpy as np
import matplotlib.pyplot as plt



def sample_correlated_g_squared(beta,a,size):
    # Covariance matrix
    cov_matrix = beta * np.array([[1, a], [a, 1]])
    mean = [0,0]
    # Sample from the multivariate normal distribution for the real and imaginary parts
    real_parts = np.random.multivariate_normal(mean, cov_matrix, size)
    imag_parts = np.random.multivariate_normal(mean, cov_matrix, size)
    correlated_g = real_parts + 1j * imag_parts
    correlated_g_squared = np.sum(np.abs(correlated_g)**2, axis=1)
    return correlated_g_squared
beta = 1 
SNR_Range = np.arange(-10, 20, 0.5)
N_MC_Iteration = 500000
R = 1
a = [0, 0.4,0.6, 0.8, 1]
beta = 1  

P_out = [[] for i in a]
for a_index, a_value in enumerate(a):
    for SNR_Value in SNR_Range:
        # convert from dB scale to linear scale 
        SNR_Linear = 10.0 ** (SNR_Value / 10.0)

        # sample g_squared
        g_squared = sample_correlated_g_squared(beta,a_value, size=N_MC_Iteration)
        # compute capacity
        capacity = np.log2(1 + SNR_Linear * g_squared)
        # compute out for given SNR
        outage_count = sum(1 for element in capacity if element < R)
        Pout_SNR_non_cor = outage_count / N_MC_Iteration
        # store result
        P_out[a_index].append(Pout_SNR_non_cor)
# plot
plt.figure()
for a_index, a_value in enumerate(a):
    plt.semilogy(SNR_Range, P_out[a_index], linewidth=2, label=f'Rayleigh Channels with coefficient {a_value}')


plt.grid(True)
plt.legend(fontsize=15, loc='lower left')
plt.xlabel('SNR, dB', fontsize=14)
plt.ylabel('Pout', fontsize=14)
plt.show()
