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
N_MC_Iteration = 800000
R = 1
beta = 1  

P_out_non_cor = []
P_out_cor = []
for SNR_Value in SNR_Range:
    # convert from dB scale to linear scale 
    SNR_Linear = 10.0 ** (SNR_Value / 10.0)
    # set correlation coefficient  
    a_correlated = 0.8
    a_non_correlated = 0
    # sample g_squared
    g_squared_non_cor = sample_correlated_g_squared(beta,a_non_correlated, size=N_MC_Iteration)
    g_squared_cor = sample_correlated_g_squared(beta,a_correlated,size=N_MC_Iteration)
    # compute capacity
    capacity_non_cor = np.log2(1 + SNR_Linear * g_squared_non_cor)
    capacity_cor = np.log2(1 + SNR_Linear * g_squared_cor)
    # compute out for given SNR
    outage_count_non_cor = sum(1 for element in capacity_non_cor if element < R)
    outage_count_cor = sum(1 for element in capacity_cor if element < R)
    Pout_SNR_non_cor = outage_count_non_cor / N_MC_Iteration
    Pout_SNR_cor = outage_count_cor / N_MC_Iteration
    # store result
    P_out_non_cor.append(Pout_SNR_non_cor)
    P_out_cor.append(Pout_SNR_cor)
# plot
plt.figure()
plt.semilogy(SNR_Range, P_out_non_cor, linewidth=2, label='i.i.d Rayleigh Channels')
plt.semilogy(SNR_Range, P_out_cor, linewidth=2, label='Correlated Rayleigh Channels')
arrow_properties = dict(facecolor='black', shrink=0.05)
plt.annotate('Downward shift', xy=(5, P_out_cor[30]), xytext=(10, 10**(-1)),arrowprops=arrow_properties, fontsize=20)
# Function to find the closest index to a given SNR value in the SNR_Range array
def find_closest_index(value, array):
    index = np.abs(array - value).argmin()
    return index

# Get the index for SNR = 5dB and SNR = 4.5dB
index_5dB = find_closest_index(5, SNR_Range)
index_4_5dB = find_closest_index(4.5, SNR_Range)

# Compute Pout for SNR = 5dB and SNR = 4.5dB for both curves
p_out_5dB_non_cor = P_out_non_cor[index_5dB]
p_out_4_5dB_non_cor = P_out_non_cor[index_4_5dB]

p_out_5dB_cor = P_out_cor[index_5dB]
p_out_4_5dB_cor = P_out_cor[index_4_5dB]

# Compute the slope for both sets of points (in log scale for P_out)
slope_non_cor = (np.log10(p_out_4_5dB_non_cor) - np.log10(p_out_5dB_non_cor)) / (4.5 - 5)
slope_cor = (np.log10(p_out_4_5dB_cor) - np.log10(p_out_5dB_cor)) / (4.5 - 5)

# Extend the lines to the left and right of SNR = 5dB
snr_extended = np.array([SNR_Range[index_5dB - 10], SNR_Range[index_5dB + 10]])

# Calculate the y-values for the extended lines based on the slope
p_out_non_cor_extended = 10 ** (np.log10(p_out_5dB_non_cor) + slope_non_cor * (snr_extended - 5))
p_out_cor_extended = 10 ** (np.log10(p_out_5dB_cor) + slope_cor * (snr_extended - 5))

plt.semilogy(snr_extended, p_out_non_cor_extended, 'k--', label = "slop line for i.i.d")
plt.semilogy(snr_extended, p_out_cor_extended, 'r--', label='slop line for correlated')


plt.grid(True)
plt.legend(fontsize=15, loc='lower left')
plt.xlabel('SNR, dB', fontsize=14)
plt.ylabel('Pout', fontsize=14)
plt.show()
