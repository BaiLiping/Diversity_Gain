import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.integrate import quad
# Parameters
SNR_dB = np.linspace(-10, 30, 1000)  # SNR range from -10 to 30 dB
M = [1, 2, 4, 8]  # Diversity order M
R = 1  # Rate in bits per symbol
N0 = 1  # Assuming N0 (noise power spectral density) is 1 for simplicity
q = 1   # Assuming q as 1 for simplicity
beta = 0.5   # Assuming β as 1 for simplicity
computation_interval = 100000

# Convert SNR from dB to linear scale for β
SNR_linear = 10 ** (SNR_dB / 10)

# the capacity is computed according to equation 5.32
def C(g_square):
    return np.log2(1 + q * g_square / N0)

# the pdf of Norm_2(g) is according to equation 5.33
def f(x, M_current, beta):
    return x**(M_current-1) * np.exp(-x/beta) / (beta**M_current * factorial(M_current-1))

# the definition of P_out_exact is given by equation 5.34
# in order to compute the exact outage probability, we sample 1e5 points from x

# Plotting the PDF function f for different values of M
plt.figure(figsize=(10, 6))
x_range = np.linspace(0, 20, 1000)  # x values range from 0 to 10
beta = 2
for M_current in M:
    # Calculate y values using the PDF function f for each M_current
    y_values = f(x_range, M_current, beta)
    # Plot the results
    plt.plot(x_range, y_values, label=f'M = {M_current}')

plt.title('PDF of Norm_2(g) for Different M Values, Beta=2')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()



# Updated calculation of P_out using the new formula
P_out = np.zeros((len(M), len(SNR_dB)))
P_out_upper = np.zeros((len(M), len(SNR_dB)))

for m_index, m in enumerate(M):
    P_out_upper[m_index, :] = ((2 ** R - 1) / SNR_linear) ** m / factorial(m)
    for SNR_index, SNR_value in enumerate(SNR_linear):
        P_out[m_index][SNR_index], error= quad(f, 0, 1, args=(m,SNR_value))

# Plotting
plt.figure(figsize=(10, 6))

# Plot for each diversity order M with dashed lines using the updated equation
colors = ['r','g','blue','magenta']
for m_index, m in enumerate(M):
    plt.semilogy(SNR_dB, P_out_upper[m_index, :], color=colors[m_index], linestyle='--')
    plt.semilogy(SNR_dB, P_out[m_index, :],color=colors[m_index], linestyle='-', label=f'M = {m}')

plt.title('SIMO exact outage probabilities and the upper bound')
plt.xlabel('SNR [dB]')
plt.ylabel(r'$P_{out}$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.show()


P_out_MISO = np.zeros((len(M), len(SNR_dB)))

for m_index, m in enumerate(M):
    P_out_MISO[m_index, :] = ((2 ** R - 1) / SNR_linear) ** m / factorial(m)


# Plotting
plt.figure(figsize=(10, 6))

# Plot for each diversity order M with dashed lines using the updated equation
colors = ['r','g','blue','magenta']
for m_index, m in enumerate(M):
    plt.semilogy(SNR_dB, P_out_upper[m_index, :], color=colors[m_index], linestyle='--')
    plt.semilogy(SNR_dB, P_out[m_index, :],color=colors[m_index], linestyle='-', label=f'M = {m}')

plt.title('SIMO exact outage probabilities and the upper bound')
plt.xlabel('SNR [dB]')
plt.ylabel(r'$P_{out}$')
plt.legend()
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.show()
