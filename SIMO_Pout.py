import numpy as np
import matplotlib.pyplot as plt


def create_exponential_correlation_matrix(size, alpha=0.3):
    """
    Create an exponential correlation matrix for MIMO channels.

    :param size: The number of antennas (size of the matrix)
    :param alpha: Correlation factor (default is 0.3 for moderate correlation)
    :return: Correlation matrix
    """
    correlation_matrix = np.zeros((size, size), dtype=complex)
    for i in range(size):
        for j in range(size):
            correlation_matrix[i, j] = np.exp(-alpha * np.abs(i - j))
    return correlation_matrix

# Example usage
num_antennas = 4  # Example number of antennas
R_rx = create_exponential_correlation_matrix(2)  # Receiver correlation matrix
R_tx = create_exponential_correlation_matrix(1)  # Transmitter correlation matrix


def generate_correlated_channel(h, R_rx, R_tx):
    return np.dot(np.dot(R_rx, h), R_tx)






def BERRayleighFadingChannel():

    SNR_Range = np.arange(-20, 30.1, 0.1) # multiple Eb/N0 values
    N_MC_Iteration = 100000
    
    R=1
    num_receive_antennas = [1, 2]
    num_transmit_antennas = [1, 2]
    P_out = []
    
    for j  in num_transmit_antennas:
        for k in num_receive_antennas:
            P_out_jk=[]
            for ii in SNR_Range:
                outage_count_jk=0
                snr_linear_jk = 10.0**(ii / 10.0)
                for i in range(N_MC_Iteration):
        
                    # Generate a random channel matrix (Rayleigh fading)
                    h = np.random.normal(size=(k, j)) + 1j * np.random.normal(size=(k, j))
                    
                    H_w = np.random.normal(size=(k, j)) + 1j * np.random.normal(size=(k, j))  # Independent channel
        
                    # Calculate channel capacity
                    channel_capacity = np.log2(np.linalg.det(np.eye(k) + snr_linear_jk * np.matmul(h, h.conj().T)))
        
                    # Check if the system is in outage
                    if channel_capacity < R:
                        outage_count_jk += 1
        
                Pout_ii=outage_count_jk/N_MC_Iteration
                P_out_jk.append(Pout_ii)
            P_out.append(P_out_jk)

    # plot
    plt.figure()
    #plt.semilogy(Eb_N0_dB, theoryBer, 'bp-', color = 'magenta', linewidth=2, label='Rayleigh-Theory')
    for i in range(4):   
        plt.semilogy(SNR_Range, P_out[i], linewidth=2, label='')
    #plt.axis([-3, 35, 10**-5, 0.5])
    plt.grid(True)
    plt.legend()
    plt.xlabel('SNR, dB')
    plt.ylabel('Pout')
    plt.title('')
    plt.show()

# Call the function to run the simulation
BERRayleighFadingChannel()
