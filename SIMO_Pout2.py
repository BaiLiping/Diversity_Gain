import numpy as np
import matplotlib.pyplot as plt
import copy

def BERRayleighFadingChannel():

    SNR_Range = np.arange(-10, 30.1, 0.1) # multiple Eb/N0 values
    N_MC_Iteration = 3000
    
    R=1
    num_receive_antennas = [2]
    num_transmit_antennas = [1]
    P_out = []
    P_out_correlated = []
    
    for j  in num_transmit_antennas:
        for k in num_receive_antennas:
            P_out_jk=[]
            P_out_jk_correlated = []
            for ii in SNR_Range:
                outage_count_jk=0
                outage_count_jk_c = 0
                snr_linear_jk = 10.0**(ii / 10.0)
                for i in range(N_MC_Iteration):
        
                    # Generate a random channel matrix (Rayleigh fading)
                    h = np.random.normal(size=(1, 1)) + 1j * np.random.normal(size=(1, 1))
                    # correlated

                    
                    # Calculate channel capacity
                    
                    channel_capacity = np.log2(np.linalg.det(np.eye(k) + snr_linear_jk * np.matmul(h, h.conj().T)))
                    channel_capacity_c = np.log2(np.linalg.det(np.eye(k) + snr_linear_jk * np.matmul(h_c, h_c.conj().T)))

        
                    # Check if the system is in outage
                    if channel_capacity < R:
                        outage_count_jk += 1

                    if channel_capacity_c < R:
                        outage_count_jk_c +=1
        
                Pout_ii=outage_count_jk/N_MC_Iteration
                Pout_ii_correlated = outage_count_jk_c/N_MC_Iteration
                P_out_jk.append(Pout_ii)
                P_out_jk_correlated.append(Pout_ii_correlated)
            P_out.append(P_out_jk)
            P_out_correlated.append(P_out_jk_correlated)

    # plot
    plt.figure()
    #plt.semilogy(Eb_N0_dB, theoryBer, 'bp-', color = 'magenta', linewidth=2, label='Rayleigh-Theory')
    for i in range(1):   
        plt.semilogy(SNR_Range, P_out[i], linewidth=2, label='')
        plt.semilogy(SNR_Range, P_out_correlated[i], linewidth=2, label='')
    #plt.axis([-3, 35, 10**-5, 0.5])
    plt.grid(True)
    plt.legend()
    plt.xlabel('SNR, dB')
    plt.ylabel('Pout')
    plt.title('')
    plt.show()

# Call the function to run the simulation
BERRayleighFadingChannel()
