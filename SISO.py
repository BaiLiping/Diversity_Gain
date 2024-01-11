import numpy as np
import matplotlib.pyplot as plt

def BERRayleighFadingChannel():

    Eb_N0_dB = np.arange(-20, 36)  # multiple Eb/N0 values
    N_MC_Iteration = 10000
    
    R=1
    P_out = []
    
    for ii in range(len(Eb_N0_dB)):
        counter_ii=0
        for i in range(N_MC_Iteration):

            # Rayleigh channel
            h_i =  (np.random.randn(1, 1) + 1j * np.random.randn(1, 1))
            EbN0Lin_ii = 10.0**(ii / 10.0)

    
            SNR_i= EbN0Lin_ii*h_i.conjugate()*h_i 
            RateMC_i=2*np.log2(SNR_i/2+1)
            
            if R>RateMC_i:
                counter_ii+=1

        Pout_ii=counter_ii/N_MC_Iteration
        P_out.append(Pout_ii)
    EbN0Lin = 10.0**(Eb_N0_dB / 10.0)
    theoryBer = 0.5 * (1 - np.sqrt(EbN0Lin / (EbN0Lin + 1)))

    # plot
    plt.figure()
    plt.semilogy(Eb_N0_dB, theoryBer, 'bp-', color = 'magenta', linewidth=2, label='Rayleigh-Theory')
    plt.semilogy(Eb_N0_dB, P_out, color='blue', linewidth=2, label='Rayleigh-Simulation')
    plt.axis([-3, 35, 10**-5, 0.5])
    plt.grid(True)
    plt.legend()
    plt.xlabel('SNR, dB')
    plt.ylabel('Bit Error Rate')
    plt.title('SISO BER for BPSK modulation in Rayleigh channel')
    plt.show()

# Call the function to run the simulation
BERRayleighFadingChannel()
