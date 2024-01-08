import numpy as np
import matplotlib.pyplot as plt

def BERRayleighFadingChannel():
    N = 10**6  # number of bits or symbols

    # Transmitter
    ip = np.random.rand(1, N) > 0.5  # generating 0,1 with equal probability
    s = 2 * ip - 1  # BPSK modulation 0 -> -1; 1 -> 1

    Eb_N0_dB = np.arange(-20, 36)  # multiple Eb/N0 values
    nErr = np.zeros(len(Eb_N0_dB))  # Error counter

    for ii in range(len(Eb_N0_dB)):
        # White gaussian noise, 0dB variance
        n = 1 / np.sqrt(2) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        # Rayleigh channel
        h = 1 / np.sqrt(2) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))

        # Channel and noise noise addition
        y = h * s + 10 ** (-Eb_N0_dB[ii] / 20) * n

        # Equalization
        yHat = y / h

        # Receiver - hard decision decoding
        ipHat = np.real(yHat) > 0
        
        # Counting the errors using bitwise XOR for boolean arrays
        nErr[ii] = np.count_nonzero(ip ^ ipHat)  # Corrected line


    simBer = nErr / N  # simulated ber
    EbN0Lin = 10.0**(Eb_N0_dB / 10.0)
    theoryBer = 0.5 * (1 - np.sqrt(EbN0Lin / (EbN0Lin + 1)))

    # plot
    plt.figure()
    plt.semilogy(Eb_N0_dB, theoryBer, 'bp-', color = 'magenta', linewidth=2, label='Rayleigh-Theory')
    plt.semilogy(Eb_N0_dB, simBer, color='blue', linewidth=2, label='Rayleigh-Simulation')
    plt.axis([-3, 35, 10**-5, 0.5])
    plt.grid(True)
    plt.legend()
    plt.xlabel('SNR, dB')
    plt.ylabel('Bit Error Rate')
    plt.title('SISO BER for BPSK modulation in Rayleigh channel')
    plt.show()

# Call the function to run the simulation
BERRayleighFadingChannel()
