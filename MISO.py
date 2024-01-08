import numpy as np
import matplotlib.pyplot as plt

def BERMISOFadingChannel():
    N = 10**6  # number of bits or symbols

    # Transmitter
    ip_1 = np.random.rand(1, N) > 0.5  # Generating 0,1 with equal probability
    s_1 = 2 * ip_1 - 1  # BPSK modulation 0 -> -1; 1 -> 1
    ip_2 = np.random.rand(1, N)> 0.5
    s_2 = 2* ip_2 -1
    s=[s_1, s_2]

    nTx = 2  # Number of transmit antennas
    Eb_N0_dB = np.arange(0, 36)  # Multiple Eb/N0 values
    nErr = np.zeros(len(Eb_N0_dB))  # Error counter

    for ii, eb_n0_dB in enumerate(Eb_N0_dB):
        # White gaussian noise, 0dB variance
        n = 1 / np.sqrt(2) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
        # Rayleigh channel
        h = 1 / np.sqrt(2) * (np.random.randn(nTx, N) + 1j * np.random.randn(nTx, N))

        # MRT: Co-phasing the transmitted signal
        s_mrt = np.conj(h) / np.linalg.norm(h, axis=0) * s

        # Noise addition
        y = np.sum(h * s_mrt, axis=0) + 10**(-eb_n0_dB/20) * n

        # Receiver - hard decision decoding
        ipHat = np.real(y) > 0

        # Counting the errors
        nErr[ii] = np.count_nonzero(ip ^ ipHat)

    simBer = nErr / N  # Simulated BER

    # Theoretical BER
    EbN0Lin = 10.0**(Eb_N0_dB / 10.0)
    theoryBer_nTx2 = 0.5 * (1 - 2 * (1 + 1.0 / EbN0Lin)**(-0.5) + (1 + 2.0 / EbN0Lin)**(-0.5))

    # Plotting
    plt.figure()
    plt.semilogy(Eb_N0_dB, theoryBer, 'b-', linewidth=2, label='MISO Theory')
    plt.semilogy(Eb_N0_dB, simBer, 'mo-', linewidth=2, label='MISO Simulation')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Eb/No, dB')
    plt.ylabel('Bit Error Rate')
    plt.title('BER for BPSK modulation in MISO Rayleigh channel')
    plt.show()

# Run the simulation
BERMISOFadingChannel()