import numpy as np
import matplotlib.pyplot as plt

def BERRayFadingChannel_Diversity():
    N = 10**6  # number of bits or symbols

    # Transmitter
    ip = np.random.rand(1, N) > 0.5  # generating 0,1 with equal probability
    s = 2 * ip - 1  # BPSK modulation 0 -> -1; 1 -> 1

    nRx = [1, 2]
    Eb_N0_dB = np.arange(-10, 36)  # multiple Eb/N0 values
    nErr = np.zeros((len(nRx)+1, len(Eb_N0_dB)))  # Error counter

    for jj, n_rx in enumerate(nRx):
        for ii, eb_n0_dB in enumerate(Eb_N0_dB):
            # White gaussian noise
            n = 1 / np.sqrt(2) * (np.random.randn(n_rx, N) + 1j * np.random.randn(n_rx, N))
            # Rayleigh channel
            h = 1 / np.sqrt(2) * (np.random.randn(n_rx, N) + 1j * np.random.randn(n_rx, N))

            # Channel and noise addition
            sD = np.kron(np.ones((n_rx, 1)), s)
            y = h * sD + 10 ** (-eb_n0_dB / 20) * n

            # Finding the power of the channel on all rx chains
            hPower = np.abs(h)**2

            # Selecting the chain with the maximum power
            ind = np.argmax(hPower, axis=0)
            ySel = y[ind, np.arange(N)]
            hSel = h[ind, np.arange(N)]

            # Equalization with the selected rx chain
            yHat_max = ySel / hSel

            # Removing the phase of the channel
            yHat_equal = y * np.exp(-1j * np.angle(h))
            
            # Adding values from all the receive chains
            yHat_equal = np.sum(yHat_equal, axis=0)

            # Receiver - hard decision decoding
            ipHat_max = np.real(yHat_max) > 0
            ipHat_equal = np.real(yHat_equal) > 0

            # Counting the errors
            nErr[jj, ii] = np.count_nonzero(ip ^ ipHat_max)  # Corrected line
            nErr[jj+1, ii] = np.count_nonzero(ip ^ ipHat_equal)  # Corrected line
        
    simBer = nErr / N  # Simulated BER

    # Theoretical BER
    EbN0Lin = 10.0**(Eb_N0_dB / 10.0)
    theoryBer_nRx1 = 0.5 * (1 - (1 + 1.0 / EbN0Lin)**(-0.5))
    theoryBer_nRx2_max = 0.5 * (1 - 2 * (1 + 1.0 / EbN0Lin)**(-0.5) + (1 + 2.0 / EbN0Lin)**(-0.5))
    theoryBer_nRx2_equal = 0.5 * (1 - np.sqrt(EbN0Lin * (EbN0Lin + 2)) / (EbN0Lin + 1))
    #p = 1/2 - 1/2 * (1 + 1.0 / EbN0Lin)**(-0.5)
    #theoryBer_nRx2 = p**2 * (1 + 2 * (1 - p))
    # Plotting
    plt.figure()
    plt.semilogy(Eb_N0_dB, theoryBer_nRx1, 'bp-', linewidth=2, label='nRx=1 (theory)')
    plt.semilogy(Eb_N0_dB, simBer[0, :], linewidth=2, label='nRx=1 (sim)')
    plt.semilogy(Eb_N0_dB, theoryBer_nRx2_max, 'rd-', linewidth=2, label='nRx=2 (theory selection)')
    plt.semilogy(Eb_N0_dB, simBer[1, :], linewidth=2, label='nRx=2 (sim selection)')
    plt.semilogy(Eb_N0_dB, theoryBer_nRx2_equal, 'rd-', linewidth=2, label='nRx=2 (theory equal)')
    plt.semilogy(Eb_N0_dB, simBer[2, :], linewidth=2, label='nRx=2 (sim equal)')
    
    plt.axis([0, 35, 1e-5, 0.5])
    plt.grid(True)
    plt.legend()
    plt.xlabel('Eb/No, dB')
    plt.ylabel('Bit Error Rate')
    plt.title('SIMO BER for BPSK modulation with Rayleigh channel')
    plt.show()

# Run the simulation
BERRayFadingChannel_Diversity()
