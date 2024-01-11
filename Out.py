import numpy as np

def simulate_mimo_outage(num_transmit_antennas, num_receive_antennas, snr_db, threshold, num_trials):
    """
    Simulate the outage probability in a MIMO system.

    :param num_transmit_antennas: Number of transmit antennas
    :param num_receive_antennas: Number of receive antennas
    :param snr_db: Signal-to-Noise Ratio in dB
    :param threshold: SNR threshold for outage
    :param num_trials: Number of trials for the simulation
    :return: Outage probability
    """
    snr_linear = 10 ** (snr_db / 10)
    outage_count = 0

    for _ in range(num_trials):
        # Generate a random channel matrix (Rayleigh fading)
        h = np.random.normal(size=(num_receive_antennas, num_transmit_antennas)) \
            + 1j * np.random.normal(size=(num_receive_antennas, num_transmit_antennas))

        # Calculate channel capacity
        channel_capacity = np.log2(np.linalg.det(np.eye(num_receive_antennas) + snr_linear * np.matmul(h, h.conj().T)))

        # Check if the system is in outage
        if channel_capacity < threshold:
            outage_count += 1

    return outage_count / num_trials

# Example usage
num_transmit_antennas = 4
num_receive_antennas = 4
snr_db = 20  # SNR in dB
threshold = 10  # Threshold for outage
num_trials = 10000  # Number of simulation trials

outage_probability = simulate_mimo_outage(num_transmit_antennas, num_receive_antennas, snr_db, threshold, num_trials)
print("Outage Probability:", outage_probability)