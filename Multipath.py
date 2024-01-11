import matplotlib.pyplot as plt

import numpy as np



# Define the coordinates of the transmitter and receiver

tx = [0, 0]

rx = [10, 0]



# Define points that represent the reflection points for the multipath effect

reflection_points = np.array([

    [3, 2],

    [7, 2],

    [5, -3],

    [2, -2],

    [8, -2]

])



# Start the plot

plt.figure(figsize=(10, 5))

plt.plot([tx[0], rx[0]], [tx[1], rx[1]], 'k--', label="Line-of-sight not available")



# Plot the transmitter and receiver

plt.scatter([tx[0], rx[0]], [tx[1], rx[1]], color='black', s=100, marker='^', label='Antennas')



# Plot the paths

for point in reflection_points:

    plt.plot([tx[0], point[0], rx[0]], [tx[1], point[1], rx[1]], color='blue')



# Annotations for distances, just examples

#plt.text(tx[0]-0.5, tx[1]-0.5, "$d_1$", ha='center', va='center')

#plt.text(rx[0]+0.5, rx[1]-0.5, "$d_i$", ha='center', va='center')

plt.text(reflection_points[-1][0], reflection_points[-1][1]+0.5, "$d_L$", ha='center', va='center')



# Enhance plot

plt.xlabel('Distance')

plt.ylabel('Distance')

plt.legend()

plt.grid(True)

plt.axis('equal')  # Ensure the scale is the same on both axes



# Show the plot

plt.show()