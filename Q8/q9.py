import matplotlib.pyplot as plt
import numpy as np
 
# Data for plotting
x = np.arange(0.0, 2.0, 0.01)
y = 1 + np.sin(2 * np.pi * x)
 
# Creating 6 subplots and unpacking the output array immediately
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(x, y, color="orange")
ax2.plot(x[3:30], y[3:30], color="green")
ax3.plot(x[30:90], y[30:90], color="blue")
ax4.plot(x[90:150], y[90:150], color="magenta")
plt.show()