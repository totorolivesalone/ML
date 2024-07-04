import numpy as np
import matplotlib.pyplot as plt
import math

degrees = range(0 , 720)
cosValues = [math.cos(math.radians(i)) for i in degrees]
plt.plot(cosValues)
plt.xlabel('Degrees')
plt.ylabel('Cosine Values')
plt.title('Cosine Curve')
plt.grid()
plt.show()