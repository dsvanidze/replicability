import matplotlib.pyplot as plt
import numpy as np
import rasterio
import pandas as pd

# Create a list of evenly-spaced numbers over the range
x = np.linspace(0, 20, 100)
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()                   # Display the plot

print("Hello World")