import seaborn as sns
import numpy as np

# Example data
data = np.random.rand(10, 10)  # Third variable as intensity

sns.heatmap(data, cmap='coolwarm', annot=True)
plt.title('Heatmap')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
