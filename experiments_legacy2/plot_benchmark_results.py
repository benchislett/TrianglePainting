import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")

# Load the CSV file
file_path = '/home/benchislett/Repos/TrianglePainting/build/benchmark_results.csv'
data = pd.read_csv(file_path)

# Pivot the data so that canvas_width becomes the x-axis groups, and runner_name becomes the adjacent bars within each group
data_pivot = data.pivot(index='canvas_width', columns='runner_name', values='iterations_per_second')
data_pivot.plot(kind='bar', figsize=(10, 6))

plt.title('Performance Comparison by Canvas Width and Runner Name', fontsize=16)
plt.xlabel('Canvas Width', fontsize=14)
plt.ylabel('Iterations per Second', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.show()