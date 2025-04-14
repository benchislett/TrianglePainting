import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def triangle_area(tri):
    """
    Compute the area of a triangle given its vertices.
    `tri` is a list of 6 numbers: [x0, y0, x1, y1, x2, y2].
    """
    x0, y0, x1, y1, x2, y2 = tri
    return abs(x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1)) / 2

def bounding_box_area(tri):
    """
    Compute the area of the axis-aligned bounding box for a triangle.
    """
    x0, y0, x1, y1, x2, y2 = tri
    min_x = min(x0, x1, x2)
    max_x = max(x0, x1, x2)
    min_y = min(y0, y1, y2)
    max_y = max(y0, y1, y2)
    return (max_x - min_x) * (max_y - min_y)

# Load JSON data (adjust the filename as needed)
with open('benchmark_results.json', 'r') as f:
    data = json.load(f)

rows = []
# Iterate over each record.
for record in data["records"]:
    runner = record["runner_name"]
    canvas_width = record["canvas_width"]
    # Here we assume that the record's "times" array has one value per input_sample.
    times = record["times"]
    input_samples = record["input_samples"]
    for i, sample in enumerate(input_samples):
        tri = sample["triangle"]
        area = triangle_area(tri)
        bbox_area = bounding_box_area(tri)
        # Compute ratio. Guard against division by zero.
        area_ratio = area / bbox_area if bbox_area > 0 else 0.0
        t = times[i]  # sample time in seconds
        # Compute iterations/second (ips) as the inverse of time.
        ips = 1.0 / t if t != 0 else 0.0
        rows.append({
            "canvas_width": canvas_width,
            "runner_name": runner,
            "triangle_area": area,
            "bbox_area": bbox_area,
            "area_ratio": area_ratio,
            "time": t,
            "ips": ips
        })

# Build a DataFrame.
df = pd.DataFrame(rows)
print(df.head())

# Set seaborn style.
sns.set(style="whitegrid")

# Smoothing parameter: adjust window size as needed.
rolling_window = 100

# Create a plot for each canvas_width.
unique_widths = df["canvas_width"].unique()
for width_val in unique_widths:
    df_sub = df[df["canvas_width"] == width_val].copy()
    
    # Sort by area_ratio (x-axis) and compute a smoothed ips value per runner.
    df_sub.sort_values(["runner_name", "area_ratio"], inplace=True)
    df_sub["ips_smoothed"] = df_sub.groupby("runner_name")["ips"].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=1).mean()
    )
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_sub,
        x="area_ratio",
        y="ips_smoothed",
        hue="runner_name",
        estimator=None,
        lw=2,
        marker=None
    )
    plt.title(f"Iterations/Second vs. Triangle Area/BBox Area Ratio (Canvas Width = {width_val})")
    plt.xlabel("Triangle Area / Bounding Box Area")
    plt.ylabel("Iterations/Second (Higher is Faster)")
    plt.legend(title="Runner", loc="best")
    plt.tight_layout()
    plt.savefig(f"performance_canvas_{width_val}.png")
    plt.show()
