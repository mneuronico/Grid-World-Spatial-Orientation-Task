import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_fns import *

main_dir = r"N:\XAI - LLM GridWorld\Ablation Exps All" #switch to 4 or 6 or 7
dataset_path = os.path.join(main_dir, "dataset.csv")
df = pd.read_csv(dataset_path)

# Collect useful information
dataset_info = {
    "Total Trials": len(df),
    "Successful Trials": df["success"].sum() if "success" in df else "N/A",
    "Failure Trials": df["failure"].sum() if "failure" in df else "N/A",
    "Representations": df["representation"].unique().tolist(),
    "World Size": df["world_size"].unique().tolist(),
    "Max Steps": df["max_steps"].unique().tolist(),
    "Models Tested": df["model_tested"].unique().tolist(),
}

# Print dataset summary
for key, value in dataset_info.items():
    print(f"{key}: {value}")
print()



grouped = df.groupby('ablated_type')['success'].agg(['mean', 'count', 'std'])
grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
print(grouped)

plt.figure(figsize=(8, 6))
plt.bar(grouped.index, grouped['mean'], yerr=grouped['sem'], capsize=5, color='skyblue', edgecolor='black')
plt.xlabel("Ablated Type")
plt.ylabel("Success Rate")
plt.title("Success Rate by Ablated Type with Standard Error")
plt.ylim(0, 1)  # Success rates range from 0 to 1
plt.show()