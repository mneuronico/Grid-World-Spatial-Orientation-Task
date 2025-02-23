import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import t
import matplotlib.pyplot as plt


save_to_pdf = False

main_dir = r"N:\XAI - LLM GridWorld\Experiment 2"
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

from matplotlib.backends.backend_pdf import PdfPages

# Use the pgf backend for matplotlib
plt.rcdefaults()

# Ensure that fonts are embedded in the PDF
plt.rcParams['pdf.fonttype'] = 42  # Output Type 3 (Type3) fonts
plt.rcParams['ps.fonttype'] = 42  # Output Type 3 (Type3) fonts

if save_to_pdf:
    # Dictionary to hold the PdfPages objects for each region
    pdf_file = PdfPages("Plots probing v1.pdf")



# Define the main directory and paths
dataset_path = os.path.join(main_dir, "dataset.csv")
paths_dir = os.path.join(main_dir, "paths")
mind_histories_dir = os.path.join(main_dir, "mind_histories")

# Load dataset
dataset = pd.read_csv(dataset_path)

# Filter dataset for trials with specific representations (ignore random trials)
filtered_trials = dataset[dataset["representation"] != "random"]

# Initialize storage for matrices and labels
X_matrices = {}
Y_matrices = {}

# Function to calculate Manhattan distance
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

# Iterate over each representation type
representations = filtered_trials["representation"].unique()
for representation in representations:
    print(f"Processing representation: {representation}")

    # Filter trials for this representation
    rep_trials = filtered_trials[filtered_trials["representation"] == representation]

    # Initialize lists to collect data for this representation
    layer_activations = []  # List of (examples, params) for each layer
    Y_representation = []  # For storing labels (correct/incorrect decisions)

    for _, trial in tqdm(rep_trials.iterrows(), total=len(rep_trials)):
        trial_id = trial["id"]
        goal_x, goal_y = trial["goal_x"], trial["goal_y"]

        # Load path file
        path_file = os.path.join(paths_dir, f"{trial_id}.csv")
        path_data = pd.read_csv(path_file)

        # Load mind history file
        mind_history_file = os.path.join(mind_histories_dir, f"{trial_id}.pkl")
        with open(mind_history_file, "rb") as f:
            mind_history = pickle.load(f)

        # Calculate distances and correctness for each step
        correct_decisions = []

        for i, row in path_data.iterrows():
            agent_x, agent_y = row["agent_x"], row["agent_y"]
            action = row["action"]
            step_number = row["step_number"]

            # Calculate current distance
            current_distance = manhattan_distance(agent_x, agent_y, goal_x, goal_y)

            if i < len(path_data) - 1:  # Non-final step
                next_agent_x = path_data.iloc[i + 1]["agent_x"]
                next_agent_y = path_data.iloc[i + 1]["agent_y"]
                next_distance = manhattan_distance(next_agent_x, next_agent_y, goal_x, goal_y)
            else:  # Final step, infer next position
                if action == "UP":
                    next_agent_x, next_agent_y = agent_x - 1, agent_y
                elif action == "DOWN":
                    next_agent_x, next_agent_y = agent_x + 1, agent_y
                elif action == "LEFT":
                    next_agent_x, next_agent_y = agent_x, agent_y - 1
                elif action == "RIGHT":
                    next_agent_x, next_agent_y = agent_x, agent_y + 1
                else:  # Invalid action
                    next_agent_x, next_agent_y = agent_x, agent_y

                next_distance = manhattan_distance(next_agent_x, next_agent_y, goal_x, goal_y)

            # Determine if the decision was correct
            correct_decisions.append(1 if next_distance < current_distance else 0)

        # Verify final step of successful trials
        if trial["success"] and next_distance != 0:
            raise ValueError(f"Inconsistent final distance for successful trial {trial_id}.")

        # Collect activations and labels
        for i, step in enumerate(mind_history):
            decision_activations = step["decision"]["activations"].mean(axis=1)  # (layers, params)
            if len(layer_activations) == 0:
                layer_activations = [[] for _ in range(decision_activations.shape[0])]  # One list per layer
            for layer_idx, layer_activation in enumerate(decision_activations):
                layer_activations[layer_idx].append(layer_activation)  # Append (params,) vector
            Y_representation.append(correct_decisions[i])

    # Convert layer-wise lists to numpy arrays
    X_representation = [np.array(layer) for layer in layer_activations]  # Each array is (examples, params)
    X_representation = np.stack(X_representation, axis=0)  # Final shape: (layers, examples, params)
    Y_representation = np.array(Y_representation)  # Shape: (examples,)

    # Save matrices for this representation
    X_matrices[representation] = X_representation
    Y_matrices[representation] = Y_representation

    # Print summary
    print(f"Representation: {representation}")
    print(f"X shape: {X_representation.shape}, dtype: {X_representation.dtype}, NaN elements: {np.isnan(X_representation).sum()}")
    print(f"Y shape: {Y_representation.shape}, dtype: {Y_representation.dtype}, NaN elements: {np.isnan(Y_representation).sum()}")


results_per_representation = {}

# Loop through each representation
for representation, X_representation in X_matrices.items():
    Y_representation = Y_matrices[representation]

    print(f"Processing representation: {representation}")

    # Precompute results for each layer
    regression_results = {}
    layers, examples, params = X_representation.shape

    for layer_idx in range(layers):
        print(f"Processing layer {layer_idx + 1}/{layers} for {representation}...")

        # Extract layer data: Shape (examples, params)
        layer_activations = X_representation[layer_idx]

        # Z-score across examples for each parameter
        layer_activations_z = (layer_activations - layer_activations.mean(axis=0)) / layer_activations.std(axis=0)

        # Store results for each parameter
        effect_sizes, errors, pvalues, corrected_pvalues = [], [], [], []

        bonferroni_correction_factor = params

        for param_idx in range(params):
            # Fit linear regression
            x = layer_activations_z[:, param_idx].reshape(-1, 1)
            y = Y_representation

            reg = LinearRegression().fit(x, y)
            beta = reg.coef_[0]
            y_pred = reg.predict(x)
            residual = y - y_pred

            # Compute standard error and t-statistic
            dof = len(y) - 2  # Degrees of freedom
            std_error = np.sqrt((residual ** 2).sum() / dof) / np.sqrt((x ** 2).sum())
            t_stat = beta / std_error

            # Compute p-value
            p_value = 2 * (1 - t.cdf(np.abs(t_stat), dof))

            # Apply Bonferroni correction
            corrected_p_value = min(p_value * bonferroni_correction_factor, 1.0)

            # Save results
            effect_sizes.append(beta)
            errors.append(std_error)
            pvalues.append(p_value)
            corrected_pvalues.append(corrected_p_value)

        # Store the results for this layer
        regression_results[layer_idx] = {
            "effect_sizes": np.array(effect_sizes),
            "errors": np.array(errors),
            "pvalues": np.array(pvalues),
            "corrected_pvalues": np.array(corrected_pvalues),
        }

        # Null Model: Shuffle labels
        shuffled_labels = np.random.permutation(Y_representation)
        null_effect_sizes, null_errors, null_pvalues, null_corrected_pvalues = [], [], [], []

        for param_idx in range(params):
            x = layer_activations_z[:, param_idx].reshape(-1, 1)
            y = shuffled_labels

            reg = LinearRegression().fit(x, y)
            beta = reg.coef_[0]
            y_pred = reg.predict(x)
            residual = y - y_pred

            dof = len(y) - 2
            std_error = np.sqrt((residual ** 2).sum() / dof) / np.sqrt((x ** 2).sum())
            t_stat = beta / std_error
            p_value = 2 * (1 - t.cdf(np.abs(t_stat), dof))
            corrected_p_value = min(p_value * bonferroni_correction_factor, 1.0)

            null_effect_sizes.append(beta)
            null_errors.append(std_error)
            null_pvalues.append(p_value)
            null_corrected_pvalues.append(corrected_p_value)

        # Store the results for this layer
        regression_results[layer_idx]["null_effect_sizes"] = np.array(null_effect_sizes)
        regression_results[layer_idx]["null_errors"] = np.array(null_errors)
        regression_results[layer_idx]["null_pvalues"] = np.array(null_pvalues)
        regression_results[layer_idx]["null_corrected_pvalues"] = np.array(null_corrected_pvalues)

        # Plot the results
        plt.figure(figsize=(10, 6))

        # Sort real model results
        real_sorted_indices = np.argsort(regression_results[layer_idx]["effect_sizes"])[::-1]
        real_effect_sizes_sorted = regression_results[layer_idx]["effect_sizes"][real_sorted_indices]
        real_corrected_pvalues_sorted = regression_results[layer_idx]["corrected_pvalues"][real_sorted_indices]

        # Sort null model results
        null_sorted_indices = np.argsort(regression_results[layer_idx]["null_effect_sizes"])[::-1]
        null_effect_sizes_sorted = regression_results[layer_idx]["null_effect_sizes"][null_sorted_indices]
        null_corrected_pvalues_sorted = regression_results[layer_idx]["null_corrected_pvalues"][null_sorted_indices]

        # Color logic for real model
        real_colors = []
        for idx in range(len(real_effect_sizes_sorted)):
            if real_corrected_pvalues_sorted[idx] < 0.05:
                if real_effect_sizes_sorted[idx] > 0:
                    real_colors.append("green")
                else:
                    real_colors.append("red")
            else:
                real_colors.append("gray")

        # Color logic for null model
        null_colors = []
        for idx in range(len(null_effect_sizes_sorted)):
            if null_corrected_pvalues_sorted[idx] < 0.05:
                if null_effect_sizes_sorted[idx] > 0:
                    null_colors.append("lightgreen")
                else:
                    null_colors.append("pink")
            else:
                null_colors.append("gray")

        # Plot non-shuffled
        plt.scatter(range(len(real_effect_sizes_sorted)), real_effect_sizes_sorted, color=real_colors, label="Original", alpha=0.7)

        # Plot shuffled
        plt.scatter(range(len(null_effect_sizes_sorted)), null_effect_sizes_sorted, color=null_colors, label="Shuffled", alpha=0.5)

        plt.title(f"Representation {representation} - Layer {layer_idx + 1} - Effect Sizes")
        plt.xlabel("Sorted Parameter Index")
        plt.ylabel("Effect Size")
        plt.legend()
        plt.show()

    results_per_representation[representation] = regression_results


    # Define the results directory and file path
results_dir = os.path.join(main_dir, "results")
results_file_path = os.path.join(results_dir, "correctness_predicting_units.pkl")

# Ensure the results directory exists
os.makedirs(results_dir, exist_ok=True)

# Prepare the data by excluding null results
filtered_results = {}
for representation, regression_results in results_per_representation.items():
    filtered_results[representation] = {}
    for layer_idx, layer_results in regression_results.items():
        # Copy only the non-null results
        filtered_results[representation][layer_idx] = {
            "effect_sizes": layer_results["effect_sizes"],
            "errors": layer_results["errors"],
            "pvalues": layer_results["pvalues"],
            "corrected_pvalues": layer_results["corrected_pvalues"],
        }

# Save the filtered results as a pickle file
with open(results_file_path, "wb") as f:
    pickle.dump(filtered_results, f)

print(f"Results saved to {results_file_path}")


# Loop through each representation
for representation, regression_results in results_per_representation.items():
    print(f"Processing plots for representation: {representation}")
    
    positive_significant_counts = []
    negative_significant_counts = []
    mean_abs_effect_size_positive = []
    mean_abs_effect_size_negative = []

    for layer_idx, layer_results in regression_results.items():
        # Extract data for this layer
        effect_sizes = layer_results["effect_sizes"]
        corrected_pvalues = layer_results["corrected_pvalues"]

        # Identify significant parameters
        significant_mask = corrected_pvalues < 0.05
        positive_significant_mask = significant_mask & (effect_sizes > 0)
        negative_significant_mask = significant_mask & (effect_sizes < 0)

        # Count significant parameters
        positive_significant_counts.append(np.sum(positive_significant_mask))
        negative_significant_counts.append(np.sum(negative_significant_mask))

        # Compute mean absolute effect sizes
        if np.any(positive_significant_mask):
            mean_abs_effect_size_positive.append(np.mean(np.abs(effect_sizes[positive_significant_mask])))
        else:
            mean_abs_effect_size_positive.append(0)

        if np.any(negative_significant_mask):
            mean_abs_effect_size_negative.append(np.mean(np.abs(effect_sizes[negative_significant_mask])))
        else:
            mean_abs_effect_size_negative.append(0)

    # Plot 1: Number of Significant Parameters vs Layer Number
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(positive_significant_counts) + 1),
        positive_significant_counts,
        color="green",
        label="Positive Significant",
    )
    plt.plot(
        range(1, len(negative_significant_counts) + 1),
        negative_significant_counts,
        color="red",
        label="Negative Significant",
    )
    plt.title(f"Number of Significant Parameters vs Layer Number ({representation})")
    plt.xlabel("Layer Number")
    plt.ylabel("Significant Parameter Count")
    plt.legend()
    pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()

    # Plot 2: Mean Absolute Effect Size vs Layer Number
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(mean_abs_effect_size_positive) + 1),
        mean_abs_effect_size_positive,
        color="green",
        label="Positive Significant",
    )
    plt.plot(
        range(1, len(mean_abs_effect_size_negative) + 1),
        mean_abs_effect_size_negative,
        color="red",
        label="Negative Significant",
    )
    plt.title(f"Mean Absolute Effect Size of Significant Parameters vs Layer Number ({representation})")
    plt.xlabel("Layer Number")
    plt.ylabel("Mean Absolute Effect Size")
    plt.legend()
    pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()


# Define the order and colors for the representations
ordered_representations = ["column_desc", "row_desc", "word_grid", "visual", "chess", "json"]
color_mapping = {
    "column_desc": "blue",
    "row_desc": "blue",
    "word_grid": "red",
    "visual": "red",
    "chess": "green",
    "json": "green",
}

import matplotlib.pyplot as plt
import numpy as np

# Initialize storage for overall data
significant_counts_all = {}  # To store counts of significant parameters
mean_abs_effect_sizes_all = {}  # To store mean absolute effect sizes

# Process each representation
for representation, regression_results in results_per_representation.items():
    print(f"Processing representation: {representation}")
    
    significant_counts = []
    mean_abs_effect_sizes = []

    for layer_idx, layer_results in regression_results.items():
        # Extract data for this layer
        effect_sizes = layer_results["effect_sizes"]
        corrected_pvalues = layer_results["corrected_pvalues"]

        # Identify significant parameters
        significant_mask = corrected_pvalues < 0.05

        # Count significant parameters
        significant_counts.append(np.sum(significant_mask))

        # Compute mean absolute effect size for significant parameters
        if np.any(significant_mask):
            mean_abs_effect_sizes.append(np.mean(np.abs(effect_sizes[significant_mask])))
        else:
            mean_abs_effect_sizes.append(0)

    # Store results for this representation
    significant_counts_all[representation] = significant_counts
    mean_abs_effect_sizes_all[representation] = mean_abs_effect_sizes

# Plot 1: Number of Significant Parameters vs Layer Number
plt.figure(figsize=(12, 6))
for representation, counts in significant_counts_all.items():
    plt.plot(
        range(1, len(counts) + 1),
        counts,
        label=representation,
        color=color_mapping[representation],
    )
plt.title("Number of Significant Parameters vs Layer Number")
plt.xlabel("Layer Number")
plt.ylabel("Significant Parameter Count")
plt.legend()
plt.tight_layout()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()

# Plot 2: Mean Absolute Effect Size vs Layer Number
plt.figure(figsize=(12, 6))
for representation, mean_effect_sizes in mean_abs_effect_sizes_all.items():
    plt.plot(
        range(1, len(mean_effect_sizes) + 1),
        mean_effect_sizes,
        label=representation,
        color=color_mapping[representation],
    )
plt.title("Mean Absolute Effect Size vs Layer Number")
plt.xlabel("Layer Number")
plt.ylabel("Mean Absolute Effect Size")
plt.legend()
plt.tight_layout()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()


from scipy.stats import wilcoxon

# Initialize storage for significant parameter counts across all representations
common_significant_counts = []
null_common_significant_counts = []

# Determine the number of layers and parameters from one of the representations
layers = len(next(iter(results_per_representation.values())))
params = len(next(iter(results_per_representation.values()))[0]["corrected_pvalues"])
spatial_representation_units = np.zeros((layers, params), dtype=bool)  # Boolean mask

# Iterate through each layer
for layer_idx in range(layers):
    # Get the significant parameter indices for all representations in this layer
    significant_indices_all_reps = []
    null_significant_indices_all_reps = []

    for representation, regression_results in results_per_representation.items():
        # Extract significant indices for this representation and layer
        layer_results = regression_results[layer_idx]
        corrected_pvalues = layer_results["corrected_pvalues"]
        significant_mask = corrected_pvalues < 0.05

        # Actual significant indices
        significant_indices_all_reps.append(set(np.where(significant_mask)[0]))

        # Null (shuffled) significant indices
        shuffled_mask = np.random.permutation(significant_mask)
        null_significant_indices_all_reps.append(set(np.where(shuffled_mask)[0]))

    # Find parameters that are significant across all representations (actual)
    common_significant_indices = set.intersection(*significant_indices_all_reps)
    common_significant_counts.append(len(common_significant_indices))

    # Mark these parameters in the spatial_representation_units mask
    for idx in common_significant_indices:
        spatial_representation_units[layer_idx, idx] = True

    # Find parameters that are significant across all representations (null)
    null_common_significant_indices = set.intersection(*null_significant_indices_all_reps)
    null_common_significant_counts.append(len(null_common_significant_indices))

stat, p_value = wilcoxon(common_significant_counts[1:], null_common_significant_counts[1:])
print("Wilcoxon test p-value:", stat, p_value)

# Plot: Number of Common Significant Parameters vs Layer Number
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(common_significant_counts) + 1),
    common_significant_counts,
    color="black",
    label="Actual Common Significant Parameters",
)
plt.plot(
    range(1, len(null_common_significant_counts) + 1),
    null_common_significant_counts,
    color="gray",
    label="Null (Shuffled) Common Significant Parameters",
)
plt.title("Number of Common Significant Parameters vs Layer Number")
plt.xlabel("Layer Number")
plt.ylabel("Significant Parameter Count")
plt.legend()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()

# Save the spatial_representation_units array
results_dir = os.path.join(main_dir, "results")
os.makedirs(results_dir, exist_ok=True)
spatial_units_file = os.path.join(results_dir, "spatial_representation_units.pkl")

with open(spatial_units_file, "wb") as f:
    pickle.dump(spatial_representation_units, f)

print(f"Spatial representation units saved to {spatial_units_file}")



# Initialize storage for significant parameter counts across all representations
common_significant_counts = []
null_common_significant_counts = []

# Determine the number of layers and parameters from one of the representations
layers = len(next(iter(results_per_representation.values())))
params = len(next(iter(results_per_representation.values()))[0]["corrected_pvalues"])
spatial_representation_units_lax = np.zeros((layers, params), dtype=bool)  # Boolean mask

# Iterate through each layer
for layer_idx in range(layers):
    # Get the significant parameter indices for all representations in this layer
    significant_indices_all_reps = []
    null_significant_indices_all_reps = []

    # Count the total number of representations
    num_representations = len(results_per_representation)

    for representation, regression_results in results_per_representation.items():
        # Extract significant indices for this representation and layer
        layer_results = regression_results[layer_idx]
        corrected_pvalues = layer_results["corrected_pvalues"]
        significant_mask = corrected_pvalues < 0.05

        # Actual significant indices
        significant_indices_all_reps.append(set(np.where(significant_mask)[0]))

        # Null (shuffled) significant indices
        shuffled_mask = np.random.permutation(significant_mask)
        null_significant_indices_all_reps.append(set(np.where(shuffled_mask)[0]))

    # Find parameters that are significant in more than half of the representations (actual)
    threshold = num_representations - 2 # More than half the representations
    significant_count_per_param = np.zeros(params, dtype=int)

    for sig_indices in significant_indices_all_reps:
        for idx in sig_indices:
            significant_count_per_param[idx] += 1

    # Find the parameters that are significant in more than half of the representations
    common_significant_indices = np.where(significant_count_per_param > threshold)[0]
    common_significant_counts.append(len(common_significant_indices))

    # Mark these parameters in the spatial_representation_units mask
    for idx in common_significant_indices:
        spatial_representation_units_lax[layer_idx, idx] = True

    # Find parameters that are significant across more than half of the representations (null)
    null_significant_count_per_param = np.zeros(params, dtype=int)

    for null_sig_indices in null_significant_indices_all_reps:
        for idx in null_sig_indices:
            null_significant_count_per_param[idx] += 1

    # Null significant parameters (shuffled)
    null_common_significant_indices = np.where(null_significant_count_per_param > threshold)[0]
    null_common_significant_counts.append(len(null_common_significant_indices))

stat, p_value = wilcoxon(common_significant_counts[1:], null_common_significant_counts[1:])
print("Wilcoxon test p-value:", stat, p_value)

# Plot: Number of Common Significant Parameters vs Layer Number
plt.figure(figsize=(10, 6))
plt.plot(
    range(1, len(common_significant_counts) + 1),
    common_significant_counts,
    color="black",
    label="Actual Common Significant Parameters",
)
plt.plot(
    range(1, len(null_common_significant_counts) + 1),
    null_common_significant_counts,
    color="gray",
    label="Null (Shuffled) Common Significant Parameters",
)
plt.title("Number of Common Significant Parameters vs Layer Number")
plt.xlabel("Layer Number")
plt.ylabel("Significant Parameter Count")
plt.legend()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()

# Save the spatial_representation_units array
results_dir = os.path.join(main_dir, "results")
os.makedirs(results_dir, exist_ok=True)
spatial_units_file = os.path.join(results_dir, "spatial_representation_units_lax.pkl")

with open(spatial_units_file, "wb") as f:
    pickle.dump(spatial_representation_units_lax, f)

print(f"Spatial representation units saved to {spatial_units_file}")