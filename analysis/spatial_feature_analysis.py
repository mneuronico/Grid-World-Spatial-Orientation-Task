import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

# -------------------------
# 1. Set Up Paths & Load Dataset
# -------------------------
main_dir = r"N:\XAI - LLM GridWorld\Experiment 2"
dataset_path = os.path.join(main_dir, "dataset.csv")
paths_dir = os.path.join(main_dir, "paths")
mind_histories_dir = os.path.join(main_dir, "mind_histories")

activations_origin = "input"
save_to_pdf = True

# Directory to store processed data (create if it doesn't exist)
processed_dir = os.path.join(main_dir, f"processed_data_{activations_origin}")
os.makedirs(processed_dir, exist_ok=True)

# Load the main dataset and filter out the random policy trials (model_size == 0)
df_dataset = pd.read_csv(dataset_path)
df_dataset = df_dataset[df_dataset["model_size"] != 0]
print(f"Total trials (non-control): {len(df_dataset)}")


# DON'T RUN AGAIN UNLESS YOU ABSOLUTELY HAVE TO

# We will accumulate:
# 1. A list of metadata (one per step) with the agent's x, y, decision, trial id, goal info, etc.
# 2. A list of activation arrays (one per step), each of shape (layers, params)
meta_data = []
activations_list = []

# Process each trial one at a time
for idx, trial_info in df_dataset.iterrows():
    trial_id = trial_info["id"]
    print("Processing trial", trial_id)
    
    # Load trial CSV (which has per-step information like agent_x, agent_y, and chosen action)
    trial_csv_path = os.path.join(paths_dir, f"{trial_id}.csv")
    try:
        df_trial = pd.read_csv(trial_csv_path)
    except Exception as e:
        print(f"Error reading {trial_csv_path}: {e}")
        continue
    
    # Load corresponding mind history pickle (each step should have a decision with activations)
    mind_history_path = os.path.join(mind_histories_dir, f"{trial_id}.pkl")
    try:
        with open(mind_history_path, "rb") as f:
            mind_history = pickle.load(f)
    except Exception as e:
        print(f"Error reading {mind_history_path}: {e}")
        continue

    # Get additional trial-level meta-data (goal positions, representation, etc.)
    goal_x = trial_info["goal_x"]
    goal_y = trial_info["goal_y"]
    representation = trial_info["representation"]
    
    # Process only the steps available in both the CSV and the mind history.
    n_steps = min(len(df_trial), len(mind_history))
    for i in range(n_steps):
        # Get step info from the CSV
        step_info = df_trial.iloc[i]
        agent_x = step_info["agent_x"]
        agent_y = step_info["agent_y"]
        decision = step_info["action"]

        # Get the activation for this step.
        # We assume each mind_history entry is a dict with key "decision" containing an "activations" array.

        # I can choose between "decision", "input" or "output" {activations_origin}
        activation = mind_history[i][activations_origin]["activations"].mean(axis=1) #this is (layers, params), we average across decision tokens
        activation = np.array(activation)  # ensure it's a NumPy array

        # Append the activation (per step) to our list
        activations_list.append(activation)
        
        # Create a meta-data dictionary for this step.
        meta = {
            "trial_id": trial_id,
            "step_number": i,
            "agent_x": agent_x,
            "agent_y": agent_y,
            "decision": decision,
            "goal_x": goal_x,
            "goal_y": goal_y,
            "representation": representation
        }
        meta_data.append(meta)

# At this point:
#   - activations_list is a list of arrays, each with shape (layers, params)
#   - meta_data is a list of dictionaries, one per step
print("Total steps processed:", len(activations_list))

# Convert the activations_list to a NumPy array.
# This will have shape (steps, layers, params)
act_array = np.stack(activations_list, axis=0)
print("Combined activations shape (steps, layers, params):", act_array.shape)

# Rearrange to shape (layers, params, steps)
act_array = np.transpose(act_array, (1, 2, 0))
print("Transposed activations shape (layers, params, steps):", act_array.shape)


######################################




# Save the meta-data as a pickle file
meta_file = os.path.join(processed_dir, "meta_data.pkl")
with open(meta_file, "wb") as f:
    pickle.dump(meta_data, f, protocol=pickle.HIGHEST_PROTOCOL)
print("Meta data saved to", meta_file)

# Save each layer's activations separately.
n_layers = act_array.shape[0]
for l in range(n_layers):
    layer_data = act_array[l, :, :]  # shape: (params, steps) for this layer
    layer_file = os.path.join(processed_dir, f"layer_{l}.pkl")
    with open(layer_file, "wb") as f:
        pickle.dump(layer_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Layer {l} data saved to {layer_file}")


act_array = None
meta_data = None
layer_data = None


analysis_dir = os.path.join(main_dir, f"analysis_results_{activations_origin}")
os.makedirs(analysis_dir, exist_ok=True)

alpha = 0.05  # significance level

# ---------------------------
# Load Meta Data
# ---------------------------
meta_file = os.path.join(processed_dir, "meta_data.pkl")
with open(meta_file, "rb") as f:
    meta_data = pickle.load(f)
meta_df = pd.DataFrame(meta_data)
n_steps = len(meta_df)
print("Total steps (meta_data):", n_steps)


# Determine the representation types (plus a general 'all' analysis)
reps = meta_df["representation"].unique().tolist()
analysis_keys = ["all"] + reps

# ---------------------------
# Determine the number of layers and parameters per layer
# ---------------------------
# Load one layer file (layer_0) to get the shape.
layer0_file = os.path.join(processed_dir, "layer_0.pkl")
with open(layer0_file, "rb") as f:
    layer0_data = pickle.load(f)  # expected shape: (n_params, n_steps)
n_params = layer0_data.shape[0]

# Get all layer files
layer_files = sorted([f for f in os.listdir(processed_dir) if f.startswith("layer_") and f.endswith(".pkl")])
n_layers = len(layer_files)
print(f"Detected {n_layers} layers, each with {n_params} parameters.")

# Total comparisons (for Bonferroni correction) is 25 tests per neuron.
total_comparisons = 25 * n_layers * n_params
print("Total comparisons for Bonferroni correction:", total_comparisons)

masks = {}  # dictionary to hold final boolean masks

for analysis_key in analysis_keys:
    print(f"\nProcessing analysis for: {analysis_key}")
    # Get indices for the filtered steps
    if analysis_key == "all":
        indices = np.arange(n_steps)
    else:
        indices = meta_df.index[meta_df["representation"] == analysis_key].to_numpy()
    n_filtered = len(indices)
    print(f"Number of steps for analysis '{analysis_key}': {n_filtered}")
    
    # Create a filtered meta-data DataFrame with continuous indexing
    subset_meta = meta_df.loc[indices].reset_index(drop=True)
    
    # Precompute the binary indicator for each grid cell (cell_x, cell_y)
    cell_indicators = {}
    for cell_x in range(5):
        for cell_y in range(5):
            indicator = (((subset_meta["agent_x"] == cell_x) & (subset_meta["agent_y"] == cell_y))
                         .astype(int).to_numpy())
            cell_indicators[(cell_x, cell_y)] = indicator  # shape: (n_filtered,)
    
    # Allocate mask array: shape (5, 5, n_layers, n_params)
    mask = np.zeros((5, 5, n_layers, n_params), dtype=bool)
    
    # Loop over layers (load each layer's activation data only once)
    for l in range(n_layers):
        print(f"working on layer {l}")
        
        layer_file = os.path.join(processed_dir, f"layer_{l}.pkl")
        with open(layer_file, "rb") as f:
            layer_data = pickle.load(f)  # shape: (n_params, n_steps)
        # Subset activations to the filtered steps; result shape: (n_params, n_filtered)
        layer_subset = layer_data[:, indices]
        
        # Loop over each grid cell
        for cell_x in range(5):
            for cell_y in range(5):
                # Get the indicator vector for this cell
                x_indicator = cell_indicators[(cell_x, cell_y)]  # shape: (n_filtered,)
                n1 = np.sum(x_indicator)
                n0 = n_filtered - n1
                # If too few samples in one group, set p-values to 1 for all neurons.
                if n1 < 2 or n0 < 2:
                    p_vals = np.ones(n_params)
                else:
                    # Split the activations by group; y has shape (n_params, n_filtered)
                    y = layer_subset
                    # Group 1: where indicator is 1, shape (n_params, n1)
                    y1 = y[:, x_indicator == 1]
                    # Group 0: where indicator is 0, shape (n_params, n0)
                    y0 = y[:, x_indicator == 0]
                    
                    # Compute means for each neuron
                    mean1 = np.mean(y1, axis=1)
                    mean0 = np.mean(y0, axis=1)
                    diff = mean1 - mean0
                    
                    # Compute sample variances with degrees of freedom = 1
                    var1 = np.var(y1, axis=1, ddof=1)
                    var0 = np.var(y0, axis=1, ddof=1)
                    
                    # Pooled variance estimate
                    s2 = ((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)
                    # Standard error for the difference in means
                    se = np.sqrt(s2 * (1 / n1 + 1 / n0))
                    # Compute t-statistics for each neuron
                    t_stat = diff / se
                    df = n1 + n0 - 2  # degrees of freedom
                    # Two-sided p-values
                    p_vals = 2 * (1 - st.t.cdf(np.abs(t_stat), df))
                
                # Apply Bonferroni correction (vectorized)
                corrected_p = p_vals * total_comparisons
                # Update mask for this cell and layer: True if corrected p-value < alpha
                mask[cell_x, cell_y, l, :] = corrected_p < alpha
    
    # Print total count of significant tests for this analysis
    n_positive = np.sum(mask)
    print(f"Analysis '{analysis_key}': {n_positive} significant tests (after correction).")
    masks[analysis_key] = mask
    
    # Save the mask to a pickle file
    mask_file = os.path.join(analysis_dir, f"mask_{analysis_key}.pkl")
    with open(mask_file, "wb") as f:
        pickle.dump(mask, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved mask for analysis '{analysis_key}' to: {mask_file}")

print("\nAll analyses complete.")


from matplotlib.backends.backend_pdf import PdfPages

# Use the pgf backend for matplotlib
plt.rcdefaults()

# Ensure that fonts are embedded in the PDF
plt.rcParams['pdf.fonttype'] = 42  # Output Type 3 (Type3) fonts
plt.rcParams['ps.fonttype'] = 42  # Output Type 3 (Type3) fonts

if save_to_pdf:
    # Dictionary to hold the PdfPages objects for each region
    pdf_file = PdfPages(f"Plots place cells {activations_origin} v2.pdf")


# ---------------------------
# Settings and Directories
# ---------------------------
analysis_dir = os.path.join(main_dir, f"analysis_results_{activations_origin}")

# ---------------------------
# 1. Load Masks for Representations (skip the "all" case)
# ---------------------------
# We assume that the masks were saved with filenames like "mask_<rep>.pkl"
mask_files = {}
for filename in os.listdir(analysis_dir):
    if filename.startswith("mask_") and filename.endswith(".pkl"):
        key = filename[len("mask_"):-len(".pkl")]
        if key != "all":  # Skip the "all" analysis if you don't need it here.
            mask_files[key] = os.path.join(analysis_dir, filename)

# Load each mask into a dictionary: key -> mask (boolean array of shape (5,5,n_layers,n_params))
masks_rep = {}
for key, filepath in mask_files.items():
    with open(filepath, "rb") as f:
        masks_rep[key] = pickle.load(f)
    print(f"Loaded mask for representation '{key}' with shape {masks_rep[key].shape}")


# ---------------------------
# 2. Plot Number of Significant Units per Layer for Each Representation
# ---------------------------
# For each mask, we count for each layer the number of neurons that are significant in at least one grid cell.
sig_units_per_layer = {}
for key, mask in masks_rep.items():
    # Collapse the grid cells by taking a logical OR over the first two dimensions.
    # This produces an array of shape (n_layers, n_params) indicating whether that neuron was
    # significant in any cell.
    any_signif = np.any(mask, axis=(0, 1))
    # Count the number of neurons per layer that are significant.
    count_per_layer = np.sum(any_signif, axis=1)  # shape: (n_layers,)
    sig_units_per_layer[key] = count_per_layer

# Plot these curves (x-axis: layer, y-axis: count of significant units)
plt.figure(figsize=(10, 6))
for key, counts in sig_units_per_layer.items():
    # Create layers starting from 1 instead of 0
    layers = np.arange(1, len(counts))
    # Slice the counts to exclude the first layer
    plt.plot(layers, counts[1:], label=key)
plt.xlabel("Layer")
plt.ylabel("Number of significant units")
plt.title("Significant units per layer (per representation) excluding first layer")
plt.legend()
plt.tight_layout()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()


# ---------------------------
# 3. Compute and Plot Common Significant Units Across Representations
# ---------------------------
# Compute the common mask (logical AND across all representation masks)
masks_list = list(masks_rep.values())
common_mask = np.logical_and.reduce(masks_list)  # shape: (5,5,n_layers,n_params)

# For each layer, count the number of neurons that are significant in at least one cell in the common mask.
common_units_per_layer = np.sum(np.any(common_mask, axis=(0, 1)), axis=1)  # shape: (n_layers,)
plt.figure(figsize=(10, 6))
layers = np.arange(1, len(common_units_per_layer))
plt.plot(layers, common_units_per_layer[1:], color="black")
plt.xlabel("Layer")
plt.ylabel("Number of common significant units")
plt.title("Common significant units per layer (across all representations, excluding first layer)")
plt.tight_layout()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()


# ---------------------------
# Border Analysis: For each analysis key, perform regression (via a vectorized t-test)
# ---------------------------
masks_border = {}  # Will hold, for each analysis key, a mask of shape (n_layers, n_params)

for analysis_key in analysis_keys:
    print(f"\nProcessing border analysis for: {analysis_key}")
    # Select steps for this analysis
    if analysis_key == "all":
        indices = np.arange(n_steps)
    else:
        indices = meta_df.index[meta_df["representation"] == analysis_key].to_numpy()
    n_filtered = len(indices)
    print(f"Number of steps for analysis '{analysis_key}': {n_filtered}")
    
    # Create a filtered meta-data DataFrame (reset index for convenience)
    subset_meta = meta_df.loc[indices].reset_index(drop=True)
    
    # Create border indicator: 1 if agent_x or agent_y is at 0 or 4 (i.e. on the border), else 0.
    border_indicator = (
        ((subset_meta["agent_x"] == 0) | (subset_meta["agent_x"] == 4)) |
        ((subset_meta["agent_y"] == 0) | (subset_meta["agent_y"] == 4))
    ).astype(int).to_numpy()  # shape: (n_filtered,)
    
    # Allocate mask for this analysis (for each layer and neuron)
    mask = np.zeros((n_layers, n_params), dtype=bool)
    
    # Loop over layers; for each, load the activations and perform the vectorized t-test
    for l in range(n_layers):
        print("processing layer", l)
        layer_file = os.path.join(processed_dir, f"layer_{l}.pkl")
        with open(layer_file, "rb") as f:
            layer_data = pickle.load(f)  # shape: (n_params, n_steps)
        # Subset activations for the filtered steps; result shape: (n_params, n_filtered)
        layer_subset = layer_data[:, indices]
        
        # Determine group sizes for border (indicator==1) and non-border (indicator==0)
        n1 = np.sum(border_indicator)
        n0 = n_filtered - n1
        # If either group has too few samples, we set p-values to 1 (non-significant)
        if n1 < 2 or n0 < 2:
            p_vals = np.ones(n_params)
        else:
            # Vectorized computation across all neurons:
            # y has shape (n_params, n_filtered)
            y = layer_subset
            # Group 1 (border)
            y1 = y[:, border_indicator == 1]  # shape: (n_params, n1)
            # Group 0 (non-border)
            y0 = y[:, border_indicator == 0]  # shape: (n_params, n0)
            
            mean1 = np.mean(y1, axis=1)
            mean0 = np.mean(y0, axis=1)
            diff = mean1 - mean0
            
            var1 = np.var(y1, axis=1, ddof=1)
            var0 = np.var(y0, axis=1, ddof=1)
            # Pooled variance estimate:
            s2 = ((n1 - 1) * var1 + (n0 - 1) * var0) / (n1 + n0 - 2)
            se = np.sqrt(s2 * (1 / n1 + 1 / n0))
            
            t_stat = diff / se
            df = n1 + n0 - 2
            p_vals = 2 * (1 - st.t.cdf(np.abs(t_stat), df))
        
        # Apply Bonferroni correction: multiply p-values by total_comparisons
        corrected_p = p_vals * total_comparisons
        mask[l, :] = corrected_p < alpha
        
    # Report and save results for this analysis
    sig_count = np.sum(mask, axis=1)
    total_sig = np.sum(mask)
    print(f"Analysis '{analysis_key}': Significant neurons per layer: {sig_count}, Total significant: {total_sig}")
    
    mask_file = os.path.join(analysis_border_dir, f"border_mask_{analysis_key}.pkl")
    with open(mask_file, "wb") as f:
        pickle.dump(mask, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved border mask for analysis '{analysis_key}' to {mask_file}")
    
    masks_border[analysis_key] = mask


    # ---------------------------
# Plotting: Number of Border-Coding Significant Neurons per Layer
# ---------------------------
plt.figure(figsize=(10, 6))
for key, mask in masks_border.items():
    if key == "all":
        continue
    # mask shape is (n_layers, n_params)
    sig_count = np.sum(mask, axis=1)
    layers = np.arange(1, n_layers)
    plt.plot(layers, sig_count[1:], label=key)
plt.xlabel("Layer")
plt.ylabel("Number of significant neurons")
plt.title("Border-Coding Significant Neurons per Layer")
plt.legend()
plt.tight_layout()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()

# ---------------------------
# Common Analysis Across Representations with Shuffled Baseline
# ---------------------------
# Exclude the "all" case.
rep_keys = [k for k in masks_border.keys() if k != "all"]
if rep_keys:
    # Stack the masks into an array of shape (n_reps, n_layers, n_params)
    stacked = np.stack([masks_border[k] for k in rep_keys], axis=0)
    # Define the common threshold (e.g., threshold = 6 means a unit must be positive in all reps)
    threshold = 6  # Adjust if the number of reps differs.
    common_border_mask = np.sum(stacked, axis=0) >= threshold  # shape: (n_layers, n_params)
    # Count common significant units per layer (real data)
    common_units_per_layer = np.sum(common_border_mask, axis=1)
    
    # Now compute a "shuffled" common mask:
    # For each representation, shuffle each layer's mask (preserving the number of positives).
    shuffled_masks_list = []
    for rep in rep_keys:
        orig_mask = masks_border[rep]  # shape: (n_layers, n_params)
        shuffled_mask = np.empty_like(orig_mask)
        for l in range(n_layers):
            # Shuffle the l-th row while preserving the number of True values.
            shuffled_mask[l, :] = np.random.permutation(orig_mask[l, :])
        shuffled_masks_list.append(shuffled_mask)
    # Stack the shuffled masks.
    shuffled_stacked = np.stack(shuffled_masks_list, axis=0)
    common_border_mask_shuffled = np.sum(shuffled_stacked, axis=0) >= threshold  # shape: (n_layers, n_params)
    common_units_per_layer_shuffled = np.sum(common_border_mask_shuffled, axis=1)

    print("number of common units:", np.sum(common_border_mask))
    print("number of common units:", np.sum(common_border_mask_shuffled))

    stat, p_value = wilcoxon(common_units_per_layer[1:], common_units_per_layer_shuffled[1:])
    print("Wilcoxon test p-value:", stat, p_value)
    
    # Plot the common curve with the shuffled baseline.
    plt.figure(figsize=(10, 6))
    layers = np.arange(1, n_layers)
    plt.plot(layers, common_units_per_layer[1:], color="black", label="Common Significant")
    plt.plot(layers, common_units_per_layer_shuffled[1:], linestyle="--", color="gray", label="Shuffled Common")
    plt.xlabel("Layer")
    plt.ylabel("Number of common significant neurons")
    plt.title("Common Border-Coding Significant Neurons per Layer")
    plt.legend()
    plt.tight_layout()
    pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()


# ---------------------------
# For each representation and each test, compute correlations and form a significance mask.
# We'll store the results in a nested dictionary:
#    results_masks[test_name][representation] = mask (shape: (n_layers, n_params))
# ---------------------------
results_masks = { test: {} for test in test_names }

for rep in representations:
    print(f"\nProcessing representation: {rep}")
    # Get the indices of steps for this representation.
    rep_indices = meta_df.index[meta_df["representation"] == rep].to_numpy()
    n_filtered = len(rep_indices)
    print(f"Number of steps for representation '{rep}': {n_filtered}")
    
    # Subset meta-data for this representation.
    rep_meta = meta_df.loc[rep_indices].reset_index(drop=True)
    
    # For each test type:
    for test_name, func in test_types.items():
        print(test_name)
        # Compute the scalar vector for the test (shape: (n_filtered,))
        X = func(rep_meta)
        if n_filtered < 3:
            print(f"  Not enough data for test {test_name} in representation {rep}. Skipping.")
            mask_rep = np.zeros((n_layers, n_params), dtype=bool)
            results_masks[test_name][rep] = mask_rep
            continue

        # Pre-compute statistics for X.
        X_mean = np.mean(X)
        X_centered = X - X_mean
        denomX = np.sqrt(np.sum(X_centered ** 2))
        
        # Allocate an array to store the mask for this representation and test.
        mask_rep = np.zeros((n_layers, n_params), dtype=bool)
        
        # Process each layer.
        for l in range(n_layers):
            print("processing layer", l)
            layer_file = os.path.join(processed_dir, f"layer_{l}.pkl")
            with open(layer_file, "rb") as f:
                layer_data = pickle.load(f)  # shape: (n_params, n_steps)
            # Subset activation data to the steps for this representation.
            # Resulting shape: (n_params, n_filtered)
            Y = layer_data[:, rep_indices]
            
            # Compute the mean and center Y along each neuron (row).
            Y_mean = np.mean(Y, axis=1, keepdims=True)
            Y_centered = Y - Y_mean  # shape: (n_params, n_filtered)
            
            # Compute numerator for Pearson correlation for each neuron.
            # Multiply each row of Y_centered with X_centered and sum over steps.
            numerator = np.sum(Y_centered * X_centered, axis=1)  # shape: (n_params,)
            
            # Compute denominator for each neuron.
            denomY = np.sqrt(np.sum(Y_centered ** 2, axis=1))  # shape: (n_params,)
            denom = denomX * denomY  # shape: (n_params,)
            
            # Avoid division by zero.
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(denom == 0, 0, numerator / denom)
            # Clip r to avoid numerical issues (avoid exactly 1 or -1).
            r = np.clip(r, -0.9999, 0.9999)
            
            # Degrees of freedom for Pearson correlation.
            df_val = n_filtered - 2
            # Compute t-statistic from r: t = r * sqrt(df/(1-r^2))
            t_stat = r * np.sqrt(df_val / (1 - r**2))
            # Compute two-sided p-value.
            p_vals = 2 * (1 - st.t.cdf(np.abs(t_stat), df=df_val))
            
            # Apply Bonferroni correction.
            corrected_p = np.minimum(p_vals * bonferroni_factor, 1.0)
            # Flag neurons as significant if corrected p-value < alpha.
            mask_layer = corrected_p < alpha  # shape: (n_params,)
            mask_rep[l, :] = mask_layer
        
        # Save the mask for this representation and test.
        results_masks[test_name][rep] = mask_rep
        out_filename = os.path.join(analysis_corr_dir, f"mask_{test_name}_{rep}.pkl")
        with open(out_filename, "wb") as f:
            pickle.dump(mask_rep, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved mask for test '{test_name}' in representation '{rep}' to {out_filename}")


from scipy.stats import wilcoxon

# ---------------------------
# Plotting: For each test type, plot number of significant parameters per layer for each representation.
# ---------------------------
for test_name in test_names:
    plt.figure(figsize=(10, 6))
    for rep in representations:
        mask_rep = results_masks[test_name][rep]  # shape (n_layers, n_params)
        # Count significant neurons per layer.
        sig_counts = np.sum(mask_rep, axis=1)
        layers = np.arange(1, n_layers)
        plt.plot(layers, sig_counts[1:], label=rep)
    plt.xlabel("Layer")
    plt.ylabel("Number of significant parameters")
    plt.title(f"Significant parameters per layer for test: {test_name}")
    plt.legend()
    plt.tight_layout()
    pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()

# ---------------------------
# Compute and Plot Common Significant Masks Across Representations for Each Test Type
# For each test type, the common mask is defined as the logical AND across all representations.
# ---------------------------
common_masks = {}  # common_masks[test_name] = mask of shape (n_layers, n_params)

for test_name in test_names:
    # Gather the masks for all representations for this test.
    masks_list = [results_masks[test_name][rep] for rep in representations]
    # Compute the "real" common mask as the logical AND across representations.
    common_mask = np.logical_and.reduce(masks_list)
    common_masks[test_name] = common_mask
    # Count common significant parameters per layer.
    common_counts = np.sum(common_mask, axis=1)  # shape: (n_layers,)
    
    # Now, compute a "shuffled" common mask.
    # For each representation, shuffle each layer's mask (preserving the number of True values).
    shuffled_masks_list = []
    for rep in representations:
        orig_mask = results_masks[test_name][rep]  # shape: (n_layers, n_params)
        shuffled_mask = np.empty_like(orig_mask)
        for l in range(n_layers):
            # Shuffle the row while keeping the same number of positives.
            shuffled_mask[l, :] = np.random.permutation(orig_mask[l, :])
        shuffled_masks_list.append(shuffled_mask)
    # Compute the common mask from the shuffled masks.
    common_mask_shuffled = np.logical_and.reduce(shuffled_masks_list)
    common_counts_shuffled = np.sum(common_mask_shuffled, axis=1)

    print("number of common units:", np.sum(common_mask))
    print("number of common units:", np.sum(common_mask_shuffled))
    
    # Plot the common curve (real vs. shuffled).
    layers = np.arange(1, n_layers)
    plt.figure(figsize=(10, 6))
    plt.plot(layers, common_counts[1:], color="black", label="Common Significant")
    plt.plot(layers, common_counts_shuffled[1:], linestyle="--", color="gray", label="Shuffled Common")
    plt.xlabel("Layer")
    plt.ylabel("Number of common significant parameters")
    plt.title(f"Common Significant Parameters per Layer ({test_name})")
    plt.legend()
    plt.tight_layout()
    pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()

    stat, p_value = wilcoxon(common_counts[1:], common_counts_shuffled[1:])
    print("Wilcoxon test p-value:", stat, p_value)
    
    # Save the common mask.
    common_mask_file = os.path.join(analysis_corr_dir, f"common_mask_{test_name}.pkl")
    with open(common_mask_file, "wb") as f:
        pickle.dump(common_mask, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved common mask for test '{test_name}' to {common_mask_file}")