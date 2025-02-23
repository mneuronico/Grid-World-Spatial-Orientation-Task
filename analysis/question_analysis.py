import os
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_1samp

save_to_pdf = False

# Set the base directory
BASE_DIR = r"N:\XAI - LLM GridWorld\Experiment 3"

# File/folder paths
MIND_HISTORY_DIR = os.path.join(BASE_DIR, "mind_histories")
SPATIAL_MASK_PATH = os.path.join(BASE_DIR, "xy_predicting_units.pkl") #"spatial_representation_units.pkl"

from matplotlib.backends.backend_pdf import PdfPages

# Use the pgf backend for matplotlib
plt.rcdefaults()

# Ensure that fonts are embedded in the PDF
plt.rcParams['pdf.fonttype'] = 42  # Output Type 3 (Type3) fonts
plt.rcParams['ps.fonttype'] = 42  # Output Type 3 (Type3) fonts

if save_to_pdf:
    # Dictionary to hold the PdfPages objects for each region
    pdf_file = PdfPages("Plots question probinbg v2.pdf")


#############################
# 1. Load the Spatial Mask
#############################
print("Loading spatial representation mask...")
with open(SPATIAL_MASK_PATH, "rb") as f:
    spatial_mask = pickle.load(f)  # Boolean NumPy array, shape: (layers, params)
    
# Determine the total number of spatial rep units (number of True values)
n_rep_units = np.sum(spatial_mask)
print(f"Spatial mask loaded. Shape: {spatial_mask.shape}, total rep units: {n_rep_units}")


#####################################
# 2. Load Mind History Files & Build X, Y
#####################################
print("Loading mind history files...")
# Get a list of all pickle files in the mind_histories directory
mind_history_files = glob.glob(os.path.join(MIND_HISTORY_DIR, "*.pkl"))
print(f"Found {len(mind_history_files)} mind history files.")

X_list = []  # to store activation vectors for each example
Y_list = []  # to store spatial label (0 or 1) for each example

for filepath in mind_history_files:
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        continue

    # Expected dictionary structure:
    # {
    #   "input": {"text": prompt, "activations": prompt_hidden_states},
    #   "output": {"text": response, "activations": response_hidden_states},
    #   "spatial": spatial   (0 or 1)
    # }
    try:
        prompt_hidden_states = data["input"]["activations"]  # shape: (layers, prompt_tokens, params)
        response_hidden_states = data["output"]["activations"]  # shape: (layers, response_tokens, params)
        label = data["spatial"]
    except KeyError as ke:
        print(f"Key error in file {filepath}: {ke}")
        continue

    # Average across tokens for prompt and response separately:
    # (resulting shapes: (layers, params))
    avg_prompt = np.mean(prompt_hidden_states, axis=1)
    avg_response = np.mean(response_hidden_states, axis=1)
    # Average across the two arrays to get one activation per layer
    avg_activation = avg_prompt#prompt_hidden_states[:,-1,:]#(avg_prompt + avg_response) / 2  # shape: (layers, params)

    # Apply the spatial mask to select only the representation units.
    # This works because both avg_activation and spatial_mask have shape (layers, params)
    rep_activation = avg_activation[spatial_mask]  # 1D array of shape (n_rep_units,)

    X_list.append(rep_activation)
    Y_list.append(label)

# Convert lists to NumPy arrays
X = np.vstack(X_list)  # shape: (num_examples, rep_units)
Y = np.array(Y_list)   # shape: (num_examples,)
n_examples = X.shape[0]
print(f"Built dataset: X shape = {X.shape}, Y shape = {Y.shape}")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Make sure Seaborn is installed
from scipy.stats import pearsonr, ttest_1samp

#####################################
# 3. Correlation Analysis (Original Mask)
#####################################
print("Computing correlation for each rep unit (original mask)...")
n_units = X.shape[1]
corr_coeffs = np.zeros(n_units)
p_values = np.zeros(n_units)

for j in range(n_units):
    # Compute Pearson correlation between activations in rep unit j and the spatial label
    r, p = pearsonr(X[:, j], Y)
    corr_coeffs[j] = r
    # Bonferroni correction: multiply p-value by number of rep units, cap at 1.
    p_values[j] = min(p * n_units, 1.0)

# One-sample t-test: are the correlation coefficients significantly different from 0?
t_stat, t_p = ttest_1samp(corr_coeffs, 0)
print("One-sample t-test for correlation coefficients (original mask):")
print(f"  t-statistic = {t_stat:.4f}, p-value = {t_p:.4f}")

from scipy.stats import wilcoxon

# One-sample Wilcoxon signed-rank test: are the correlation coefficients significantly different from 0?
w_stat, w_p = wilcoxon(corr_coeffs)
print("Wilcoxon signed-rank test for correlation coefficients (original mask):")
print(f"  W-statistic = {w_stat:.4f}, p-value = {w_p:.4f}")

# Plotting
plt.figure(figsize=(12, 5))

# Density (KDE) plot for correlation coefficients with vertical lines at 0 and the mean
plt.subplot(1, 2, 1)
sns.kdeplot(corr_coeffs, shade=True, bw_adjust=1.5)
mean_corr = np.mean(corr_coeffs)
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Density")
plt.xlim(-1.6,1.6)
plt.ylim(0,0.8)
plt.title("Density Plot of Rep Unit Correlation Coefficients\n(Original Mask)")
plt.legend()

# Histogram for Bonferroni corrected p-values with a vertical line at 0.05
plt.subplot(1, 2, 2)
plt.hist(p_values, bins=30, edgecolor="k")
plt.axvline(0.05, color="red", linestyle="--", label="p = 0.05")
plt.xlabel("Bonferroni Corrected p-value")
plt.ylabel("Frequency")
plt.title("Histogram of Corrected p-values\n(Original Mask)")
plt.legend()

plt.tight_layout()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()


#####################################
# 4. Null Analysis: Shuffled Mask
#####################################
np.random.seed(42)

print("Performing null analysis by shuffling the spatial mask (within each layer)...")
# Create a null (shuffled) mask: For each layer, shuffle the boolean values.
null_mask = np.empty_like(spatial_mask)
n_layers, n_params = spatial_mask.shape
for layer in range(n_layers):
    # np.random.permutation shuffles the entries of the boolean array.
    null_mask[layer] = np.random.permutation(spatial_mask[layer])
    
# (Optional) Check that the number of True values per layer is preserved
if not np.all([np.sum(null_mask[i]) == np.sum(spatial_mask[i]) for i in range(n_layers)]):
    print("Warning: The number of True values was not preserved in the shuffling!")

# For each example, compute the null activation vector using the shuffled mask.
X_null_list = []
for filepath in mind_history_files:
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        continue

    try:
        prompt_hidden_states = data["input"]["activations"]  # shape: (layers, prompt_tokens, params)
        response_hidden_states = data["output"]["activations"]  # shape: (layers, response_tokens, params)
    except KeyError as ke:
        continue

    avg_prompt = np.mean(prompt_hidden_states, axis=1)    # shape: (layers, params)
    avg_response = np.mean(response_hidden_states, axis=1)  # shape: (layers, params)
    avg_activation = (avg_prompt + avg_response) / 2         # shape: (layers, params)

    # Apply the null (shuffled) mask:
    null_rep_activation = avg_activation[null_mask]  # 1D array, shape: (n_rep_units,)
    X_null_list.append(null_rep_activation)

X_null = np.vstack(X_null_list)  # shape: (num_examples, rep_units)
print(f"Built null dataset: X_null shape = {X_null.shape}")

# Compute correlation for the null model
null_corr_coeffs = np.zeros(n_units)
null_p_values = np.zeros(n_units)
for j in range(n_units):
    r, p = pearsonr(X_null[:, j], Y)
    null_corr_coeffs[j] = r
    null_p_values[j] = min(p * n_units, 1.0)

# One-sample t-test for null correlation coefficients
null_t_stat, null_t_p = ttest_1samp(null_corr_coeffs, 0)
print("One-sample t-test for correlation coefficients (null model):")
print(f"  t-statistic = {null_t_stat:.4f}, p-value = {null_t_p:.4f}")


# One-sample Wilcoxon signed-rank test: are the correlation coefficients significantly different from 0?
w_stat, w_p = wilcoxon(null_corr_coeffs)
print("Wilcoxon signed-rank test for correlation coefficients (null mask):")
print(f"  W-statistic = {w_stat:.4f}, p-value = {w_p:.4f}")

# Plot histograms for null model
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.kdeplot(null_corr_coeffs, shade=True, bw_adjust=1.5)
mean_corr = np.mean(null_corr_coeffs)
plt.axvline(0, color="black", linestyle="--")
plt.xlabel("Correlation Coefficient")
plt.ylabel("Density")
plt.xlim(-1.6,1.6)
plt.ylim(0,0.8)
plt.title("Density Plot of Rep Unit Correlation Coefficients\n(Shuffle)")
plt.legend()

# Histogram for Bonferroni corrected p-values with a vertical line at 0.05
plt.subplot(1, 2, 2)
plt.hist(null_p_values, bins=30, edgecolor="k")
plt.axvline(0.05, color="red", linestyle="--", label="p = 0.05")
plt.xlabel("Bonferroni Corrected p-value")
plt.ylabel("Frequency")
plt.title("Histogram of Corrected p-values\n(Shuffle)")
plt.legend()
plt.tight_layout()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()

print("Analysis complete.")

if save_to_pdf:
    # Close the PdfPages object to finalize the PDF file
    pdf_file.close()