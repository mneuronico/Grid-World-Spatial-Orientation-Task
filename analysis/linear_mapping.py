import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

save_to_pdf = True

# ---------------------------
# Settings and Directories
# ---------------------------
BASE_DIR = r"N:\XAI - LLM GridWorld\Experiment 2"
processed_dir = os.path.join(BASE_DIR, "processed_data_input")  # contains meta_data.pkl and layer_*.pkl
results_dir = os.path.join(BASE_DIR, "linear_prediction_results")
os.makedirs(results_dir, exist_ok=True)


from matplotlib.backends.backend_pdf import PdfPages

# Use the pgf backend for matplotlib
plt.rcdefaults()

# Ensure that fonts are embedded in the PDF
plt.rcParams['pdf.fonttype'] = 42  # Output Type 3 (Type3) fonts
plt.rcParams['ps.fonttype'] = 42  # Output Type 3 (Type3) fonts

if save_to_pdf:
    # Dictionary to hold the PdfPages objects for each region
    pdf_file = PdfPages(f"Plots linear mapping.pdf")


meta_file = os.path.join(processed_dir, "meta_data.pkl")
with open(meta_file, "rb") as f:
    meta_data = pickle.load(f)
meta_df = pd.DataFrame(meta_data)
n_steps_total = len(meta_df)
print("Total steps in meta data:", n_steps_total)


# ---------------------------
# Define Grid and Target Matrix Builder
# ---------------------------
# For each step, we want to create a binary vector of length 50:
# - The first 25 entries: 1 at the agent's cell, 0 elsewhere.
# - The next 25 entries: 1 at the goal's cell, 0 elsewhere.
def build_target_matrix(df):
    n = df.shape[0]
    Y = np.zeros((n, 50), dtype=int)
    for i, row in df.iterrows():
        # Compute cell indices (assuming row-major order: index = x + 5*y)
        agent_idx = int(row["agent_x"] + 5 * row["agent_y"])
        goal_idx = int(row["goal_x"] + 5 * row["goal_y"])
        Y[i, agent_idx] = 1
        Y[i, 25 + goal_idx] = 1
    return Y

# ---------------------------
# Get Representation Types
# ---------------------------
representations = meta_df["representation"].unique().tolist()
print("Representation types:", representations)

# ---------------------------
# Determine number of layers and parameters per layer
# ---------------------------
# Load one layer file to get the number of parameters.
layer0_file = os.path.join(processed_dir, "layer_0.pkl")
with open(layer0_file, "rb") as f:
    layer0_data = pickle.load(f)  # Expected shape: (n_params, total_steps)
n_params = layer0_data.shape[0]
# Find all layer files:
layer_files = sorted([f for f in os.listdir(processed_dir) if f.startswith("layer_") and f.endswith(".pkl")])
n_layers = len(layer_files)
print(f"Detected {n_layers} layers, each with {n_params} parameters.")


from sklearn.metrics import explained_variance_score

# ---------------------------
# Main Analysis: For each representation and each layer, fit a linear model
# ---------------------------
# We'll store the results in a nested dictionary:
# results[representation][layer] = {"train_r2": ..., "test_r2": ..., "n_samples": ...}
results = {rep: {} for rep in representations}

# Set random state for reproducibility
random_state = 42

for rep in representations:
    print("\nProcessing representation:", rep)
    # Get indices of steps for this representation.
    rep_indices = meta_df.index[meta_df["representation"] == rep].to_numpy()
    n_samples = len(rep_indices)
    print(f"  Number of steps: {n_samples}")
    
    # Subset meta-data for this representation.
    rep_meta = meta_df.loc[rep_indices].reset_index(drop=True)
    # Build Y (target matrix) for these steps.
    Y = build_target_matrix(rep_meta)  # shape: (n_samples, 50)
    
    # For each layer, get the activation data and perform linear regression.
    for l in range(n_layers):
        # Load layer l activations. Each layer file is expected to have shape (n_params, total_steps)
        layer_file = os.path.join(processed_dir, f"layer_{l}.pkl")
        with open(layer_file, "rb") as f:
            layer_data = pickle.load(f)
        # Subset to the current representation's steps: shape becomes (n_params, n_samples)
        X_layer = layer_data[:, rep_indices]
        # Transpose X_layer so that each row is a step, each column is a parameter.
        X = X_layer.T  # shape: (n_samples, n_params)
        
        # Split the data into 90% training and 10% testing.
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.1, random_state=random_state
        )
        
        # --- Actual Model ---
        model = LinearRegression()
        model.fit(X_train, Y_train)
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        r2_train = r2_score(Y_train, Y_train_pred)
        r2_test = r2_score(Y_test, Y_test_pred)
        ev_train = explained_variance_score(Y_train, Y_train_pred)
        ev_test = explained_variance_score(Y_test, Y_test_pred)
        
        # --- Shuffled Control ---
        # Shuffle the target values in the training set to break the relationship.
        Y_train_shuffled = np.random.permutation(Y_train)
        model_shuffled = LinearRegression()
        model_shuffled.fit(X_train, Y_train_shuffled)
        Y_test_pred_shuffled = model_shuffled.predict(X_test)
        r2_test_shuffled = r2_score(Y_test, Y_test_pred_shuffled)
        ev_test_shuffled = explained_variance_score(Y_test, Y_test_pred_shuffled)
        
        # Store the results, including the shuffled control.
        results[rep][l] = {
            "model": model,
            "n_samples": n_samples,
            "X_test": X_test,
            "Y_test": Y_test,
            "train_r2": r2_train,
            "test_r2": r2_test,
            "test_r2_shuffled": r2_test_shuffled,
            "train_ev": ev_train,
            "test_ev": ev_test,
            "test_ev_shuffled": ev_test_shuffled
        }
        print(f"  Layer {l}: train R² = {r2_train:.3f}, test R² = {r2_test:.3f}, shuffled test R² = {r2_test_shuffled:.3f}, train EV = {ev_train:.3f}, test EV = {ev_test:.3f}, shuffled test EV = {ev_test_shuffled:.3f}")
        

# Set random state for reproducibility and number of permutations for null distribution.
random_state = 42
np.random.seed(random_state)
n_permutations = 10

# Prepare a dictionary to store results.
results = {rep: {} for rep in representations}

# Total number of comparisons for Bonferroni correction.
total_comparisons = len(representations) * n_layers

for rep in representations:
    print("\nProcessing representation:", rep)
    # Get indices for current representation.
    rep_indices = meta_df.index[meta_df["representation"] == rep].to_numpy()
    n_samples = len(rep_indices)
    print(f"  Number of steps: {n_samples}")
    
    # Subset meta-data and build target matrix Y (n_samples x target_dim)
    rep_meta = meta_df.loc[rep_indices].reset_index(drop=True)
    Y = build_target_matrix(rep_meta)  # for example, shape: (n_samples, 50)
    
    for l in range(n_layers):
        print(f"  Processing layer {l}")
        # Load layer activations: expect shape (n_params, total_steps)
        layer_file = os.path.join(processed_dir, f"layer_{l}.pkl")
        with open(layer_file, "rb") as f:
            layer_data = pickle.load(f)
        
        # Subset activations for current representation and transpose to shape (n_samples, n_params)
        X_layer = layer_data[:, rep_indices]
        X = X_layer.T
        
        # Split the data (90% train, 10% test)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.1, random_state=random_state
        )
        
        # --- Actual Model ---
        model = LinearRegression()
        model.fit(X_train, Y_train)
        Y_test_pred = model.predict(X_test)
        test_metric = r2_score(Y_test, Y_test_pred)  # or use explained_variance_score
        
        # --- Permutation Test ---
        permutation_scores = []
        for i in range(n_permutations):
            print(i)
            
            # Shuffle Y_train to break the relationship.
            Y_train_shuffled = np.random.permutation(Y_train)
            model_perm = LinearRegression()
            model_perm.fit(X_train, Y_train_shuffled)
            Y_test_pred_perm = model_perm.predict(X_test)
            score_perm = r2_score(Y_test, Y_test_pred_perm)
            permutation_scores.append(score_perm)
        
        permutation_scores = np.array(permutation_scores)
        # p-value: proportion of null distribution that meets or exceeds the actual performance.
        p_value = np.mean(permutation_scores >= test_metric)
        # Bonferroni correction: multiply the p-value by total number of tests.
        p_value_adjusted = p_value * total_comparisons
        # Make sure the adjusted p-value does not exceed 1.
        p_value_adjusted = min(p_value_adjusted, 1.0)
        
        significance = p_value_adjusted < 0.05
        
        # Store the results for this representation and layer.
        results[rep][l] = {
            "model": model,
            "n_samples": n_samples,
            "X_test": X_test,
            "Y_test": Y_test,
            "test_r2": test_metric,
            "null_distribution": permutation_scores,
            "raw_p_value": p_value,
            "adjusted_p_value": p_value_adjusted,
            "significant": significance
        }
        
        print(f"    Layer {l}: test R² = {test_metric:.3f}, raw p-value = {p_value:.3f}, "
              f"adjusted p-value = {p_value_adjusted:.3f}, significant? {significance}")

# ---------------------------
# Plotting: For each representation, plot training and test R² curves across layers.
# ---------------------------
for rep in representations:
    train_scores = [results[rep][l]["train_r2"] for l in range(n_layers)]
    test_scores = [results[rep][l]["test_r2"] for l in range(n_layers)]
    layers = np.arange(1, n_layers)
    
    plt.plot(layers, train_scores[1:], label=f"{rep}Train R²")
    #plt.plot(layers, test_scores, marker="s", label=f"{rep}Test R²")
    plt.xlabel("Layer")
    plt.ylabel("R² Score")
    plt.title(f"Linear Prediction Performance for Training Set")
    plt.legend()
    plt.tight_layout()

pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()
plt.show()

for rep in representations:
    train_scores = [results[rep][l]["train_r2"] for l in range(n_layers)]
    test_scores = [results[rep][l]["test_r2"] for l in range(n_layers)]
    layers = np.arange(1, n_layers)
    
    #plt.plot(layers, train_scores, marker="o", label=f"{rep}Train R²")
    plt.plot(layers, test_scores[1:], label=f"{rep}Test R²")
    plt.xlabel("Layer")
    plt.ylabel("R² Score")
    plt.title(f"Linear Prediction Performance for Test Set")
    plt.legend()
    plt.tight_layout()

pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()
plt.show()

from sklearn.metrics import r2_score
import matplotlib.colors as mcolors
import seaborn as sns

# Get the ordered list of representations.
reps_list = representations  # existing order (we'll reorder later)
n_reps = len(reps_list)

# Create a dictionary to hold a cross-prediction matrix per layer.
cross_r2_by_layer = {}

for l in range(n_layers):
    R = np.zeros((n_reps, n_reps))  # rows: model from rep_i, columns: test set from rep_j
    for i, rep_i in enumerate(reps_list):
        model_i = results[rep_i][l]["model"]
        for j, rep_j in enumerate(reps_list):
            X_test_j = results[rep_j][l]["X_test"]
            Y_test_j = results[rep_j][l]["Y_test"]
            Y_pred = model_i.predict(X_test_j)
            r2_val = r2_score(Y_test_j, Y_pred)
            R[i, j] = r2_val
    cross_r2_by_layer[l] = R

# Now average the cross-prediction matrices across layers (excluding layer 0).
R_sum = np.zeros((n_reps, n_reps))
count = 0
for l in range(1, n_layers):  # Exclude first layer.
    R_sum += cross_r2_by_layer[l]
    count += 1
R_avg = R_sum / count

# Define the desired order.
ordered_reps = ['json', 'chess', 'visual', 'word_grid', 'row_desc', 'column_desc']
# Get the indices in reps_list corresponding to the ordered representations.
indices = [reps_list.index(rep) for rep in ordered_reps]
# Reorder the matrix accordingly.
R_ordered = R_avg[indices, :][:, indices]

# Create custom annotations: two decimals unless value < -10, in which case use integer.
annot = np.empty_like(R_ordered, dtype=object)
for i in range(R_ordered.shape[0]):
    for j in range(R_ordered.shape[1]):
        value = R_ordered[i, j]
        if value < -10:
            annot[i, j] = f"{int(value)}"
        else:
            annot[i, j] = f"{value:.2f}"

# Plot the average cross-prediction matrix as a heatmap.
plt.figure(figsize=(8, 6))
sns.heatmap(R_ordered, annot=annot, fmt="", cmap="hot", vmin=-15, vmax=1,
            xticklabels=ordered_reps, yticklabels=ordered_reps, square=True)
plt.title("Average Cross-Prediction R² Across Layers")
plt.xlabel("Test Set Representation")
plt.ylabel("Model Trained on Representation")
plt.tight_layout()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()


# Get the ordered list of representations.
reps_list = representations  # existing order (we'll reorder later)
n_reps = len(reps_list)

# -----------------------------
# Compute Cross-Prediction R² Matrix (averaged across layers, excluding layer 0)
# -----------------------------
cross_r2_by_layer = {}
for l in range(n_layers):
    R = np.zeros((n_reps, n_reps))  # rows: model from rep_i, columns: test set from rep_j
    for i, rep_i in enumerate(reps_list):
        model_i = results[rep_i][l]["model"]
        for j, rep_j in enumerate(reps_list):
            X_test_j = results[rep_j][l]["X_test"]
            Y_test_j = results[rep_j][l]["Y_test"]
            r2_val = r2_score(Y_test_j, model_i.predict(X_test_j))
            R[i, j] = r2_val
    cross_r2_by_layer[l] = R

# Average across layers 1 to n_layers-1 (exclude layer 0)
R_sum = np.zeros((n_reps, n_reps))
for l in range(1, n_layers):
    R_sum += cross_r2_by_layer[l]
R_avg = R_sum / (n_layers - 1)

# -----------------------------
# Permutation Test for Significance (across layers)
# -----------------------------
n_permutations = 10
# We'll store p-values for each rep_i, rep_j, for layers 1 to n_layers-1
p_values_layers = np.zeros((n_reps, n_reps, n_layers - 1))

for l in range(1, n_layers):  # exclude layer 0
    for i, rep_i in enumerate(reps_list):
        model_i = results[rep_i][l]["model"]
        for j, rep_j in enumerate(reps_list):
            X_test_j = results[rep_j][l]["X_test"]
            Y_test_j = results[rep_j][l]["Y_test"]
            observed_r2 = r2_score(Y_test_j, model_i.predict(X_test_j))
            null_scores = []
            for _ in range(n_permutations):
                # Shuffle the target values to break any real relationship.
                Y_test_shuffled = np.random.permutation(Y_test_j)
                null_r2 = r2_score(Y_test_shuffled, model_i.predict(X_test_j))
                null_scores.append(null_r2)
            null_scores = np.array(null_scores)
            p_val = np.mean(null_scores >= observed_r2)
            p_values_layers[i, j, l - 1] = p_val

# Average p-values over layers.
p_values_matrix = np.mean(p_values_layers, axis=2)

# -----------------------------
# Bonferroni Correction
# -----------------------------
# Total comparisons: each pair (n_reps * n_reps)
total_comparisons = n_reps * n_reps
p_values_matrix_adj = np.minimum(p_values_matrix * total_comparisons, 1.0)

# -----------------------------
# Reorder Matrices
# -----------------------------
ordered_reps = ['json', 'chess', 'visual', 'word_grid', 'row_desc', 'column_desc']
indices = [reps_list.index(rep) for rep in ordered_reps]
R_ordered = R_avg[indices, :][:, indices]
p_values_ordered = p_values_matrix_adj[indices, :][:, indices]

# -----------------------------
# Create Annotation Matrix with Asterisks for Significance
# -----------------------------
annot = np.empty_like(R_ordered, dtype=object)
for i in range(R_ordered.shape[0]):
    for j in range(R_ordered.shape[1]):
        value = R_ordered[i, j]
        # Append an asterisk if the corrected p-value is below 0.05.
        sig_marker = "*" if p_values_ordered[i, j] < 0.05 else ""
        if value < -10:
            annot[i, j] = f"{int(value)}{sig_marker}"
        else:
            annot[i, j] = f"{value:.2f}{sig_marker}"

# -----------------------------
# Plot the Annotated Heatmap
# -----------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(R_ordered, annot=annot, fmt="", cmap="hot", vmin=-15, vmax=1,
            xticklabels=ordered_reps, yticklabels=ordered_reps, square=True)
plt.title("Average Cross-Prediction R² Across Layers\n(Asterisk: Bonferroni-adjusted p < 0.05)")
plt.xlabel("Test Set Representation")
plt.ylabel("Model Trained on Representation")
plt.tight_layout()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()


n_permutations = 10  # Or use a smaller number like 10 for a quick test; here we'll use 1000 for robustness.

# Create a structure to store the permutation test results for each cross-prediction pair across layers.
# We will average results over layers 1:n_layers (excluding layer 0, as before).
cross_perm_results = {}  # keys will be (rep_i, rep_j), values: list of p-values from each layer

for rep_i in reps_list:
    for rep_j in reps_list:
        pvals_layers = []  # to store the p-value for each layer
        for l in range(31, n_layers):  # excluding layer 0
            model_i = results[rep_i][l]["model"]
            X_test_j = results[rep_j][l]["X_test"]
            Y_test_j = results[rep_j][l]["Y_test"]

            # Observed performance.
            observed_r2 = r2_score(Y_test_j, model_i.predict(X_test_j))

            # Build the null distribution by shuffling Y_test.
            null_scores = []
            for _ in range(n_permutations):
                Y_test_shuffled = np.random.permutation(Y_test_j)
                null_r2 = r2_score(Y_test_shuffled, model_i.predict(X_test_j))
                null_scores.append(null_r2)
            null_scores = np.array(null_scores)

            # p-value: fraction of permutations that yield an R² >= the observed R².
            p_val = np.mean(null_scores >= observed_r2)
            pvals_layers.append(p_val)
        
        # For reporting, you could average the p-value over layers or report the worst-case (largest) p-value.
        avg_p_val = np.mean(pvals_layers)
        cross_perm_results[(rep_i, rep_j)] = avg_p_val

# Adjust for multiple comparisons using Bonferroni correction.
# Total number of comparisons is n_reps * n_reps.
total_comparisons = n_reps * n_reps

for key, p_val in cross_perm_results.items():
    p_adj = min(p_val * total_comparisons, 1.0)
    cross_perm_results[key] = p_adj


# Get the ordered list of representations.
reps_list = representations  # existing order (we'll reorder later)
n_reps = len(reps_list)

# Create a dictionary to hold a cross-prediction matrix per layer.
cross_r2_by_layer = {}

for l in range(n_layers):
    R = np.zeros((n_reps, n_reps))  # rows: model from rep_i, columns: test set from rep_j
    for i, rep_i in enumerate(reps_list):
        model_i = results[rep_i][l]["model"]
        for j, rep_j in enumerate(reps_list):
            X_test_j = results[rep_j][l]["X_test"]
            Y_test_j = results[rep_j][l]["Y_test"]
            Y_pred = model_i.predict(X_test_j)
            r2_val = r2_score(Y_test_j, Y_pred)
            R[i, j] = r2_val
    cross_r2_by_layer[l] = R

R_avg = cross_r2_by_layer[32]

# Define the desired order.
ordered_reps = ['json', 'chess', 'visual', 'word_grid', 'row_desc', 'column_desc']
# Get the indices in reps_list corresponding to the ordered representations.
indices = [reps_list.index(rep) for rep in ordered_reps]
# Reorder the matrix accordingly.
R_ordered = R_avg[indices, :][:, indices]

# Create custom annotations: two decimals unless value < -10, in which case use integer.
annot = np.empty_like(R_ordered, dtype=object)
for i in range(R_ordered.shape[0]):
    for j in range(R_ordered.shape[1]):
        value = R_ordered[i, j]
        if value < -10:
            annot[i, j] = f"{int(value)}"
        else:
            annot[i, j] = f"{value:.2f}"

# Plot the average cross-prediction matrix as a heatmap.
plt.figure(figsize=(8, 6))
sns.heatmap(R_ordered, annot=annot, fmt="", cmap="hot", vmin=-15, vmax=1,
            xticklabels=ordered_reps, yticklabels=ordered_reps, square=True)
plt.title("Cross-Prediction R² For Last Layer")
plt.xlabel("Test Set Representation")
plt.ylabel("Model Trained on Representation")
plt.tight_layout()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()


# For the last layer only.
layer = n_layers - 1

# Compute the cross-prediction R² matrix for the last layer.
n_reps = len(representations)
R_last = np.zeros((n_reps, n_reps))
for i, rep_i in enumerate(representations):
    model_i = results[rep_i][layer]["model"]
    for j, rep_j in enumerate(representations):
        X_test_j = results[rep_j][layer]["X_test"]
        Y_test_j = results[rep_j][layer]["Y_test"]
        r2_val = r2_score(Y_test_j, model_i.predict(X_test_j))
        R_last[i, j] = r2_val

# Define the desired order.
ordered_reps = ['json', 'chess', 'visual', 'word_grid', 'row_desc', 'column_desc']
indices = [representations.index(rep) for rep in ordered_reps]
R_ordered = R_last[indices, :][:, indices]

# -----------------------------
# Permutation Test for Significance in Last Layer
# -----------------------------
n_permutations = 10
p_values_matrix = np.zeros((n_reps, n_reps))
for i, rep_i in enumerate(representations):
    model_i = results[rep_i][layer]["model"]
    for j, rep_j in enumerate(representations):
        X_test_j = results[rep_j][layer]["X_test"]
        Y_test_j = results[rep_j][layer]["Y_test"]
        observed_r2 = r2_score(Y_test_j, model_i.predict(X_test_j))
        null_scores = []
        for _ in range(n_permutations):
            Y_test_shuffled = np.random.permutation(Y_test_j)
            null_r2 = r2_score(Y_test_shuffled, model_i.predict(X_test_j))
            null_scores.append(null_r2)
        null_scores = np.array(null_scores)
        print(null_scores)
        print(observed_r2)
        p_value = np.mean(null_scores >= observed_r2)
        p_values_matrix[i, j] = p_value

# Reorder the p-value matrix.
p_ordered = p_values_matrix[indices, :][:, indices]

# Apply Bonferroni correction.
total_comparisons = len(ordered_reps) ** 2
p_ordered_adj = np.minimum(p_ordered * total_comparisons, 1.0)

# -----------------------------
# Create Annotation Matrix with Asterisks for Significance
# -----------------------------
annot = np.empty_like(R_ordered, dtype=object)
for i in range(R_ordered.shape[0]):
    for j in range(R_ordered.shape[1]):
        value = R_ordered[i, j]
        sig_marker = "*" if p_ordered_adj[i, j] < 0.05 else ""
        if value < -10:
            annot[i, j] = f"{int(value)}{sig_marker}"
        else:
            annot[i, j] = f"{value:.2f}{sig_marker}"

# -----------------------------
# Plot the Heatmap for the Last Layer
# -----------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(R_ordered, annot=annot, fmt="", cmap="hot", vmin=-15, vmax=1,
            xticklabels=ordered_reps, yticklabels=ordered_reps, square=True)
plt.title("Cross-Prediction R² For Last Layer\n(Asterisk: Bonferroni-adjusted p < 0.05)")
plt.xlabel("Test Set Representation")
plt.ylabel("Model Trained on Representation")
plt.tight_layout()
pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()


if save_to_pdf:
    # Close the PdfPages object to finalize the PDF file
    pdf_file.close()