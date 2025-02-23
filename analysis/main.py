import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import sem
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

save_to_pdf = False

# Define the experiment directories
experiment_dirs = [d for d in os.listdir('.') if os.path.isdir(d)]

# Initialize a dictionary to hold dataset information
datasets_info = {}

# Define color and line styles
color_line_styles = {
    "json": ("green", "-"),
    "chess": ("green", "--"),
    "visual": ("red", "-"),
    "word_grid": ("red", "--"),
    "row_desc": ("blue", "-"),
    "column_desc": ("blue", "--"),
    "json+visual": ("magenta", "-"),
    "visual+json": ("magenta", "--"),
    "visual+row_desc": ("purple", "-"),
    "row_desc+visual": ("purple", "--"),
    "row_desc+json": ("gold", "-"),
    "json+row_desc": ("gold", "--"),
    "visual+row_desc+json": ("black", "-"),
    "visual+json+row_desc": ("black", "-"),
    "row_desc+column_desc": ("cyan", "-")
}

# Load each dataset and print useful information
for experiment in experiment_dirs:
    dataset_path = os.path.join(experiment, "dataset.csv")
    
    if os.path.exists(dataset_path):
        # Load dataset
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
        
        # Add to datasets_info
        datasets_info[experiment] = dataset_info
        
        # Print dataset summary
        print(f"--- {experiment.upper()} ---")
        for key, value in dataset_info.items():
            print(f"{key}: {value}")
        print()
    else:
        print(f"No dataset.csv found for {experiment}.\n")



# Use the pgf backend for matplotlib
plt.rcdefaults()

# Ensure that fonts are embedded in the PDF
plt.rcParams['pdf.fonttype'] = 42  # Output Type 3 (Type3) fonts
plt.rcParams['ps.fonttype'] = 42  # Output Type 3 (Type3) fonts

if save_to_pdf:
    # Dictionary to hold the PdfPages objects for each region
    pdf_file = PdfPages("Plots analisis v1.pdf")

# Load the 1-goal dataset
dataset_path = os.path.join("1-goal", "dataset.csv")
df = pd.read_csv(dataset_path)

# Filter out random model (model_size = 0)
filtered_df = df[df["model_size"] > 0]

# Calculate mean random success rate
random_success_rate = df[df["model_size"] == 0]["success"].mean()

# Function to compute success rate and standard error
def compute_success_rate(data, representations):
    filtered = data[data["representation"].isin(representations)]
    grouped = filtered.groupby(["model_size", "representation"])
    success_rate = grouped["success"].mean()
    std_error = grouped["success"].apply(sem)
    return success_rate.unstack(), std_error.unstack()

# Function to create square plots with adjustments
def plot_square(data, title, representations, x_log=True, size=5):
    success_rate, std_error = compute_success_rate(data, representations)
    print(success_rate)
    model_sizes = sorted(data["model_size"].unique())  # Unique model sizes for ticks
    
    fig, ax = plt.subplots(figsize=(size, size))
    
    # Plot random success rate as a horizontal dashed gray line
    ax.axhline(y=random_success_rate, color="gray", linestyle="--", label="Random Success Rate")
    
    for rep in representations:
        if rep in success_rate.columns:  # Ensure data exists for the representation
            color, linestyle = color_line_styles.get(rep, ("black", "-"))
            ax.plot(success_rate.index, success_rate[rep], label=rep, color=color, linestyle=linestyle)
            ax.fill_between(success_rate.index, 
                            success_rate[rep] - std_error[rep], 
                            success_rate[rep] + std_error[rep], 
                            color=color, alpha=0.2)
    if x_log:
        ax.set_xscale('log')
    ax.set_xticks(model_sizes)
    ax.set_xticklabels([str(int(x)) for x in model_sizes], fontsize=12)
    ax.set_yticks(np.linspace(0, 1, 5))  # Set y-ticks between 0 and 1
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))  # Format y-ticks
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Model Size (log scale)", fontsize=14)
    ax.set_ylabel("Success Rate", fontsize=14)
    ax.legend(fontsize=12, loc="best")
    plt.tight_layout()
    pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()

# Function to create square subplots with adjustments
def plot_multi_square(data, title, subplots, x_log=True, size=5):
    n_subplots = len(subplots)
    fig, axes = plt.subplots(1, n_subplots, figsize=(size * n_subplots, size))
    fig.suptitle(title, fontsize=16)
    
    model_sizes = sorted(data["model_size"].unique())  # Unique model sizes for ticks
    
    for ax, reps in zip(axes, subplots):
        success_rate, std_error = compute_success_rate(data, reps)
        print(success_rate)
        
        # Plot random success rate as a horizontal dashed gray line
        ax.axhline(y=random_success_rate, color="gray", linestyle="--", label="Random Success Rate")
        
        for rep in reps:
            if rep in success_rate.columns:  # Ensure data exists for the representation
                color, linestyle = color_line_styles.get(rep, ("black", "-"))
                ax.plot(success_rate.index, success_rate[rep], label=rep, color=color, linestyle=linestyle)
                ax.fill_between(success_rate.index, 
                                success_rate[rep] - std_error[rep], 
                                success_rate[rep] + std_error[rep], 
                                color=color, alpha=0.2)
        if x_log:
            ax.set_xscale('log')
        ax.set_xticks(model_sizes)
        ax.set_xticklabels([str(int(x)) for x in model_sizes], fontsize=12)
        ax.set_yticks(np.linspace(0, 1, 5))  # Set y-ticks between 0 and 1
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))  # Format y-ticks
        ax.set_title(", ".join(reps), fontsize=14)
        ax.set_xlabel("Model Size (log scale)", fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.legend(fontsize=10, loc="best")
    plt.tight_layout()
    pdf_file.savefig(plt.gcf()) if save_to_pdf else plt.show()




# Filter dataset based on specified representations and exclude model_size 0
valid_representations = ["json", "chess", "visual", "word_grid", "row_desc", "column_desc"]
filtered_df = df[(df["representation"].isin(valid_representations)) & (df["model_size"] != 0)]

# Compute success rates for each model_size
success_rates = filtered_df.groupby("model_size")["success"].mean()

# Perform Chi-Square test to see the relationship between success rate and model size
contingency_table = pd.crosstab(filtered_df["model_size"], filtered_df["success"])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)


# Prepare data for linear regression
X = filtered_df["model_size"]  # Independent variable
y = filtered_df["success"].astype(int)  # Dependent variable (binary: 0 or 1)

# Add constant for intercept in regression model
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Get slope and p-value
slope = model.params["model_size"]
p_value = model.pvalues["model_size"]

# Define the groups
cartesian = ["json", "chess"]
topographic = ["visual", "word_grid"]
textual = ["row_desc", "column_desc"]

# Ensure the "group" column is correctly assigned for Cartesian vs. Topographic
filtered_df["group"] = filtered_df["representation"].apply(
    lambda x: "cartesian" if x in cartesian else "topographic" if x in topographic else None
)

# Filter only relevant groups
grouped_df = filtered_df[filtered_df["group"].notna()]

# Fit a binomial GLM (logistic regression) for Cartesian vs. Topographic
model_cart_topo = smf.glm("success ~ C(group) + model_size", data=grouped_df, family=sm.families.Binomial()).fit()

# Extract coefficients and p-values
cart_topo_coef = model_cart_topo.params.get("C(group)[T.topographic]", None)
cart_topo_p = model_cart_topo.pvalues.get("C(group)[T.topographic]", None)

# Now assign group for Cartesian vs. Textual
filtered_df["group"] = filtered_df["representation"].apply(
    lambda x: "cartesian" if x in cartesian else "textual" if x in textual else None
)

# Filter only relevant groups
grouped_text_df = filtered_df[filtered_df["group"].notna()]

# Fit a binomial GLM (logistic regression) for Cartesian vs. Textual
model_cart_text = smf.glm("success ~ C(group) + model_size", data=grouped_text_df, family=sm.families.Binomial()).fit()

# Extract coefficients and p-values
cart_text_coef = model_cart_text.params.get("C(group)[T.textual]", None)
cart_text_p = model_cart_text.pvalues.get("C(group)[T.textual]", None)

# Function to compute efficiency as in the provided function
def compute_efficiency(data, representations):
    # Filter data for specified representations and successful trials
    filtered = data[(data["representation"].isin(representations)) & (data["success"])].copy()
    
    # Compute efficiency as initial_distance divided by steps_taken
    filtered["efficiency"] = filtered["initial_distance"] / filtered["steps_taken"]
    
    return filtered

# Compute efficiency for the entire dataset
efficiency_df = compute_efficiency(filtered_df, valid_representations)

### 1. Linear Model for Efficiency ~ Model Size ###
eff_model = smf.ols("efficiency ~ model_size", data=efficiency_df).fit()

# Extract slope and p-value
eff_slope = eff_model.params["model_size"]
eff_p = eff_model.pvalues["model_size"]

### 2. Binomial GLM for Cartesian vs. Topographic on Efficiency ###
efficiency_df["group"] = efficiency_df["representation"].apply(lambda x: "cartesian" if x in cartesian else "topographic" if x in topographic else None)
eff_grouped_df = efficiency_df[efficiency_df["group"].notna()]

eff_model_cart_topo = smf.ols("efficiency ~ C(group) + model_size", data=eff_grouped_df).fit()

# Extract coefficients and p-values
eff_cart_topo_coef = eff_model_cart_topo.params["C(group)[T.topographic]"]
eff_cart_topo_p = eff_model_cart_topo.pvalues["C(group)[T.topographic]"]

### 3. Binomial GLM for Cartesian vs. Textual on Efficiency ###
efficiency_df["group"] = efficiency_df["representation"].apply(lambda x: "cartesian" if x in cartesian else "textual" if x in textual else None)
eff_grouped_text_df = efficiency_df[efficiency_df["group"].notna()]

eff_model_cart_text = smf.ols("efficiency ~ C(group) + model_size", data=eff_grouped_text_df).fit()

# Extract coefficients and p-values
eff_cart_text_coef = eff_model_cart_text.params["C(group)[T.textual]"]
eff_cart_text_p = eff_model_cart_text.pvalues["C(group)[T.textual]"]


# Compute final distance ratio only for unsuccessful trials
def compute_final_distance_ratio(data, representations):
    # Filter for unsuccessful trials in the specified representations
    filtered = data[(data["success"] == False) & (data["representation"].isin(representations))].copy()

    # Compute final distance ratio as final_distance divided by initial_distance
    filtered["final_distance_ratio"] = filtered["final_distance"] / filtered["initial_distance"]

    return filtered

# Compute final distance ratio for the entire dataset
final_distance_df = compute_final_distance_ratio(filtered_df, valid_representations)

### 1. Linear Model for Final Distance Ratio ~ Model Size ###
fdr_model = smf.ols("final_distance_ratio ~ model_size", data=final_distance_df).fit()

# Extract slope and p-value
fdr_slope = fdr_model.params["model_size"]
fdr_p = fdr_model.pvalues["model_size"]

### 2. Cartesian vs. Topographic on Final Distance Ratio ###
final_distance_df["group"] = final_distance_df["representation"].apply(lambda x: "cartesian" if x in cartesian else "topographic" if x in topographic else None)
fdr_grouped_df = final_distance_df[final_distance_df["group"].notna()]

fdr_model_cart_topo = smf.ols("final_distance_ratio ~ C(group) + model_size", data=fdr_grouped_df).fit()

# Extract coefficients and p-values
fdr_cart_topo_coef = fdr_model_cart_topo.params["C(group)[T.topographic]"]
fdr_cart_topo_p = fdr_model_cart_topo.pvalues["C(group)[T.topographic]"]

### 3. Cartesian vs. Textual on Final Distance Ratio ###
final_distance_df["group"] = final_distance_df["representation"].apply(lambda x: "cartesian" if x in cartesian else "textual" if x in textual else None)
fdr_grouped_text_df = final_distance_df[final_distance_df["group"].notna()]

fdr_model_cart_text = smf.ols("final_distance_ratio ~ C(group) + model_size", data=fdr_grouped_text_df).fit()

# Extract coefficients and p-values
fdr_cart_text_coef = fdr_model_cart_text.params["C(group)[T.textual]"]
fdr_cart_text_p = fdr_model_cart_text.pvalues["C(group)[T.textual]"]

# Function to perform t-tests comparing model data against random policy
def compare_with_random_fixed(model_data, random_data):
    if model_data.empty or random_data.empty:
        return float('nan'), float('nan')  # Avoid errors if one group is empty
    
    t_stat, p_value = stats.ttest_ind(model_data["final_distance_ratio"], 
                                      random_data["final_distance_ratio"], 
                                      equal_var=False)  # Welch’s t-test
    return t_stat, p_value


# Define representation groups
cartesian = ["json", "chess"]
topographic = ["visual", "word_grid"]
textual = ["row_desc", "column_desc"]

# Compute final distance ratio for models 8 and 11
filtered_models = compute_final_distance_ratio(df, model_sizes=[8, 11])

# Compute final distance ratio for the random policy group
random_data = df[(df["model_tested"].str.lower() == "random") & (df["success"] == False)].copy()
if not random_data.empty:
    random_data["final_distance_ratio"] = random_data["final_distance"] / random_data["initial_distance"]
    random_data["representation"] = "random"  # Assign a placeholder representation

# Filter data by representation groups
cartesian_data = filtered_models[filtered_models["representation"].isin(cartesian)]
topographic_data = filtered_models[filtered_models["representation"].isin(topographic)]
textual_data = filtered_models[filtered_models["representation"].isin(textual)]

# Perform t-tests for each representation group
cartesian_t, cartesian_p = compare_with_random_fixed(cartesian_data, random_data)
topographic_t, topographic_p = compare_with_random_fixed(topographic_data, random_data)
textual_t, textual_p = compare_with_random_fixed(textual_data, random_data)


filtered_models = compute_final_distance_ratio(df, model_sizes=[1, 3])

# Compute final distance ratio for the random policy group
random_data = df[(df["model_tested"].str.lower() == "random") & (df["success"] == False)].copy()
if not random_data.empty:
    random_data["final_distance_ratio"] = random_data["final_distance"] / random_data["initial_distance"]
    random_data["representation"] = "random"  # Assign a placeholder representation

# Filter data by representation groups
cartesian_data = filtered_models[filtered_models["representation"].isin(cartesian)]
topographic_data = filtered_models[filtered_models["representation"].isin(topographic)]
textual_data = filtered_models[filtered_models["representation"].isin(textual)]

# Perform t-tests for each representation group
cartesian_t, cartesian_p = compare_with_random_fixed(cartesian_data, random_data)
topographic_t, topographic_p = compare_with_random_fixed(topographic_data, random_data)
textual_t, textual_p = compare_with_random_fixed(textual_data, random_data)


filtered_models = compute_final_distance_ratio(df, model_sizes=[70, 90])

# Compute final distance ratio for the random policy group
random_data = df[(df["model_tested"].str.lower() == "random") & (df["success"] == False)].copy()
if not random_data.empty:
    random_data["final_distance_ratio"] = random_data["final_distance"] / random_data["initial_distance"]
    random_data["representation"] = "random"  # Assign a placeholder representation

# Filter data by representation groups
cartesian_data = filtered_models[filtered_models["representation"].isin(cartesian)]
topographic_data = filtered_models[filtered_models["representation"].isin(topographic)]
textual_data = filtered_models[filtered_models["representation"].isin(textual)]

# Perform t-tests for each representation group
cartesian_t, cartesian_p = compare_with_random_fixed(cartesian_data, random_data)
topographic_t, topographic_p = compare_with_random_fixed(topographic_data, random_data)
textual_t, textual_p = compare_with_random_fixed(textual_data, random_data)

######################


def get_data(dataset_name, model_name=None, world_size=None, dimension=None, representations=None, max_steps=None, success=None):
    # Load the dataset
    dataset = pd.read_csv(dataset_name)

    # Ensure model_name and representations are lists if provided
    model_name = [model_name] if isinstance(model_name, str) else model_name
    representations = [representations] if isinstance(representations, str) else representations

    # Filter the dataset based on provided parameters, allowing None values to bypass filtering
    filtered_dataset = dataset[
        (dataset['model_tested'].isin(model_name) if model_name else True) &
        (dataset['world_size'] == world_size if world_size is not None else True) &
        (dataset['world_dimension'] == dimension if dimension is not None else True) &
        (dataset['representation'].isin(representations) if representations else True) &
        (dataset['max_steps'] == max_steps if max_steps is not None else True) &
        (dataset['success'] == success if success is not None else True)
    ]

    return filtered_dataset


data = get_data('dataset.csv', None, 5, 2,
                 representations = None, max_steps = None, success = None)


def visualize_policy(data, paths_dir='paths'):
    """Generates a policy plot for each model and representation type based on actions in failed trials."""
    
    # Get unique models and representations
    models = sorted(data["model_tested"].unique())
    data["representation"] = data["representation"].fillna("random")
    representations = sorted(data["representation"].unique())
    
    # Define action vectors for arrow plotting
    action_vectors = {
        "UP": (0, -1),
        "DOWN": (0, 1),
        "LEFT": (-1, 0),
        "RIGHT": (1, 0)
    }

    for model in models:
        for rep in representations:
            # Filter dataset for the current model and representation
            filtered_dataset = data[
                (data["model_tested"] == model) & 
                (data["representation"] == rep)
            ]
            
            # Skip if no data for this combination
            if filtered_dataset.empty:
                continue

            # Initialize a dictionary to hold actions at each relative position
            policy_data = {}

            # Process each row in the filtered dataset
            for idx, row in filtered_dataset.iterrows():
                trial_id = row['id']
                goal_x = row['goal_x']
                goal_y = row['goal_y']

                # Path to the corresponding path CSV file
                path_file = os.path.join(paths_dir, f"{trial_id}.csv")

                # Check if the path file exists
                if not os.path.isfile(path_file):
                    continue  # Skip if the path file is missing

                # Load the path data
                path_df = pd.read_csv(path_file)

                # Process each step in the path
                for _, step in path_df.iterrows():
                    agent_x = step['agent_x']
                    agent_y = step['agent_y']
                    action = step['action']

                    if action in action_vectors.keys():
                        # Compute relative position (centered at the goal)
                        relative_x = agent_y - goal_y  # Swap x and y
                        relative_y = agent_x - goal_x  # Swap x and y
                        key = (relative_x, relative_y)
    
                        # Initialize the list if the key is not present
                        if key not in policy_data:
                            policy_data[key] = []
    
                        # Append the action to the list for this relative position
                        policy_data[key].append(action)

            # Determine the extent of the grid based on the actual data
            relative_x_values = [key[0] for key in policy_data.keys()]
            relative_y_values = [key[1] for key in policy_data.keys()]

            min_x = min(relative_x_values)
            max_x = max(relative_x_values)
            min_y = min(relative_y_values)
            max_y = max(relative_y_values)

            # Set up the plot with adjusted size and limits
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(min_x - 0.5, max_x + 0.5)
            ax.set_ylim(min_y - 0.5, max_y + 0.5)
            ax.set_xticks(range(min_x, max_x + 1))
            ax.set_yticks(range(min_y, max_y + 1))
            ax.invert_yaxis()  # Match grid world orientation
            ax.grid(False)

            # Plot each cell
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    key = (x, y)
                    square = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor='white', edgecolor='black', linewidth=2)
                    ax.add_patch(square)

                    # Highlight the goal cell
                    if x == 0 and y == 0:
                        square.set_facecolor('yellow')
                        continue  # No arrow on the goal cell

                    # If there is policy data for this cell, plot the arrow
                    if key in policy_data:
                        actions = policy_data[key]
                        total_actions = len(actions)
                        action_counts = pd.Series(actions).value_counts()
                        most_common_action = action_counts.idxmax()
                        confidence = action_counts.max() / total_actions

                        # Map confidence to color (black at 0.25, red at 1.0)
                        c_norm = (confidence - 0.25) / (1.0 - 0.25)
                        c_norm = max(0.0, min(c_norm, 1.0))
                        color = (c_norm, 0, 0)

                        # Get the vector for the most common action
                        dx, dy = action_vectors[most_common_action]

                        # Adjust arrow properties
                        arrow_length = 0.6
                        start_x = x
                        start_y = y
                        dx_arrow = dx * arrow_length
                        dy_arrow = dy * arrow_length

                        # Offset to center the arrow
                        offset_x = -dx_arrow / 2
                        offset_y = -dy_arrow / 2

                        # Plot the arrow
                        ax.arrow(
                            start_x + offset_x, start_y + offset_y, dx_arrow, dy_arrow,
                            head_width=0.3, head_length=0.2, fc=color, ec=color,
                            linewidth=5,  # Increase thickness for better visibility
                            length_includes_head=True
                        )

            # Set labels and title
            ax.set_xlabel('Relative X Position')
            ax.set_ylabel('Relative Y Position')
            ax.set_title(f'Policy Visualization for Model: {model}, Representation: {rep}')

            # Show the plot
            plt.show()



visualize_policy(data)