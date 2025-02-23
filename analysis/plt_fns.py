import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import sem
import seaborn as sns

def get_representation_colors():
    # Define color and line styles
    return {
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
    
    
def plot_success_bar(data, title, representations, size=5, maxy=1, random_success_rate = None):
    color_line_styles = get_representation_colors()
    
    # Filter data for the selected representations
    filtered = data[data["representation"].isin(representations)]
    success_rates = filtered.groupby("representation")["success"].mean().reindex(representations)
    std_error = filtered.groupby("representation")["success"].sem().reindex(representations)

    # Create the bar plot
    plot_data = pd.DataFrame({
        "Representation": representations,
        "Success Rate": success_rates,
        "Standard Error": std_error
    })
    fig, ax = plt.subplots(figsize=(size, size))
    sns.barplot(
        data=plot_data,
        x="Representation",
        y="Success Rate",
        hue="Representation",  # Assign x to hue
        palette={rep: color_line_styles[rep][0] for rep in representations},
        dodge=False,  # Ensure bars are not shifted
        legend=False,  # Disable the legend
        ax=ax
    )
    
    # Add error bars manually
    ax.errorbar(x=range(len(representations)), y=success_rates, yerr=std_error, fmt='none', ecolor='black', capsize=5)

    # Add random success rate as a dashed line
    if random_success_rate is None:
        random_success_rate = data[data["model_tested"] == "random"]["success"].mean()
    
    ax.axhline(y=random_success_rate, color="gray", linestyle="--")

    # Customize the plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Representation", fontsize=14)
    ax.set_ylabel("Success Rate", fontsize=14)
    ax.set_ylim(0, maxy)
    #ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}"))  # Format y-ticks
    plt.xticks(range(len(representations)), labels=representations, rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()
    
    
def plot_efficiency_bar(data, title, representations, size=5, maxy=1):
    color_line_styles = get_representation_colors()
    
    # Filter data for the selected representations and successful trials
    filtered = data[(data["representation"].isin(representations)) & (data["success"] == True)]
    efficiency = (filtered["initial_distance"] / filtered["steps_taken"]).groupby(filtered["representation"]).mean().reindex(representations)
    std_error = (filtered["initial_distance"] / filtered["steps_taken"]).groupby(filtered["representation"]).sem().reindex(representations)

    # Create the bar plot
    plot_data = pd.DataFrame({
        "Representation": representations,
        "Efficiency": efficiency,
        "Standard Error": std_error
    })
    fig, ax = plt.subplots(figsize=(size, size))
    sns.barplot(
        data=plot_data,
        x="Representation",
        y="Efficiency",
        hue="Representation",  # Assign x to hue
        palette={rep: color_line_styles[rep][0] for rep in representations},
        dodge=False,  # Ensure bars are not shifted
        legend=False,  # Disable the legend
        ax=ax
    )
    
    # Add error bars manually
    ax.errorbar(x=range(len(representations)), y=efficiency, yerr=std_error, fmt='none', ecolor='black', capsize=5)

    # Add random efficiency rate as a dashed line
    random_data = data[(data["model_tested"] == "random") & (data["success"] == True)]
    random_efficiency = (random_data["initial_distance"] / random_data["steps_taken"]).mean()
    ax.axhline(y=random_efficiency, color="gray", linestyle="--", label="Random Efficiency")

    # Customize the plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Representation", fontsize=14)
    ax.set_ylabel("Efficiency", fontsize=14)
    ax.set_ylim(0, maxy)
    ax.legend(fontsize=12)
    plt.xticks(range(len(representations)), labels=representations, rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_final_distance_ratio_bar(data, title, representations, size=5, maxy=1):
    color_line_styles = get_representation_colors()
    
    # Filter data for the selected representations and unsuccessful trials
    filtered = data[(data["representation"].isin(representations)) & (data["success"] == False)]
    final_distance_ratio = (filtered["final_distance"] / filtered["initial_distance"]).groupby(filtered["representation"]).mean().reindex(representations)
    std_error = (filtered["final_distance"] / filtered["initial_distance"]).groupby(filtered["representation"]).sem().reindex(representations)

    # Create the bar plot
    plot_data = pd.DataFrame({
        "Representation": representations,
        "Final Distance Ratio": final_distance_ratio,
        "Standard Error": std_error
    })
    fig, ax = plt.subplots(figsize=(size, size))
    sns.barplot(
        data=plot_data,
        x="Representation",
        y="Final Distance Ratio",
        hue="Representation",  # Assign x to hue
        palette={rep: color_line_styles[rep][0] for rep in representations},
        dodge=False,  # Ensure bars are not shifted
        legend=False,  # Disable the legend
        ax=ax
    )
    
    # Add error bars manually
    ax.errorbar(x=range(len(representations)), y=final_distance_ratio, yerr=std_error, fmt='none', ecolor='black', capsize=5)

    # Add random final distance ratio as a dashed line
    random_data = data[(data["model_tested"].str.lower() == "random") & (data["success"] == False)]
    random_final_distance_ratio = (random_data["final_distance"] / random_data["initial_distance"]).mean()
    ax.axhline(y=random_final_distance_ratio, color="gray", linestyle="--", label="Random Baseline")

    # Customize the plot
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Representation", fontsize=14)
    ax.set_ylabel("Final Distance Ratio", fontsize=14)
    ax.set_ylim(0, maxy)
    ax.legend(fontsize=12)
    plt.xticks(range(len(representations)), labels=representations, rotation=45, fontsize=12)
    plt.tight_layout()
    plt.show()