import os
import glob
import pickle
import numpy as np
import pandas as pd
from Agent import *
from fns import *
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from GridWorld import GridWorld
import traceback

BASE_DIR = r"N:\XAI - LLM GridWorld\Experiment 7"
grid_size = 5
max_steps = 2*grid_size

model_id = "Llama-3.1-8B-Instruct"
model_size = 8
output_folder = BASE_DIR

# File/folder paths
SPATIAL_MASK_PATH = os.path.join(BASE_DIR, "xy_predicting_units.pkl") # correctness_predicting_units_binary.pkl

print("Loading spatial representation mask...")
with open(SPATIAL_MASK_PATH, "rb") as f:
    spatial_mask = pickle.load(f)  # Boolean NumPy array, shape: (layers, params)


# Determine the total number of spatial rep units (number of True values)
n_rep_units = np.sum(spatial_mask)
print(f"Spatial mask loaded. Shape: {spatial_mask.shape}, total rep units: {n_rep_units}")
print(f"{round(100 * n_rep_units / (spatial_mask.shape[0]*spatial_mask.shape[1]),2)}%")


system = f"""You are a navigation assistant tasked with guiding an Agent to a Goal in a {grid_size}x{grid_size} grid world. The Agent will start in a random position, and your objective is to provide directions that bring the Agent to the Goal.

Each time you receive an updated state of the world, choose the optimal next move to bring the Agent closer to the Goal. You may only respond with a JSON object containing a single field named "action", which should contain one of the following strings in all capital letters: "UP", "DOWN", "LEFT", or "RIGHT".

Ensure that your response is strictly in this format, with no additional text or commentary, since it will be used automatically and with no human supervision by a Python script to get the next world state.

Remember:

Your goal is to bring the Agent closer to the Goal as efficiently as possible.
Only respond with the JSON object in the exact format specified, using one of the four allowed action strings."""

model, tokenizer = load_model("meta-llama/"+model_id, load_in_8bit=True)


# Define all available representations
# all_representations = ["visual", "json", "row_desc", "word_grid", "chess", "column_desc",
#                        ["visual", "row_desc"], ["row_desc", "json"], ["json", "visual"], ["visual", "row_desc", "json"]]

#all_representations = ["visual", "json", "row_desc", "word_grid", "chess", "column_desc"]
all_representations = ["json"]
# Initialize the GridWorld
world = GridWorld(output_folder, size=grid_size, max_steps=max_steps)

for i in range(50):
    #filtered_mask = get_percentage_per_layer(spatial_mask, keep_percentage=0.02)
    #filtered_mask = randomize_mask_per_layer(spatial_mask, keep_percentage=0.1)
    
    for rep in all_representations:
        print(rep, i)

        if isinstance(rep, list):
            representations = rep
        else:
            representations = [rep]

        try:
            world.sim(
                model=model,
                tokenizer=tokenizer,
                model_id=model_id,
                model_size=model_size,
                system=system,
                representations=representations,
                render=True,
                print_rep=True,
                mask = spatial_mask[1:] #filtered_mask[1:] # this is the ablation mask, excluding the embedding layer
            )
        except Exception as e:
            print("An error occurred:")
            traceback.print_exc()