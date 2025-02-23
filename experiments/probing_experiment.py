import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pickle

from fns import *
from GridWorld import GridWorld
import traceback

grid_size = 5
max_steps = 2*grid_size

#model_id = "Llama-3.2-3B-Instruct"
#model_size = 3
#output_folder = r"N:\XAI - LLM GridWorld\Experiment 1"
#dtype = torch.float16

model_id = "Llama-3.1-8B-Instruct"
model_size = 8
output_folder = r"N:\XAI - LLM GridWorld\Experiment 2"


system = f"""You are a navigation assistant tasked with guiding an Agent to a Goal in a {grid_size}x{grid_size} grid world. The Agent will start in a random position, and your objective is to provide directions that bring the Agent to the Goal.

Each time you receive an updated state of the world, choose the optimal next move to bring the Agent closer to the Goal. You may only respond with a JSON object containing a single field named "action", which should contain one of the following strings in all capital letters: "UP", "DOWN", "LEFT", or "RIGHT".

Ensure that your response is strictly in this format, with no additional text or commentary, since it will be used automatically and with no human supervision by a Python script to get the next world state.

Remember:

Your goal is to bring the Agent closer to the Goal as efficiently as possible.
Only respond with the JSON object in the exact format specified, using one of the four allowed action strings."""

model, tokenizer = load_model("meta-llama/"+model_id, load_in_8bit=True)



#all_representations = ["visual", "json", "row_desc", "word_grid", "chess", "column_desc"]
#all_representations = ["json"]
all_representations = ["random"]
model, tokenizer, model_id, system = None, None, None, ""
model_size = 0

# Initialize the GridWorld
world = GridWorld(output_folder, size=grid_size, max_steps=max_steps)

for i in range(150):
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
                print_rep=True
            )
        except Exception as e:
            print("An error occurred:")
            traceback.print_exc()