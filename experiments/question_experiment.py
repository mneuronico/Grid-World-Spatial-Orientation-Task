import pandas as pd
from Agent import *
from fns import *
import os
import pickle

def load_and_label(csv_file, spatial_label):
    """
    Loads a CSV file with an "examples" column, renames it to "prompt",
    and adds a "spatial" column with the provided label (1 or 0).
    """
    df = pd.read_csv(csv_file, encoding='latin1')
    df = df.rename(columns={"examples": "prompt"})
    df["spatial"] = spatial_label
    return df

# List of file names and their corresponding spatial label (1 for spatial, 0 for non-spatial)
files = [
    ("spatial_reasoning_examples_1.csv", 1),
    ("non_spatial_reasoning_examples_1.csv", 0),
    ("spatial_reasoning_examples_2.csv", 1),
    ("non_spatial_reasoning_examples_2.csv", 0)
]

# Load each file and collect the DataFrames in a list
dfs = [load_and_label(file, label) for file, label in files]

# Concatenate all DataFrames into a single DataFrame
dataset = pd.concat(dfs, ignore_index=True)

# Add an "id" column with unique ids starting from 0
dataset.reset_index(inplace=True)
dataset.rename(columns={'index': 'id'}, inplace=True)

# Display the first few rows of the final dataset
print(dataset.head())



model_id = "Llama-3.1-8B-Instruct"
model_size = 8
output_folder = r"N:\XAI - LLM GridWorld\Experiment 3"

model, tokenizer = load_model("meta-llama/"+model_id, load_in_8bit=True)

system = "You have been tasked with answering reasoning questions. Please answer precisely and briefly."


for index, row in dataset.iterrows():
    i = row['id']
    prompt = row['prompt']
    spatial = row['spatial']

    agent = Agent(model, tokenizer, system = system, save_message_history=True)
    
    try:
        response, output, prompt_hidden_states, response_hidden_states = agent.run(prompt, mindreading=True)
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
    
    output = {
        "input": {"text": prompt, "activations": prompt_hidden_states},
        "output": {"text": response, "activations": response_hidden_states},
        "spatial": spatial
    }
    
    #save mind history
    mind_history_dir = os.path.join(output_folder, "mind_histories")
    os.makedirs(mind_history_dir, exist_ok=True)
    
    mind_history_path = os.path.join(mind_history_dir, f"{i}.pkl")
    with open(mind_history_path, "wb") as mind_history_file:
        pickle.dump(output, mind_history_file)

    print("id:",i)
    print("prompt:", prompt)
    print("spatial:", spatial)
    print("response:", response)
    print("prompt hidden states", prompt_hidden_states.shape)
    print("response hidden states", response_hidden_states.shape)
    print("\n------------\n")