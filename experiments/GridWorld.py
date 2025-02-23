import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from matplotlib.colors import ListedColormap
import json
import pandas as pd
import time
import traceback
import re
from Agent import *
import pickle
import os

class GridWorld:
    def __init__(self, main_dir, mind_dir=None, dim=2, size=5, max_steps=10):
        self.main_dir = main_dir
        if mind_dir is None:
            self.mind_dir = main_dir
        else:
            self.mind_dir = mind_dir
        self.dim = dim
        self.size = size
        self.max_steps = max_steps
       
    def generate_position(self, size):
        # Generate a random position within the grid
        position = (random.randint(0, size - 1), random.randint(0, size - 1))
        
        # Calculate the center position
        center_position = (size // 2, size // 2)
        
        # Check if the random position is the center; if so, regenerate
        while position == center_position:
            position = (random.randint(0, size - 1), random.randint(0, size - 1))
        
        return position
        
    def reset(self):
        """Initialize the game with random positions for the agent and the goal, and set up tracking."""
        # Check for dataset.csv file
        dataset_path = os.path.join(self.main_dir, "dataset.csv")
        if os.path.exists(dataset_path):
            dataset = pd.read_csv(dataset_path)
            self.id = dataset["id"].max() + 1
        else:
            # Create a new dataset.csv file with the necessary columns
            dataset = pd.DataFrame(columns=[
                "id", "model_tested", "model_size", "world_dimension", "world_size", "max_steps",
                "initial_agent_x", "initial_agent_y", "goal_x", "goal_y", "representation", "success",
                "steps_taken", "initial_distance", "final_distance"
            ])
            dataset.to_csv(dataset_path, index=False)
            self.id = 0

        # Initialize game state
        if self.size != 3:
            self.agent_position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
        else:
            self.agent_position = self.generate_position(self.size)
            
        self.goal_position = self._place_goal()
        self.steps_taken = 0
        self.action_history = []
        self.path_history = []
        self._update_grid()

    def _place_goal(self):
        """Place the goal at a random position, ensuring it's not the same or adjacent to the agent's starting position."""
        while True:
            goal_position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if goal_position != self.agent_position and not self._is_adjacent(goal_position, self.agent_position):
                return goal_position

    def _is_adjacent(self, pos1, pos2):
        """Check if two positions are adjacent in the grid."""
        return abs(pos1[0] - pos2[0]) <= 1 and abs(pos1[1] - pos2[1]) <= 1

    def _update_grid(self):
        """Update the grid with the agent and goal positions."""
        self.grid = np.zeros((self.size, self.size))
        self.grid[self.agent_position] = 1  # Agent
        self.grid[self.goal_position] = 2  # Goal

    def extract_action_from_response(self, response):
        # Check if the response is empty or None
        if not response:
            print("Empty or None response received.")
            return None
        
        # Attempt to parse the response directly as JSON
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            # If JSON is invalid, try to extract a valid JSON object using regex
            json_match = re.search(r'(\{.*?"action":\s*"(UP|DOWN|LEFT|RIGHT)"\s*\})', response)
            if json_match:
                try:
                    # Parse the extracted JSON safely
                    data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    
                    # One last attempt: look for "action:" followed by one of the actions
                    action_match = re.search(r'action:\s*"(UP|DOWN|LEFT|RIGHT)"', response)
                    if action_match:
                        action = action_match.group(1)
                        print(f"Extracted action directly from text: {action}")
                        return action
                    else:
                        print("Failed to parse the extracted JSON snippet and no plain text match found.")
                        return None
            else:
                print("No valid JSON with 'action' field found in response.")
                return None
    
        # Ensure that 'action' key is present and has a valid value
        if "action" in data and data["action"] in {"UP", "DOWN", "LEFT", "RIGHT"}:
            return data["action"]
        else:
            print("Invalid 'action' field or value in response JSON.")
            return None



    def manhattan_distance(self, pos1, pos2):
        """Calculate the Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def update(self, action):
        """Update the agent's position based on the action."""
        if self.is_game_over():
            return "Game is over. Please reset the game."

        self.path_history.append(self.agent_position)

        x, y = self.agent_position
        if action == "UP" and x > 0:
            x -= 1
        elif action == "DOWN" and x < self.size - 1:
            x += 1
        elif action == "LEFT" and y > 0:
            y -= 1
        elif action == "RIGHT" and y < self.size - 1:
            y += 1

        new_position = (x, y)
        if new_position != self.agent_position:
            self.agent_position = new_position
            self._update_grid()
        self.steps_taken += 1
        
        self.action_history.append(action)

    def visual_grid(self):
        """Generate the visual representation similar to get_text_representation."""
        return "\n".join(
            [
                " ".join(
                    ["A" if (i, j) == self.agent_position else 
                     "G" if (i, j) == self.goal_position else "." for j in range(self.size)]
                ) for i in range(self.size)
            ]
        )

    def json_representation(self):
        """Generate a JSON representation with grid size and positions of agent and goal."""
        state = {
            "grid_size": self.size,
            "agent_position": {"column": self.agent_position[1], "row": self.agent_position[0]},
            "goal_position": {"column": self.goal_position[1], "row": self.goal_position[0]}
        }
        return json.dumps(state, indent=2)

    def row_description(self):
        """Generate a natural language description of each row of the grid with grouped empty cells."""
        description = []
        for i in range(self.size):
            row_content = []
            empty_count = 0
            
            for j in range(self.size):
                if (i, j) == self.agent_position:
                    if empty_count > 0:
                        row_content.append(f"{empty_count} empty cell{'s' if empty_count > 1 else ''}")
                        empty_count = 0
                    row_content.append("the agent")
                elif (i, j) == self.goal_position:
                    if empty_count > 0:
                        row_content.append(f"{empty_count} empty cell{'s' if empty_count > 1 else ''}")
                        empty_count = 0
                    row_content.append("the goal")
                else:
                    empty_count += 1
            
            if empty_count > 0:
                row_content.append(f"{empty_count} empty cell{'s' if empty_count > 1 else ''}")

            row_description = ", then ".join(row_content)
            description.append(f"The {self._ordinal(i+1)} row has {row_description}.")
        
        return "\n".join(description)

    def column_description(self):
        """Generate a natural language description of each column of the grid with grouped empty cells."""
        description = []
        for j in range(self.size):
            col_content = []
            empty_count = 0
            
            for i in range(self.size):
                if (i, j) == self.agent_position:
                    if empty_count > 0:
                        col_content.append(f"{empty_count} empty cell{'s' if empty_count > 1 else ''}")
                        empty_count = 0
                    col_content.append("the agent")
                elif (i, j) == self.goal_position:
                    if empty_count > 0:
                        col_content.append(f"{empty_count} empty cell{'s' if empty_count > 1 else ''}")
                        empty_count = 0
                    col_content.append("the goal")
                else:
                    empty_count += 1
            
            if empty_count > 0:
                col_content.append(f"{empty_count} empty cell{'s' if empty_count > 1 else ''}")

            column_description = ", then ".join(col_content)
            description.append(f"The {self._ordinal(j+1)} column has {column_description}.")
        
        return "\n".join(description)

    @staticmethod
    def _ordinal(n):
        """Convert an integer into its ordinal representation, e.g., 1 -> 'first', 2 -> 'second'."""
        ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        return ordinals[n-1] if n <= len(ordinals) else f"{n}th"

    def chess_notation(self):
        """Generate a JSON format that provides grid size and positions of the agent and goal using chess notation."""
        # Convert row index to letter (A, B, C...) and column to 1-based index
        agent_position = f"{chr(65 + self.agent_position[1])}{self.agent_position[0] + 1}"
        goal_position = f"{chr(65 + self.goal_position[1])}{self.goal_position[0] + 1}"

        # Create JSON with grid size, agent, and goal positions
        state = {
            "grid_size": self.size,
            "agent_position": agent_position,
            "goal_position": goal_position
        }
        return json.dumps(state, indent=2)

    def word_grid(self):
        """Generate a word-based grid representation where each cell is a descriptive word."""
        word_grid = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if (i, j) == self.agent_position:
                    row.append("agent")
                elif (i, j) == self.goal_position:
                    row.append("goal")
                else:
                    row.append("empty")
            word_grid.append(" ".join(row))
        return "\n".join(word_grid)

    def get_representations(self, representations):
        """
        Concatenate the requested representations into a single string.
        
        :param representations: List of representation identifiers as strings.
        :return: Concatenated string of the specified representations.
        """
        available_representations = {
            "visual": self.visual_grid,
            "json": self.json_representation,
            "row_desc": self.row_description,
            "column_desc": self.column_description,
            "chess": self.chess_notation,
            "word_grid": self.word_grid
        }
        
        selected_representations = [
            available_representations[rep]() for rep in representations if rep in available_representations
        ]
        
        return "\n\n".join(selected_representations)

    def render(self):
        """Render the grid visually in Jupyter using IPython's display.clear_output."""
        # Define custom colors: white for empty, dark blue for agent, yellow for goal, green if both are in the same cell
        cmap = ListedColormap(["white", "darkblue", "yellow", "green"])
        
        clear_output(wait=True)  # Clear the previous plot to update
        plt.figure(figsize=(5, 5))
        
        # Use matshow to display the grid, which might align better in a 5x5 setting
        mat = plt.matshow(self.grid, cmap=cmap, fignum=1, vmin=0, vmax=3)
    
        # Set up grid lines and ticks for better alignment
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))
        plt.gca().set_xticks([x - 0.5 for x in range(1, self.size)], minor=True)
        plt.gca().set_yticks([y - 0.5 for y in range(1, self.size)], minor=True)
        plt.grid(which="minor", color="black", linestyle="-", linewidth=1)
        
        # Adjust title and display
        plt.title(f"Steps Taken: {self.steps_taken}")
        plt.show()

    def is_game_over(self):
        """Check if the game is over."""
        return self.agent_position == self.goal_position

    def sim(self, model, tokenizer, model_id = None, model_size = 0, system = "", representations = ["visual"], render = True, print_rep = True, mask = None):
        self.reset()
        
        agent = Agent(model, tokenizer, system = system, save_message_history=True)
        
        # Calculate the initial Manhattan distance between the agent and the goal
        initial_distance = self.manhattan_distance(self.agent_position, self.goal_position)

        initial_position = self.agent_position
        
        mind_history = []
        
        while (not self.is_game_over()) and self.steps_taken < self.max_steps:
            state = self.get_representations(representations)

            if print_rep:
                print(state)
            
            if render:
                self.render()
    
            if model_id is None:
                action = np.random.choice(["UP", "DOWN", "LEFT", "RIGHT"])
                mind_history.append({"input": {"text": state, "activations": None},
                                     "output": {"text": action, "activations": None},
                                     "decision": {"text": action, "activations": None}})
            else:
                try:
                    response, output, prompt_hidden_states, response_hidden_states = agent.run(state, mindreading=True, mask = mask)
                except Exception as e:
                    print("An error occurred:")
                    traceback.print_exc()
                
                try:
                    action = self.extract_action_from_response(response)
                    
                    if action is not None:
                        decision_hidden_states = get_custom_hidden_states(output, tokenizer, action)
                    else:
                        decision_hidden_states = None
                    
                except Exception as e:
                    print("An error occurred:")
                    traceback.print_exc()
                
                mind_history.append({"input": {"text": state, "activations": prompt_hidden_states},
                                     "output": {"text": response, "activations": response_hidden_states},
                                     "decision": {"text": action, "activations": decision_hidden_states}})

            print("Extracted action:", action)
            self.update(action)

        if print_rep:
            print(state)
        
        if render:
            self.render()
        
        # Calculate the final Manhattan distance between the agent and the goal
        final_distance = self.manhattan_distance(self.agent_position, self.goal_position)
        
        # Record result in dataset.csv
        dataset_path = os.path.join(self.main_dir, "dataset.csv")
        dataset = pd.read_csv(dataset_path)

        all_reps = "+".join(representations)

        # Create a new DataFrame for the current row and concatenate it
        new_row = pd.DataFrame([{
            "id": self.id,
            "model_tested": model_id if model_id is not None else "random",
            "model_size": model_size,
            "world_dimension": self.dim,
            "world_size": self.size,
            "max_steps": self.max_steps,
            "initial_agent_x": initial_position[0],
            "initial_agent_y": initial_position[1],
            "goal_x": self.goal_position[0],
            "goal_y": self.goal_position[1],
            "representation": all_reps,
            "success": self.is_game_over(),
            "steps_taken": self.steps_taken,
            "initial_distance": initial_distance,
            "final_distance": final_distance
        }])

        # Use pd.concat instead of append to add the new row and save back to the dataset
        dataset = pd.concat([dataset, new_row], ignore_index=True)
        dataset.to_csv(dataset_path, index=False)

        # Record path and actions for this trial in paths/{id}.csv
        path_dir = os.path.join(self.main_dir, "paths")
        os.makedirs(path_dir, exist_ok=True)
        trial_path = os.path.join(path_dir, f"{self.id}.csv")
        path_data = pd.DataFrame({
            "step_number": range(len(self.path_history)),
            "agent_x": [pos[0] for pos in self.path_history],
            "agent_y": [pos[1] for pos in self.path_history],
            "action": self.action_history
        })
        path_data.to_csv(trial_path, index=False)
        
        #save mind history
        mind_history_dir = os.path.join(self.mind_dir, "mind_histories")
        os.makedirs(mind_history_dir, exist_ok=True)

        mind_history_path = os.path.join(mind_history_dir, f"{self.id}.pkl")
        with open(mind_history_path, "wb") as mind_history_file:
            pickle.dump(mind_history, mind_history_file)

        print(f"Mind history saved to {mind_history_path}")