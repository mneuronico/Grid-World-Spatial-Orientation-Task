from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

class Agent:
    def __init__(self, model, tokenizer, system="You are a helpful assistant who gives accurate but brief answers.", max_length=128, save_message_history=False):
        self.model = model
        self.tokenizer = tokenizer
        self.system_message = system
        self.max_length = max_length
        self.save_message_history = save_message_history
        # Initialize message history with system message
        self.message_history = [{'role': 'system', 'content': system}]
        
    def run(self, prompt, mindreading=False, mask = None):
        # Append user message to history
        if self.save_message_history:
            # Use self.message_history
            current_message_history = self.message_history
        else:
            # Create a copy of the message history up to system message
            current_message_history = [msg for msg in self.message_history]
        
        current_message_history.append({'role': 'user', 'content': prompt})
        
        # Build tokenized input from message history
        try:
            tokenized_input, _ = build_tokenized_input(current_message_history, self.tokenizer)
        except Exception as e:
            print(f"Error preparing input: {e}")
            raise

        # --- Setup ablation hooks if mask is provided ---
        hooks = []
        if mask is not None:
            # Expect mask shape: (num_layers, hidden_dim)
            num_layers, hidden_dim = mask.shape
            # Access the transformer layers from model.model.layers
            transformer_layers = self.model.model.layers
            if len(transformer_layers) != num_layers:
                raise ValueError(f"Provided mask has {num_layers} layers but model has {len(transformer_layers)} layers.")
            # Register a forward hook on each layer.
            for i, layer in enumerate(transformer_layers):
                # Convert boolean mask to float tensor (1.0 where True means ablate).
                layer_mask = torch.tensor(mask[i], dtype=torch.float32, device=self.model.device)
                # Reshape to (1, 1, hidden_dim) for proper broadcasting.
                layer_mask = layer_mask.view(1, 1, -1)
                hook_fn = create_ablation_hook(layer_mask)
                hook_handle = layer.register_forward_hook(hook_fn)
                hooks.append(hook_handle)

        # Generate response
        output = self.model.generate(
            tokenized_input["input_ids"].to("cuda"),
            max_new_tokens=self.max_length,
            return_dict_in_generate=True,
            output_hidden_states=True
        )

        # Remove hooks.
        for handle in hooks:
            handle.remove()
        
        # Extract assistant's response
        generated_tokens = output.sequences[0]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        # Extract the assistant's response from the generated text
        generated_response = extract_assistant_response(generated_text)
        print(f"Generated Response: {generated_response}")
        
        if self.save_message_history:
            # Append assistant's response to message history
            self.message_history.append({'role': 'assistant', 'content': generated_response})
        
        if mindreading:
            # Get hidden states for prompt and response
            prompt_hidden_states = get_prompt_hidden_states(output, self.tokenizer)
            response_hidden_states = get_response_hidden_states(output, self.tokenizer)
            return generated_response, output, prompt_hidden_states, response_hidden_states
        else:
            return generated_response

# Define a forward hook creator for ablation.
def create_ablation_hook(layer_mask_tensor):
    def hook(module, input, output):
        if isinstance(output, tuple):
            # Use the dtype of the first element of the tuple.
            effective_mask = layer_mask_tensor.to(output[0].dtype)
            modified_tensor = output[0] * (1.0 - effective_mask)
            # Return a new tuple with the modified tensor as the first element.
            return (modified_tensor,) + output[1:]
        else:
            effective_mask = layer_mask_tensor.to(output.dtype)
            return output * (1.0 - effective_mask)
    return hook

def build_tokenized_input(message_history, tokenizer):
    """
    Formats the message history according to the system and user/assistant roles,
    tokenizes the input for the model, and identifies the token indices for the last user message.

    Args:
        message_history (list): A list of dictionaries with 'role' and 'content'.
        tokenizer: The tokenizer associated with the model.

    Returns:
        tuple: A dictionary of tokenized inputs and a tuple (start_idx, end_idx)
               representing the token indices for the last user message.
    """
    # Define the role prompts
    system_role = "<|start_header_id|>system<|end_header_id|>\n"
    user_role = "<|start_header_id|>user<|end_header_id|>\n"
    assistant_role = "<|start_header_id|>assistant<|end_header_id|>\n"
    eot_token = "<|eot_id|>"
    
    # Build the formatted input
    formatted_messages = ""
    for message in message_history:
        if message['role'] == 'system':
            formatted_messages += f"{system_role}{message['content']}{eot_token}"
        elif message['role'] == 'user':
            formatted_messages += f"{user_role}{message['content']}{eot_token}"
        elif message['role'] == 'assistant':
            formatted_messages += f"{assistant_role}{message['content']}{eot_token}"
        else:
            raise ValueError(f"Unknown role: {message['role']}")
    # Always end with assistant role prompt, as per the model's expected format
    formatted_messages += f"{assistant_role}"
    
    # Tokenize the formatted input
    tokenized_input = tokenizer(formatted_messages, return_tensors="pt")
    
    return tokenized_input, None  # Indices can be obtained later if needed

def extract_assistant_response(generated_text):
    """
    Extracts the assistant's response from the generated text.
    """
    # Split by assistant role marker
    assistant_role_marker = "<|start_header_id|>assistant<|end_header_id|>\n"
    parts = generated_text.split(assistant_role_marker)
    if len(parts) < 2:
        raise ValueError("Assistant's response not found in the generated text.")
    # The last part after the last assistant role marker
    last_assistant_response = parts[-1]
    # Remove anything after the eot_token
    last_assistant_response = last_assistant_response.split("<|eot_id|>")[0]
    return last_assistant_response.strip()

def get_current_step_indices(output, tokenizer):
    """
    Gets the token indices for the last user message and the last assistant response.

    Returns:
        tuple: (user_start_idx, user_end_idx, assistant_start_idx, assistant_end_idx)
    """
    token_ids = output.sequences[0].tolist()
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Tokenized markers
    user_start_tokens = tokenizer.tokenize("<|start_header_id|>user<|end_header_id|>\n")
    assistant_start_tokens = tokenizer.tokenize("<|start_header_id|>assistant<|end_header_id|>\n")
    eot_token = "<|eot_id|>"
    
    # Find all user message starts
    user_starts = []
    for i in range(len(decoded_tokens) - len(user_start_tokens) + 1):
        if decoded_tokens[i:i + len(user_start_tokens)] == user_start_tokens:
            user_starts.append(i + len(user_start_tokens))
    if not user_starts:
        raise ValueError("Could not find user messages in tokens.")
    last_user_start_idx = user_starts[-1]
    
    # User message ends at next eot_token after user start
    for i in range(last_user_start_idx, len(decoded_tokens)):
        if decoded_tokens[i] == eot_token:
            last_user_end_idx = i
            break
    else:
        raise ValueError("Could not find end of last user message.")
    
    # Find all assistant message starts
    assistant_starts = []
    for i in range(len(decoded_tokens) - len(assistant_start_tokens) + 1):
        if decoded_tokens[i:i + len(assistant_start_tokens)] == assistant_start_tokens:
            assistant_starts.append(i + len(assistant_start_tokens))
    if not assistant_starts:
        raise ValueError("Could not find assistant messages in tokens.")
    last_assistant_start_idx = assistant_starts[-1]
    
    # Assistant message ends at next eot_token or end of sequence
    for i in range(last_assistant_start_idx, len(decoded_tokens)):
        if decoded_tokens[i] == eot_token:
            last_assistant_end_idx = i
            break
    else:
        last_assistant_end_idx = len(decoded_tokens)
    
    return last_user_start_idx, last_user_end_idx, last_assistant_start_idx, last_assistant_end_idx

def get_hidden_states(output):
    """
    Collects all hidden states from the model's output into a single numpy array.

    Args:
        output: The model's output containing hidden states.

    Returns:
        A numpy array of shape (layers, tokens, hidden_size).
    """
    hidden_states = output.hidden_states
    
    # Number of layers
    num_layers = len(hidden_states[0])

    # Number of input tokens (from the first generation step)
    input_tokens = hidden_states[0][0].shape[1]

    # Total number of generation steps (including the initial step)
    generation_steps = len(hidden_states)

    # Number of generated tokens (excluding the initial step)
    generated_tokens = generation_steps - 1

    # Total number of tokens (input tokens + generated tokens)
    total_tokens = input_tokens + generated_tokens

    # Dimensionality of the hidden states
    hidden_size = hidden_states[0][0].shape[2]

    # Initialize an array to hold all hidden states
    hidden_states_array = np.zeros((num_layers, total_tokens, hidden_size))

    # Iterate over each layer
    for layer in range(num_layers):
        # Extract hidden states for the input tokens from the first generation step
        input_hidden_states = hidden_states[0][layer][0, :, :].cpu().numpy()
        hidden_states_array[layer, :input_tokens, :] = input_hidden_states

        # Extract hidden states for the generated tokens from subsequent steps
        for t in range(1, generation_steps):
            # Index of the generated token in the total sequence
            token_index = input_tokens + t - 1

            # Hidden state for the generated token at this layer and step
            generated_hidden_state = hidden_states[t][layer][0, 0, :].cpu().numpy()
            hidden_states_array[layer, token_index, :] = generated_hidden_state

    return hidden_states_array

def get_prompt_hidden_states(output, tokenizer):
    """
    Gets the hidden states corresponding to the last user message.
    """
    hidden_states = get_hidden_states(output)
    user_start_idx, user_end_idx, _, _ = get_current_step_indices(output, tokenizer)
    return hidden_states[:, user_start_idx:user_end_idx, :]

def get_response_hidden_states(output, tokenizer):
    """
    Gets the hidden states corresponding to the last assistant response.
    """
    hidden_states = get_hidden_states(output)
    _, _, assistant_start_idx, assistant_end_idx = get_current_step_indices(output, tokenizer)
    return hidden_states[:, assistant_start_idx:assistant_end_idx, :]

def get_custom_string_indices(output, tokenizer, custom_string):
    token_ids = output.sequences[0].tolist()
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Get current step indices
    user_start_idx, user_end_idx, assistant_start_idx, assistant_end_idx = get_current_step_indices(output, tokenizer)
    
    # Extract the assistant's tokens
    assistant_tokens = decoded_tokens[assistant_start_idx:assistant_end_idx]
    assistant_text = tokenizer.convert_tokens_to_string(assistant_tokens)
    
    # Find the custom string in the assistant's text
    start_char = assistant_text.find(custom_string)
    if start_char == -1:
        raise ValueError(f"Custom string '{custom_string}' not found in the assistant's response.")
    end_char = start_char + len(custom_string)
    
    # Map character positions back to token indices
    cumulative_length = 0
    start_idx = None
    end_idx = None
    for idx, token in enumerate(assistant_tokens):
        token_str = tokenizer.convert_tokens_to_string([token])
        token_length = len(token_str)
        if start_idx is None and cumulative_length + token_length > start_char:
            start_idx = assistant_start_idx + idx
        if cumulative_length + token_length >= end_char:
            end_idx = assistant_start_idx + idx + 1
            break
        cumulative_length += token_length
    if start_idx is None or end_idx is None:
        raise ValueError(f"Could not map custom string '{custom_string}' to token indices.")
    
    return start_idx, end_idx


def get_custom_hidden_states(output, tokenizer, custom_string):
    """
    Gets the hidden states corresponding to the custom string in the last interaction.
    """
    hidden_states = get_hidden_states(output)
    start_idx, end_idx = get_custom_string_indices(output, tokenizer, custom_string)
    return hidden_states[:, start_idx:end_idx, :]
