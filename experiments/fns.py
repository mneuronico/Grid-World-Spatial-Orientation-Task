from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

def load_model(model_id, dtype = torch.float16, load_in_8bit=False):
    # Step 1: Load model and tokenizer
    print("Step 1: Loading model and tokenizer...")
    print(f"Model ID: {model_id}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

    try:
        if load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                load_in_8bit=True,
                device_map="auto",
                output_hidden_states=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map="auto",
                output_hidden_states=True
            )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
      
    return model, tokenizer
    
    
def build_tokenized_input(user_message: str, tokenizer, system = "Answer briefly."):
    """
    Formats a user message according to the system and user/assistant roles,
    tokenizes the input for the model, and identifies the token indices for the user message.

    Args:
        user_message (str): The user's input message.
        tokenizer: The tokenizer associated with the LLaMA model.

    Returns:
        tuple: A dictionary of tokenized inputs and a tuple (start_idx, end_idx)
               representing the token indices for the user message.
    """
    # Define the system and assistant prompts
    system_role = "<|start_header_id|>system<|end_header_id|>\n"
    user_role = "<|start_header_id|>user<|end_header_id|>\n"
    assistant_role = "<|start_header_id|>assistant<|end_header_id|>\n"

    # Combine into the full formatted string
    formatted_input = f"{system_role}{system}<|eot_id|>{user_role}{user_message}<|eot_id|>{assistant_role}"

    # Tokenize the formatted input
    tokenized_input = tokenizer(formatted_input, return_tensors="pt")

    # Decode tokens to locate the user message
    decoded_tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])

    # Locate the "user<|eot_id|>" sequence
    user_start_token = "user"
    eot_token = "<|end_header_id|>"
    try:
        user_start_idx = decoded_tokens.index(user_start_token)
        eot_idx = decoded_tokens.index(eot_token, user_start_idx)  # Find <|eot_id|> after "user"
        user_message_start_idx = eot_idx + 2  # First token after "user<|eot_id|>"
    except ValueError:
        raise ValueError("Could not locate 'user<|end_header_id|>' sequence in tokens.")

    # Find the end of the user message
    user_tokenized = tokenizer.tokenize(user_message)
    user_message_end_idx = user_message_start_idx + len(user_tokenized)

    user_message_extracted_tokens = decoded_tokens[user_message_start_idx:user_message_end_idx]
    print("Extracted Tokens:", user_message_extracted_tokens)

    return tokenized_input, (user_message_start_idx, user_message_end_idx)
    
    
def get_response(prompt, model, tokenizer, system = "You are a helpful assistant who gives accurate but brief answers.", max_length = 128):
    try:
        # Use the build_tokenized_input function to prepare the formatted input and get indices
        tokenized_input, user_indices = build_tokenized_input(prompt, tokenizer, system = system)
        user_start_idx, user_end_idx = user_indices
        print(f"User Message Indices: Start={user_start_idx}, End={user_end_idx}")
    except Exception as e:
        print(f"Error preparing input: {e}")
        raise
        
    output = model.generate(
        tokenized_input["input_ids"].to("cuda"),
        max_length=max_length,           # Limit the length of the response
        return_dict_in_generate=True,   # Return hidden states and other outputs
        output_hidden_states=True       # Include hidden states in the outputs
    )
    
    generated_tokens = output.sequences[0]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    generated_response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>\n")[1]
    generated_response = generated_response.split("<|eot_id|>")[0]
    print(f"Generated Response: {generated_response}")
    
    return output, generated_response, tokenized_input
    
    
def get_user_message_indices(output, tokenizer):
    token_ids = output.sequences[0].tolist()
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Tokenized form of markers
    user_start_tokens = tokenizer.tokenize("<|start_header_id|>user<|end_header_id|>\n")
    eot_token = "<|eot_id|>"
    
    try:
        # Find the sequence `<|start_header_id|>user<|end_header_id|>\n` in the tokens
        for i in range(len(decoded_tokens) - len(user_start_tokens) + 1):
            if decoded_tokens[i:i + len(user_start_tokens)] == user_start_tokens:
                user_start_idx = i + len(user_start_tokens)
                break
        else:
            raise ValueError("Could not locate user section in the tokenized input.")
        
        # Locate the first `<|eot_id|>` after the user message starts
        for j in range(user_start_idx, len(decoded_tokens)):
            if decoded_tokens[j] == eot_token:
                user_message_end_idx = j
                break
        else:
            raise ValueError("Could not find end of user message.")
        
    except ValueError as e:
        raise ValueError(f"Error locating user message: {e}")
    
    return user_start_idx, user_message_end_idx

def get_assistant_response_indices(output, tokenizer):
    token_ids = output.sequences[0].tolist()
    decoded_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Tokenized form of markers
    assistant_start_tokens = tokenizer.tokenize("<|start_header_id|>assistant<|end_header_id|>\n")
    eot_token = "<|eot_id|>"
    
    try:
        # Find the sequence `<|start_header_id|>assistant<|end_header_id|>\n` in the tokens
        for i in range(len(decoded_tokens) - len(assistant_start_tokens) + 1):
            if decoded_tokens[i:i + len(assistant_start_tokens)] == assistant_start_tokens:
                assistant_start_idx = i + len(assistant_start_tokens)
                break
        else:
            raise ValueError("Could not locate assistant section in the tokenized input.")
        
        # Locate the first `<|eot_id|>` after the assistant response starts
        for j in range(assistant_start_idx, len(decoded_tokens)):
            if decoded_tokens[j] == eot_token:
                assistant_response_end_idx = j
                break
        else:
            assistant_response_end_idx = len(decoded_tokens)  # Assistant response goes to the end
        
    except ValueError as e:
        raise ValueError(f"Error locating assistant response: {e}")
    
    return assistant_start_idx, assistant_response_end_idx

def get_custom_string_indices(output, tokenizer, custom_string):
    token_ids = output.sequences[0].tolist()
    # Decode the tokens to get the full text
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    
    # Find the character positions of the custom string in the text
    start_char = text.find(custom_string)
    if start_char == -1:
        raise ValueError(f"Custom string '{custom_string}' not found in text.")
    end_char = start_char + len(custom_string)
    
    # Tokenize the text with offsets to get mapping from tokens to character positions
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding['offset_mapping']
    
    # Find the token indices that cover the custom string
    start_idx = None
    end_idx = None
    for idx, (start, end) in enumerate(offsets):
        if start <= start_char < end:
            start_idx = idx
        if start < end_char <= end:
            end_idx = idx + 1  # +1 because the end index is exclusive
            break
        if start_idx is not None and end >= end_char:
            end_idx = idx + 1
            break
    if start_idx is None or end_idx is None:
        raise ValueError(f"Could not map custom string '{custom_string}' to token indices.")
    
    return start_idx, end_idx


def get_hidden_states(output):
    """
    Collects all hidden states from the model's output into a single numpy array.

    Args:
        hidden_states: A tuple where each element corresponds to a generation step.
                       Each element is a tuple of hidden states from all layers at that step.
                       The first element contains hidden states for all input tokens,
                       while subsequent elements contain hidden states for one token each.

    Returns:
        A numpy array of shape (layers, tokens, params), where:
            - layers: Number of transformer layers in the model.
            - tokens: Total number of tokens (input + generated).
            - params: Dimensionality of the hidden states.
    """
    
    hidden_states = output.hidden_states
    
    # Number of layers (e.g., 29)
    num_layers = len(hidden_states[0])

    # Number of input tokens (from the first generation step)
    input_tokens = hidden_states[0][0].shape[1]

    # Total number of generation steps (including the initial step)
    generation_steps = len(hidden_states)

    # Number of generated tokens (excluding the initial step)
    generated_tokens = generation_steps

    # Total number of tokens (input tokens + generated tokens)
    total_tokens = input_tokens + generated_tokens

    # Dimensionality of the hidden states (e.g., 4096)
    hidden_size = hidden_states[0][0].shape[2]

    # Initialize an array to hold all hidden states
    hidden_states_array = np.zeros((num_layers, total_tokens, hidden_size))

    # Iterate over each layer
    for layer in range(num_layers):
        # Extract hidden states for the input tokens from the first generation step
        # Shape: (1, input_tokens, hidden_size)
        input_hidden_states = hidden_states[0][layer][0, :, :].cpu().numpy()
        hidden_states_array[layer, :input_tokens, :] = input_hidden_states

        # Extract hidden states for the generated tokens from subsequent steps
        for t in range(1, generation_steps):
            # Index of the generated token in the total sequence
            token_index = input_tokens + t - 1

            # Hidden state for the generated token at this layer and step
            # Shape: (1, 1, hidden_size)
            generated_hidden_state = hidden_states[t][layer][0, 0, :].cpu().numpy()
            hidden_states_array[layer, token_index, :] = generated_hidden_state

    return hidden_states_array
    
    
def get_prompt_hidden_states(output, tokenizer):
    hidden_states = get_hidden_states(output)
    user_start_idx, user_end_idx = get_user_message_indices(output, tokenizer)
    return hidden_states[:, user_start_idx:user_end_idx, :]
    
def get_response_hidden_states(output, tokenizer):
    hidden_states = get_hidden_states(output)
    assistant_start_idx, assistant_end_idx = get_assistant_response_indices(output, tokenizer)
    return hidden_states[:, assistant_start_idx:assistant_end_idx, :]
    
def get_custom_hidden_states(output, tokenizer, custom_string):
    hidden_states = get_hidden_states(output)
    custom_start_idx, custom_end_idx = get_custom_string_indices(output, tokenizer, custom_string)
    return hidden_states[:, custom_start_idx:custom_end_idx, :]