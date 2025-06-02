import ollama
import os
import time

ollama_api_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')

try:
    print(f"--- Attempting to connect to Ollama server at: {ollama_api_url} ---")
    client = ollama.Client(host=ollama_api_url)

    # --- Step 1: Verify model availability ---
    print("\n--- Listing available models on the Ollama server ---")
    models_list = client.list() # Fetches a list of models currently available on the server

    # --- IMPORTANT DEBUG STEP ---
    print(f"Raw models_list response from Ollama: {models_list}")
    # --- END DEBUG STEP ---

    if not isinstance(models_list, dict) or 'models' not in models_list:
        print("ERROR: Ollama client.list() did not return expected dictionary with 'models' key.")
        exit(1)

    found_model = False
    for model_info in models_list.get('models', []):
        if not isinstance(model_info, dict) or 'name' not in model_info:
            print(f"WARNING: Found a model entry without a 'name' key: {model_info}")
            continue # Skip this malformed entry

        if model_info['name'].startswith('llama3'): # Check for the model name you pulled
            print(f"SUCCESS: Found model: {model_info['name']}")
            found_model = True
            break

    if not found_model:
        print("ERROR: The 'llama3' model was not found on the Ollama server. "
              "Please ensure 'ollama pull llama3' ran successfully in the workflow, "
              "and that the server has fully started.")
        exit(1)

    # --- Step 2: Perform a chat completion request ---
    messages_for_chat = [
        {'role': 'user', 'content': 'What is the capital of Japan?'},
    ]

    print("\n--- Sending a chat completion request to the 'llama3' model ---")
    response = client.chat(
        model='llama3', # Specify the model to use for this request
        messages=messages_for_chat,
    )

    print("\nOllama Response (Capital of Japan):")
    print(response['message']['content'].strip())

    # --- Step 3: Demonstrate a streaming response (optional) ---
    print("\n--- Demonstrating a streaming chat completion response ---")
    stream_messages = [
        {'role': 'user', 'content': 'Tell me a very short, cheerful story about a cat.'},
    ]
    stream_response = client.chat(
        model='llama3',
        messages=stream_messages,
        stream=True
    )
    print("Streaming story:")
    for chunk in stream_response:
        print(chunk['message']['content'], end='', flush=True)
    print("\n\n--- Streaming response complete ---")

except ollama.ResponseError as e:
    print(f"\nERROR: Ollama API encountered an issue!")
    print(f"Details: {e.error} (HTTP Status Code: {e.status_code})")
    print("Possible reasons: Ollama server not fully started, model not pulled, or incorrect model name.")
    exit(1)
except Exception as e:
    print(f"\nERROR: An unexpected error occurred during Ollama interaction: {e}")
    print("This might be due to the Ollama server not being fully ready, "
          "or an issue with the response format. Check the raw `models_list` output above.")
    exit(1)
