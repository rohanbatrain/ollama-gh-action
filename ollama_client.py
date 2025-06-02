import ollama
import os
import time

ollama_api_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')

try:
    print(f"--- Attempting to connect to Ollama server at: {ollama_api_url} ---")
    client = ollama.Client(host=ollama_api_url)

    # --- Debugging (can be removed after verifying fix) ---
    # print("\n--- Listing available models (for debug) ---")
    # models_response = client.list()
    # print(f"Raw models_response from Ollama: {models_response}")
    #
    # # New way to access models from the updated Ollama client response
    # if hasattr(models_response, 'models') and isinstance(models_response.models, list):
    #     print("Models found (via new structure):")
    #     for model_info in models_response.models:
    #         # Access attributes directly, e.g., model_info.name
    #         print(f"- {model_info.model}") # Use .model for the full name like 'llama3:latest'
    # else:
    #     print("WARNING: Could not interpret models list from client.list() response.")
    # --- End Debugging ---

    # --- Direct Prompt Test ---
    print("\n--- Running a direct prompt test with llama3 ---")
    test_prompt = "What is the capital of Canada?"
    print(f"Prompt: '{test_prompt}'")

    response = client.chat(
        model='llama3', # Ensure this model was pulled in the GitHub Action workflow
        messages=[
            {'role': 'user', 'content': test_prompt},
        ],
    )

    print("\nOllama Response to test prompt:")
    print(response['message']['content'].strip())

    # --- Demonstrating a streaming response (optional, as in previous examples) ---
    print("\n--- Demonstrating a streaming chat completion response ---")
    stream_messages = [
        {'role': 'user', 'content': 'Tell me a very short, cheerful story about a cat.'},
    ]
    stream_response = client.chat(
        model='llama3',
        messages=stream_messages,
        stream=True # Set stream=True to receive the response in chunks
    )
    print("Streaming story:")
    for chunk in stream_response:
        print(chunk['message']['content'], end='', flush=True) # flush=True ensures immediate output
    print("\n\n--- Streaming response complete ---")


except ollama.ResponseError as e:
    print(f"\nERROR: Ollama API encountered an issue!")
    print(f"Details: {e.error} (HTTP Status Code: {e.status_code})")
    print("Possible reasons: Ollama server not fully started, model not pulled, or incorrect model name.")
    print(f"OLLAMA_API_URL used: {ollama_api_url}")
    exit(1)
except Exception as e:
    print(f"\nERROR: An unexpected error occurred during Ollama interaction: {e}")
    print("This might be due to the Ollama server not being fully ready, "
          "or an issue with the response format. Check the raw `models_response` output above.")
    print(f"OLLAMA_API_URL used: {ollama_api_url}")
    exit(1)
