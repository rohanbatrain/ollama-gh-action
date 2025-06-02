import ollama
import os
import time

# Get the Ollama API URL from the environment variable set in the GitHub Actions workflow.
# Default to localhost if the environment variable is not found (useful for local testing).
ollama_api_url = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')

try:
    print(f"Connecting to Ollama server at: {ollama_api_url}")
    # Initialize the Ollama client with the specified host.
    client = ollama.Client(host=ollama_api_url)

    # Verify that the model is available by listing local models.
    print("Listing available models...")
    models_list = client.list()
    found_model = False
    for model_info in models_list.get('models', []):
        if model_info['name'].startswith('llama3'): # Check for llama3 or whatever model you pulled
            print(f"Found model: {model_info['name']}")
            found_model = True
            break
    if not found_model:
        print("Error: 'llama3' model not found. Ensure it was pulled successfully in the workflow.")
        exit(1)

    # Define the messages for the chat completion.
    messages = [
        {'role': 'user', 'content': 'What is the capital of Japan?'},
    ]

    print("\nSending request to Ollama model (llama3)...")
    # Make a chat completion request to the Ollama server.
    response = client.chat(
        model='llama3', # Use the name of the model you pulled
        messages=messages,
    )

    # Print the model's response.
    print("\nOllama Response:")
    print(response['message']['content'])

    # Example of streaming response (optional)
    print("\n--- Streaming Response Example ---")
    stream_messages = [
        {'role': 'user', 'content': 'Tell me a very short, funny story.'},
    ]
    stream_response = client.chat(
        model='llama3',
        messages=stream_messages,
        stream=True # Enable streaming
    )
    for chunk in stream_response:
        # Print each chunk as it arrives
        print(chunk['message']['content'], end='', flush=True)
    print("\n--- End Streaming Response Example ---\n")


except ollama.ResponseError as e:
    # Handle API-specific errors from Ollama (e.g., model not found, server error).
    print(f"Ollama API Error: {e.error} (Status Code: {e.status_code})")
    print("Please check the Ollama server logs and ensure the model is correctly pulled.")
    exit(1)
except Exception as e:
    # Handle any other unexpected errors.
    print(f"An unexpected error occurred: {e}")
    exit(1)
