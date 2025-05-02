import ollama
import json
def run_ollama_prompt(model_name, prompt):
    """
    Run an Ollama model with a given prompt.

    :param model_name: The name of the model to use (e.g., 'llama2', 'mistral')
    :param prompt: The input prompt for the model to respond to
    :return: Model's response content
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Example usage
# print(run_ollama_prompt('llama2', 'What is the capital of Japan?'))


def from_str_to_dict(model_output):
    """
    Verifies the output from the Ollama model, parses it as JSON, 
    and extracts the keys and values.

    :param model_output: The JSON-formatted response from the model
    """
    try:
        # Parse the model output as JSON
        data = json.loads(model_output)

        # Print extracted keys and values
        print("Extracted JSON data:")
        # for key, value in data.items():
            # print(f"{key}: {value}")

        return data

    except json.JSONDecodeError:
        print("Response was not valid JSON:")
        print(model_output)
    except Exception as e:
        print(f"Error: {e}")
