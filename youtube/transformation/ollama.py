import ollama
import json
def run_ollama_prompt(model_name, messages):
    """
    Run an Ollama model using structured chat messages.

    :param model_name: The name of the model (e.g., 'llama2', 'mistral')
    :param messages: A list of messages with roles (system, user, etc.)
    :return: Model's response content
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages
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


def build_user_prompt(names, text):
    name_list = ', '.join(names)
    user_prompt= f"""
You are an information extractor. From the transcript below, extract the exact sentences that discuss these names: {name_list}.

For each name:

Return the exact quoted text (no paraphrasing).

Group multiple mentions for the same name as a list.

Return ONLY JSON like this:
{{
"players": {{
"name_1": ["quote 1", "quote 2"],
"name_2": ["quote 1"],
...
}}
}}

Transcript:
{text}
"""
    return [{"role": "user", "content": user_prompt.strip()}]



def build_system_user_prompt(player_names, contexts):
    system_message = """You are a football transfer analyst. Read the text and extract transfer information for the mentioned players.

Focus only on sentences where the player's name is mentioned. Ignore all other text.

Only extract information about player transfers (buying, selling, loaning, rumors, negotiations, interest, talks, discussions, bid, offer, deal, agreement, contract, signing, move, switch, departure, arrival, target, linked, approach, enquiry, pursuit, speculation, considering, exploring, monitoring, tracking, scouting, wants, desires, keen, eyeing, planning, preparing, close to, verge of, set to, expected to, likely to, potential, possible, reported, alleged, suggested, claimed).

If a player's name is not mentioned in any transfer context, write "No transfer news mentioned"."""

    user_prompt = f"""Players: {player_names}

players & their mentions:
{contexts}
Extract transfer information for each player in this format:
Player_name(corrected): transfer information"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]



def extract_json(text):
    """Extracts the first complete JSON object from a string by tracking bracket balance."""
    start = text.find('{')
    if start == -1:
        return None

    bracket_count = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            bracket_count += 1
        elif text[i] == '}':
            bracket_count -= 1

        if bracket_count == 0:
            json_str = text[start:i+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print("JSON decode error:", e)
                return None
    return None
import json

def build_json_system_user_prompt(names, text):
    name_list = ', '.join(name.strip() for name in names)
    cleaned_text = text.strip()

    # Build the fill-in JSON structure
    fill_in_structure = {name.strip(): "" for name in names}
    fill_in_json = json.dumps(fill_in_structure, indent=2, ensure_ascii=False)

    system_message = f"""
You are a summarizer. Given a transcript and a list of names (players, coaches, or clubs), fill in the provided JSON object.

Instructions:
- For each name in the JSON object, provide a SHORT STRING summary focused ONLY on their football future or movement (e.g., transfers, retirements, renewals, appointments, departures).
- ONLY fill in the values directly inside the provided JSON — no new keys, lists, or nested objects.
- Your output must be a valid, flat JSON. Each value must be either:
    - a short string (1–2 sentences max), or
    - null (if the entity is not relevant or mentioned).
- Do NOT create keys that weren’t given. Do NOT add nested structures or arrays.

Fill in the JSON object below with the correct summaries:
{fill_in_json}
"""

    user_prompt = f"""
Transcript:
{cleaned_text}
"""
    return [
        {"role": "system", "content": system_message.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]
