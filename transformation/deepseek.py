import requests

class ChatModel:
    def __init__(self, api_key, model="deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, messages, temperature=0.7, max_tokens=512):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()  # <-- Return the full dict, not just message
    def run_prompt(self, messages, temperature=0.7, max_tokens=512):
        """
        Run a single-prompt conversation using this ChatModel instance.
        """
        try:
            response = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {e}"