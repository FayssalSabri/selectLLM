import requests

class OllamaInterface:
    """
    Interface pour interagir avec Ollama via HTTP.
    """

    def __init__(self, model: str = "mistral", host: str = "http://localhost:11434"):
        """
        Args:
            model (str): Nom du modèle Ollama.
            host (str): URL de l'API Ollama locale.
        """
        self.model = model
        self.host = host

    def generate(self, prompt: str) -> str:
        """
        Envoie un prompt au modèle via Ollama.

        Args:
            prompt (str): Texte du prompt.
        
        Returns:
            str: Réponse générée.
        """
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False  # Désactive le streaming pour récupérer tout d'un coup
            }
        )
        if response.status_code != 200:
            raise ValueError(f"Ollama error: {response.status_code} {response.text}")

        return response.json()["response"].strip()