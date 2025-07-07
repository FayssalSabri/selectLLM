import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class LLM_Interface:
    """
    Une classe d'interface pour charger un modèle de langage open-source
    et générer du texte à partir d'un prompt.
    """
    def __init__(self, model_id: str):
        """
        Initialise et charge le modèle et le tokenizer.
        
        Args:
            model_id (str): L'identifiant du modèle sur le Hub Hugging Face.
        """
        print(f"Chargement du modèle : {model_id}...")
        
        # Configuration pour charger le modèle en 4-bit (économie de mémoire)
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.float16
        # )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True # Nécessaire pour certains modèles comme Phi-2
        )
        print("Modèle chargé avec succès.")

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Génère une réponse textuelle à partir d'un prompt.
        
        Args:
            prompt (str): Le prompt à envoyer au modèle.
            max_new_tokens (int): Le nombre maximum de tokens à générer.
            
        Returns:
            str: La réponse textuelle du modèle.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        # Le slicing [inputs.input_ids.shape[1]:] retire le prompt de la sortie
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return response_text.strip()