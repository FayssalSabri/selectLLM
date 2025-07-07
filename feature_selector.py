import json
import re
from llm_interface import LLM_Interface

class FeatureSelector:
    """
    Implémente les stratégies de sélection de caractéristiques basées sur un LLM.
    """
    def __init__(self, llm_interface: LLM_Interface):
        self.llm = llm_interface

    def _construct_score_prompt(self, concept: str, task: str) -> str:
        return f"""
        You are a machine learning expert.
        For the feature "{concept}", your task is to provide an importance score (between 0 and 1) to predict whether {task}.
    
        Provide ONLY ONE JSON block inside a markdown code block, with no text before or after.
    
        The output must be a markdown code block in JSON format as follows:
        ```json
        {{
            "reasoning": "Briefly explain why this specific feature is important or not.",
            "score": X
        }}
        ```
    
        Example of a correct answer (DO NOT COPY, adapt to the feature):
        ```json
        {{
            "reasoning": "The number of incoming packets can help detect malicious traffic because attacks often generate high packet rates.",
            "score": 0.8
        }}
        ```
    
        Focus on the specific feature "{concept}". Do not copy the example.
        """


    def _parse_score_response(self, response: str) -> float:
        # Trouve le bloc JSON dans la réponse et l'extrait
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if not match:
            print(f"Avertissement : Impossible de parser la réponse : {response}")
            return 0.0 # Retourne un score par défaut en cas d'échec
        
        try:
            data = json.loads(match.group(1))
            return float(data.get("score", 0.0))
        except (json.JSONDecodeError, TypeError):
            print(f"Avertissement : Erreur de décodage JSON pour : {response}")
            return 0.0

    def get_scores(self, concepts: list[str], task_description: str) -> dict[str, float]:
        """
        Implémente la méthode LLM-SCORE.
        
        Returns:
            dict: Un dictionnaire mappant chaque concept à son score d'importance.
        """
        scores = {}
        for i, concept in enumerate(concepts):
            print(f"Scoring de la caractéristique {i+1}/{len(concepts)} : {concept}")
            prompt = self._construct_score_prompt(concept, task_description)
            response = self.llm.generate(prompt)
            print(f"Réponse du LLM pour {concept} : {response}")
            scores[concept] = self._parse_score_response(response)
        
        # Trier les scores par ordre décroissant
        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        return sorted_scores

    # Vous pouvez ajouter ici la méthode get_rank pour LLM-RANK de manière similaire.
 