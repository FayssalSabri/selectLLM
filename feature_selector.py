
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
        # IMPORTANT : Utilisez ici le template de prompt exact de l'Annexe C de l'article.
        # Le format de sortie JSON en markdown est crucial pour la fiabilité.
        return f"""
        Pour la caractéristique "{concept}", votre tâche est de fournir un score d'importance (entre 0 et 1) pour prédire {task}.

        La sortie doit être un bloc de code markdown au format JSON comme suit :
        ```json
        {{
            "reasoning": "Votre raisonnement logique...",
            "score": 0.0
        }}
        ```
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
            print(f"Prompt pour {concept} : {prompt}")
            response = self.llm.generate(prompt)
            print(f"Réponse du LLM pour {concept} : {response}")
            scores[concept] = self._parse_score_response(response)
        
        # Trier les scores par ordre décroissant
        sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        return sorted_scores

    # Vous pouvez ajouter ici la méthode get_rank pour LLM-RANK de manière similaire.
 