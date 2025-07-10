import json
import re
from llm_interface import LLM_Interface
from collections import defaultdict
import numpy as np
import streamlit as st
class FeatureSelector:
    """
    Implémente les stratégies de sélection de caractéristiques basées sur un LLM.
    """
    def __init__(self, llm_interface: LLM_Interface):
        self.llm = llm_interface

    def _construct_score_prompt(self, concept: str, task: str) -> str:
        return f"""
        You are a machine learning expert.
        your task is to provide an importance score (between 0 and 1) to predict whether {task}.
        The score should reflect the importance of the feature "{concept}" in the context of the task.
        The score should be based on the following criteria:
        - How well does this feature help in distinguishing between different classes?
        - How relevant is this feature to the task at hand?
        - Does this feature provide unique information that is not captured by other features?
        - Is this feature likely to be useful in a real-world scenario for the given task?

    
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
 

class FeatureSelectorP2:
    def __init__(self, llm_interface):
        self.llm = llm_interface

    def _construct_batch_score_prompt(self, concepts: list[str], task: str, data_preview: str) -> str:
        return f"""
        You are a world-class machine learning expert specializing in feature selection.
        Your task is to evaluate the importance of a list of features for a given prediction task.

        **Prediction Task:** "{task}"

        **Candidate Features:**
        {concepts}

        **Data Preview (first 5 rows):**
        ```
        {data_preview}
        ```

        Based on all this information, provide an importance score (from 0.0 for irrelevant to 1.0 for essential) for EACH feature.

        **Output Format:**
        Provide ONLY ONE JSON object inside a markdown code block. The JSON object should map each feature name to its score. Do not include reasoning in this response.

        Example of a correct output format:
        ```json
        {{
            "feature_name_1": 0.8,
            "feature_name_2": 0.3,
            "feature_name_3": 0.95
        }}
        ```
        """

    def _parse_batch_score_response(self, response: str) -> dict:
        """
        Parses the LLM response to extract a JSON object, even if it's not perfectly formatted.
        """
        # First, try to find a JSON object within markdown ```json ... ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        
        # If not found, try to find any JSON object {...} in the string
        if not match:
            match = re.search(r'(\{.*\})', response, re.DOTALL)

        if not match:
            st.warning(f"Impossible de trouver un objet JSON dans la réponse du LLM : {response[:200]}...")
            return {}
            
        json_string = match.group(1)
        try:
            # Clean up the string just in case (e.g., trailing commas)
            # This is a common issue with LLM-generated JSON
            cleaned_json_string = re.sub(r',\s*\}', '}', json_string)
            cleaned_json_string = re.sub(r',\s*\]', ']', cleaned_json_string)
            return json.loads(cleaned_json_string)
        except json.JSONDecodeError as e:
            st.warning(f"Erreur de décodage JSON : {e}. Réponse brute : {json_string[:200]}...")
            return {}

    def get_scores_robust(self, concepts: list[str], task_description: str, data_preview: str, n_runs: int = 1) -> dict[str, float]:
        """
        Implements a robust version of LLM-SCORE using batching and self-consistency.
        """
        all_scores = defaultdict(list)
        
        for i in range(n_runs):
            st.write(f"Exécution de l'analyse LLM n°{i+1}/{n_runs}...")
            prompt = self._construct_batch_score_prompt(concepts, task_description, data_preview)
            response = self.llm.generate(prompt)
            scores = self._parse_batch_score_response(response)
            for feature, score in scores.items():
                if feature in concepts:
                    all_scores[feature].append(float(score))
        
        # Average the scores from all runs
        final_scores = {feature: np.mean(scores) for feature, scores in all_scores.items() if scores}
        
        # Sort scores
        sorted_scores = dict(sorted(final_scores.items(), key=lambda item: item[1], reverse=True))
        return sorted_scores
