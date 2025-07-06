

import pandas as pd
from llm_interface import LLM_Interface
from feature_selector import FeatureSelector
from evaluation import evaluate_feature_subset

import pandas as pd

def load_local_dataset(filepath: str) -> pd.DataFrame:
    """
    Charge un dataset local à partir d'un fichier CSV, Excel ou Parquet.
    Exemple :
        df = load_local_dataset("data/my_file.csv")
    """
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.endswith(".xlsx") or filepath.endswith(".xls"):
        df = pd.read_excel(filepath)
    elif filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Format de fichier non pris en charge: {filepath}")
    
    print(f"✅ Dataset chargé : {filepath} | {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


if __name__ == "__main__":
    # --- 1. Configuration ---
    # Choisissez un modèle de ~4B de paramètres
    # MODEL_ID = "microsoft/phi-2" 
    # models--sshleifer--tiny-gpt2 omdel de 1.5B pour tester
    # MODEL_ID = "sshleifer/tiny-gpt2"  # Exemple de modèle de 1.5B, mais vous pouvez choisir un modèle plus petit si nécessaire

    # model de 4B pour tester
    # MODEL_ID = "meta-llama/Meta-Llama-3-4B-Instruct"  # Exemple de modèle de 4B, mais vous pouvez choisir un modèle plus petit si nécessaire
    # MODEL_ID = "meta-llama/Meta-Llama-3-8B 

    MODEL_ID = "google/gemma-2-2b-it"  # Exemple de modèle de 8B, mais vous pouvez choisir un modèle plus petit si nécessaire

    
    # Pour tester, utilisons un jeu de données simple de scikit-learn
    from sklearn.datasets import load_breast_cancer
    cancer = load_breast_cancer(as_frame=True)
    
    dataset = cancer.frame

    # dataset = load_local_dataset("data/Benign_BruteForce_Mirai_balanced.csv")

    CONCEPTS = [col for col in dataset.columns if col != "Label"]
    TARGET_COLUMN = "Label"
    TASK_DESCRIPTION = "si le trafic réseau est begign ou malicious"

    # --- 2. Initialisation des modules ---
    llm_interface = LLM_Interface(model_id=MODEL_ID)
    selector = FeatureSelector(llm_interface=llm_interface)

    # --- 3. Exécution de la sélection de caractéristiques (LLM-SCORE) ---
    print("\n--- Démarrage de la sélection avec LLM-SCORE ---")
    feature_scores = selector.get_scores(concepts=CONCEPTS, task_description=TASK_DESCRIPTION)
    
    print("\nScores des caractéristiques obtenus :")
    for feature, score in feature_scores.items():
        print(f"- {feature}: {score:.4f}")

    sorted_features = list(feature_scores.keys())

    # --- 4. Évaluation en aval ---
    print("\n--- Démarrage de l'évaluation en aval ---")
    
    # Évaluer avec différents pourcentages de caractéristiques
    proportions = [0.1, 0.3, 0.5, 0.7, 1.0]

    
    for prop in proportions:
        num_features = int(len(sorted_features) * prop)
        if num_features == 0: continue
        
        subset = sorted_features[:num_features]
        
        auroc = evaluate_feature_subset(
            data=dataset,
            selected_features=subset,
            target_column=TARGET_COLUMN
        )
        
        print(f"AUROC avec les {len(subset)} ({prop*100}%) meilleures caractéristiques : {auroc:.4f}")