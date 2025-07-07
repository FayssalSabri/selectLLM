import logging
import pandas as pd
from llm_interface import LLM_Interface
from feature_selector import FeatureSelector
from evaluation import evaluate_feature_subset
from ollamaInterface import OllamaInterface
import pandas as pd
import sys
import warnings

warnings.filterwarnings("ignore")


# def setup_logger(log_filepath: str):
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)  # Capture tous les niveaux

#     # Formatter commun
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

#     # Handler console (stdout)
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setLevel(logging.INFO)  # Niveau affiché en console
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     # Handler fichier
#     fh = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
#     fh.setLevel(logging.DEBUG)  # Niveau enregistré dans le fichier
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)

#     return logger

    
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
    
    logger.info(f" Dataset chargé : {filepath} | {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


# if __name__ == "__main__":
#     logger = setup_logger("./run.log")  

#     logger.info("--- Démarrage du script ---")
#     # --- 1. Configuration ---
#     # Choisissez un modèle de ~4B de paramètres
#     # MODEL_ID = "microsoft/phi-2" 
#     # models--sshleifer--tiny-gpt2 omdel de 1.5B pour tester
#     # MODEL_ID = "google/gemma-2b"  
#     # MODEL_ID ="meta-llama/Llama-3.2-3B-Instruct"

#     # model de 4B pour tester
#     # MODEL_ID = "meta-llama/Meta-Llama-3-4B-Instruct"  
#     # MODEL_ID = "meta-llama/Meta-Llama-3-8B 

#     # MODEL_ID = "google/gemma-2-2b-it"  # Exemple de modèle de 8B, mais vous pouvez choisir un modèle plus petit si nécessaire

    
#     # Pour tester, utilisons un jeu de données simple de scikit-learn
#     # from sklearn.datasets import load_breast_cancer
#     # cancer = load_breast_cancer(as_frame=True)
    
#     # dataset = cancer.frame

#     dataset = load_local_dataset("data/Benign_BruteForce_Mirai_balanced.csv")

#     CONCEPTS = [col for col in dataset.columns if col != "Label"]
#     TARGET_COLUMN = "Label"
#     TASK_DESCRIPTION = "whether the network traffic is benign or malicious"

#     # --- 2. Initialisation des modules ---
#     # llm_interface = LLM_Interface(model_id=MODEL_ID)
#     ollama_llm = OllamaInterface(model="mistral")

#     selector = FeatureSelector(llm_interface=ollama_llm)

#     # --- 3. Exécution de la sélection de caractéristiques (LLM-SCORE) ---
#     logger.info("\n--- Démarrage de la sélection avec LLM-SCORE ---")
#     feature_scores = selector.get_scores(concepts=CONCEPTS, task_description=TASK_DESCRIPTION)
    

#     logger.info("\nScores des caractéristiques obtenus :")
#     for feature, score in feature_scores.items():
#         logger.info(f"- {feature}: {score:.4f}")

#     sorted_features = list(feature_scores.keys())

#     # --- 4. Évaluation en aval ---
#     logger.info("\n--- Démarrage de l'évaluation en aval ---")
    
#     # Évaluer avec différents pourcentages de caractéristiques
#     proportions = [0.1, 0.3, 0.5, 0.7, 1.0]

#     for prop in proportions:
#         num_features = int(len(sorted_features) * prop)
#         if num_features == 0:
#             continue
    
#         subset = sorted_features[:num_features]
    
#         results = evaluate_feature_subset(
#             data=dataset,
#             selected_features=subset,
#             target_column=TARGET_COLUMN
#         )
    
#         logger.info(f"Pour les {len(subset)} ({prop*100:.0f}%) meilleures caractéristiques :")
#         logger.info(f"  - AUROC    : {results['auc']:.4f}")
#         logger.info(f"  - Accuracy : {results['accuracy']:.4f}\n")

#     logger.info("--- Fin du script ---")



# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC

# if __name__ == "__main__":
#     logger = setup_logger("./run.log")  

#     logger.info("--- Démarrage du script ---")

#     dataset = load_local_dataset("data/Benign_BruteForce_Mirai_balanced.csv")

#     CONCEPTS = [col for col in dataset.columns if col != "Label"]
#     TARGET_COLUMN = "Label"
#     TASK_DESCRIPTION = "whether the network traffic is benign or malicious"

#     ollama_llm = OllamaInterface(model="mistral")
#     selector = FeatureSelector(llm_interface=ollama_llm)

#     logger.info("\n--- Démarrage de la sélection avec LLM-SCORE ---")
#     feature_scores = selector.get_scores(concepts=CONCEPTS, task_description=TASK_DESCRIPTION)

#     logger.info("\nScores des caractéristiques obtenus :")
#     for feature, score in feature_scores.items():
#         logger.info(f"- {feature}: {score:.4f}")

#     sorted_features = sorted(feature_scores, key=feature_scores.get, reverse=True)

#     logger.info("\n--- Démarrage de l'évaluation en aval (multi-modèles) ---")

#     #  Définis tes modèles ici
#     models = {
#         "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
#         "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
#         "SVC": SVC(probability=True, random_state=42),
#     }

#     proportions = [0.1, 0.3, 0.5, 0.7, 1.0]

#     for prop in proportions:
#         num_features = int(len(sorted_features) * prop)
#         if num_features == 0:
#             continue

#         subset = sorted_features[:num_features]

#         logger.info(f"\n######### Pour les {len(subset)} ({prop*100:.0f}%) meilleures caractéristiques #########")

#         for model_name, model in models.items():
#             results = evaluate_feature_subset(
#                 data=dataset,
#                 selected_features=subset,
#                 target_column=TARGET_COLUMN,
#                 model=model
#             )

#             logger.info(f"----------- Modèle : {model_name} -----------")
#             logger.info(f"     ---> AUROC    : {results['auc']:.4f}")
#             logger.info(f"     ---> Accuracy : {results['accuracy']:.4f}")

#     logger.info("--- Fin du script ---")


import logging
import sys

def setup_logger(name: str, log_filepath: str):
    """
    Configure un logger dédié pour chaque LLM.
    """
    logger = logging.getLogger(name)

    # Supprime les handlers précédents si le logger existe déjà
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

if __name__ == "__main__":
    llm_models = ["mistral", "phi3:mini", "llama3.2","gemma3:4b"]  # Tes différents LLM

    for llm_name in llm_models:
        log_file = f"./{llm_name.capitalize()}_log.log"

        logger = setup_logger(llm_name, log_file)
        logger.info(f"===== Début du test pour le LLM: {llm_name} =====")

        # Exemple : ton pipeline complet pour ce LLM
        ollama_llm = OllamaInterface(model=llm_name)
        selector = FeatureSelector(llm_interface=ollama_llm)

        dataset = load_local_dataset("data/Benign_BruteForce_Mirai_balanced.csv")

        CONCEPTS = [col for col in dataset.columns if col != "Label"]
        TARGET_COLUMN = "Label"
        TASK_DESCRIPTION = "whether the network traffic is benign or malicious"

        feature_scores = selector.get_scores(concepts=CONCEPTS, task_description=TASK_DESCRIPTION)

        logger.info("--- Scores des caractéristiques ---")
        for feature, score in feature_scores.items():
            logger.info(f"- {feature}: {score:.4f}")

        sorted_features = sorted(feature_scores, key=feature_scores.get, reverse=True)

        # Exemple : boucle multi-modèles pour l'évaluation
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "SVC": SVC(probability=True, random_state=42),
        }

        for prop in [0.1, 0.3, 0.5, 0.7, 1.0]:
            num_features = int(len(sorted_features) * prop)
            if num_features == 0:
                continue

            subset = sorted_features[:num_features]

            logger.info(f"\n##### {llm_name.upper()} : Pour {prop*100:.0f}% meilleures caractéristiques #####")

            for model_name, model in models.items():
                results = evaluate_feature_subset(
                    data=dataset,
                    selected_features=subset,
                    target_column=TARGET_COLUMN,
                    model=model
                )
                logger.info(f"--> Modèle : {model_name} | AUROC: {results['auc']:.4f} | ACC: {results['accuracy']:.4f}")

        logger.info(f"===== Fin du test pour le LLM: {llm_name} =====\n\n")

