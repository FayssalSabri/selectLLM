
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize
import time
# def evaluate_feature_subset(
#     data: pd.DataFrame, 
#     selected_features: list[str], 
#     target_column: str,
#     model: BaseEstimator = None  # modèle par défaut
# ) -> dict:
#     if model is None:
#         from sklearn.ensemble import RandomForestClassifier
#         model = RandomForestClassifier(n_estimators=100, random_state=42)

#     if not selected_features:
#         print("Aucune feature sélectionnée => AUROC=0.5, ACC=0.5")
#         return {"auc": 0.5, "accuracy": 0.5}

#     numeric_features = [
#         f for f in selected_features if pd.api.types.is_numeric_dtype(data[f])
#     ]
#     if not numeric_features:
#         print("Aucune feature numérique => AUROC=0.5, ACC=0.5")
#         return {"auc": 0.5, "accuracy": 0.5}

#     X = data[numeric_features].copy()
#     y = data[target_column]

#     X.replace([np.inf, -np.inf], np.nan, inplace=True)
#     X.dropna(inplace=True)
#     y = y.loc[X.index]

#     if X.empty:
#         print("Plus de lignes après nettoyage => AUROC=0.5, ACC=0.5")
#         return {"auc": 0.5, "accuracy": 0.5}

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     if X_train.empty or X_test.empty:
#         print("Train ou test vide => AUROC=0.5, ACC=0.5")
#         return {"auc": 0.5, "accuracy": 0.5}

#     if len(set(y_train)) < 2 or len(set(y_test)) < 2:
#         print("Train ou test mono-classe => AUROC=0.5, ACC=0.5")
#         return {"auc": 0.5, "accuracy": 0.5}

#     model.fit(X_train, y_train)

#     # Tentative pour obtenir un score pour AUROC
#     y_scores = None
#     try:
#         predictions_proba = model.predict_proba(X_test)
#         # predictions_proba shape: (n_samples, n_classes)
#         y_scores = predictions_proba
#     except AttributeError:
#         try:
#             decision_scores = model.decision_function(X_test)
#             y_scores = decision_scores
#         except AttributeError:
#             y_scores = None

#     if y_scores is not None:
#         try:
#             classes = np.unique(y_train)
#             if len(classes) > 2:
#                 # multi-classe
#                 y_test_bin = label_binarize(y_test, classes=classes)
#                 # y_scores shape doit être (n_samples, n_classes)
#                 auc = roc_auc_score(y_test_bin, y_scores, multi_class='ovr')
#             else:
#                 # binaire
#                 # On récupère la colonne positive si proba shape est (n_samples, 2)
#                 if y_scores.ndim == 2 and y_scores.shape[1] == 2:
#                     y_scores_pos = y_scores[:, 1]
#                 else:
#                     y_scores_pos = y_scores.ravel()
#                 auc = roc_auc_score(y_test, y_scores_pos)
#         except Exception as e:
#             print(f"Erreur AUROC: {e} => AUROC=0.5")
#             auc = 0.5
#     else:
#         print("Pas de probas ou scores disponibles => AUROC=0.5")
#         auc = 0.5

#     predictions = model.predict(X_test)
#     acc = accuracy_score(y_test, predictions)

#     return {"auc": auc, "accuracy": acc}

def evaluate_feature_subset_with_time(
    data: pd.DataFrame,
    selected_features: list[str],
    target_column: str,
    model: BaseEstimator
) -> dict:
    """
    Evaluates a subset of features and measures training/inference time.
    """
    if not selected_features:
        return {"auc": 0.5, "accuracy": 0.5, "training_time": 0, "inference_time": 0}

    numeric_features = [f for f in selected_features if pd.api.types.is_numeric_dtype(data[f])]
    if not numeric_features:
        return {"auc": 0.5, "accuracy": 0.5, "training_time": 0, "inference_time": 0}

    X = data[numeric_features].copy()
    y = data[target_column]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y = y.loc[X.index]

    if X.empty:
        return {"auc": 0.5, "accuracy": 0.5, "training_time": 0, "inference_time": 0}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if X_train.empty or X_test.empty or len(set(y_train)) < 2 or len(set(y_test)) < 2:
        return {"auc": 0.5, "accuracy": 0.5, "training_time": 0, "inference_time": 0}

    # --- Training time ---
    start_train_time = time.time()
    model.fit(X_train, y_train)
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    # --- Inference time ---
    start_infer_time = time.time()
    predictions = model.predict(X_test)
    end_infer_time = time.time()
    inference_time = end_infer_time - start_infer_time

    # --- Metrics calculation ---
    acc = accuracy_score(y_test, predictions)
    
    y_scores = None
    try:
        y_scores = model.predict_proba(X_test)
    except AttributeError:
        try:
            y_scores = model.decision_function(X_test)
        except AttributeError:
            pass # y_scores remains None

    auc = 0.5 # default
    if y_scores is not None:
        try:
            classes = np.unique(y_train)
            if len(classes) > 2:
                y_test_bin = label_binarize(y_test, classes=classes)
                auc = roc_auc_score(y_test_bin, y_scores, multi_class='ovr')
            else:
                y_scores_pos = y_scores[:, 1] if y_scores.ndim == 2 and y_scores.shape[1] == 2 else y_scores.ravel()
                auc = roc_auc_score(y_test, y_scores_pos)
        except Exception:
            pass # auc remains 0.5
            
    return {"auc": auc, "accuracy": acc, "training_time": training_time, "inference_time": inference_time}
