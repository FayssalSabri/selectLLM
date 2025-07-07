# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score, accuracy_score

# def evaluate_feature_subset(
#     data: pd.DataFrame, 
#     selected_features: list[str], 
#     target_column: str
# ) -> dict:
#     """
#     Entraîne et évalue un modèle de classification binaire
#     sur un sous-ensemble de caractéristiques numériques.
    
#     Returns:
#         dict: AUROC et Accuracy obtenus.
#     """
#     if not selected_features:
#         print("Aucune feature sélectionnée => AUROC=0.5, ACC=0.5")
#         return {"auc": 0.5, "accuracy": 0.5}

#     # Garder seulement les features numériques
#     numeric_features = [
#         f for f in selected_features if pd.api.types.is_numeric_dtype(data[f])
#     ]
#     if not numeric_features:
#         print("Aucune feature numérique => AUROC=0.5, ACC=0.5")
#         return {"auc": 0.5, "accuracy": 0.5}

#     X = data[numeric_features].copy()
#     y = data[target_column]

#     # Nettoyage inf et NaN
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

#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)

#     # Pour AUROC
#     predictions_proba = model.predict_proba(X_test)
#     if predictions_proba.shape[1] == 2:
#         y_scores = predictions_proba[:, 1]
#         auc = roc_auc_score(y_test, y_scores)
#     else:
#         auc = roc_auc_score(y_test, predictions_proba, multi_class="ovr")

#     # Pour Accuracy
#     predictions = model.predict(X_test)
#     acc = accuracy_score(y_test, predictions)

#     return {"auc": auc, "accuracy": acc}


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.base import BaseEstimator
from sklearn.preprocessing import label_binarize

def evaluate_feature_subset(
    data: pd.DataFrame, 
    selected_features: list[str], 
    target_column: str,
    model: BaseEstimator = None  # modèle par défaut
) -> dict:
    if model is None:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    if not selected_features:
        print("Aucune feature sélectionnée => AUROC=0.5, ACC=0.5")
        return {"auc": 0.5, "accuracy": 0.5}

    numeric_features = [
        f for f in selected_features if pd.api.types.is_numeric_dtype(data[f])
    ]
    if not numeric_features:
        print("Aucune feature numérique => AUROC=0.5, ACC=0.5")
        return {"auc": 0.5, "accuracy": 0.5}

    X = data[numeric_features].copy()
    y = data[target_column]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y = y.loc[X.index]

    if X.empty:
        print("Plus de lignes après nettoyage => AUROC=0.5, ACC=0.5")
        return {"auc": 0.5, "accuracy": 0.5}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if X_train.empty or X_test.empty:
        print("Train ou test vide => AUROC=0.5, ACC=0.5")
        return {"auc": 0.5, "accuracy": 0.5}

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        print("Train ou test mono-classe => AUROC=0.5, ACC=0.5")
        return {"auc": 0.5, "accuracy": 0.5}

    model.fit(X_train, y_train)

    # Tentative pour obtenir un score pour AUROC
    y_scores = None
    try:
        predictions_proba = model.predict_proba(X_test)
        # predictions_proba shape: (n_samples, n_classes)
        y_scores = predictions_proba
    except AttributeError:
        try:
            decision_scores = model.decision_function(X_test)
            y_scores = decision_scores
        except AttributeError:
            y_scores = None

    if y_scores is not None:
        try:
            classes = np.unique(y_train)
            if len(classes) > 2:
                # multi-classe
                y_test_bin = label_binarize(y_test, classes=classes)
                # y_scores shape doit être (n_samples, n_classes)
                auc = roc_auc_score(y_test_bin, y_scores, multi_class='ovr')
            else:
                # binaire
                # On récupère la colonne positive si proba shape est (n_samples, 2)
                if y_scores.ndim == 2 and y_scores.shape[1] == 2:
                    y_scores_pos = y_scores[:, 1]
                else:
                    y_scores_pos = y_scores.ravel()
                auc = roc_auc_score(y_test, y_scores_pos)
        except Exception as e:
            print(f"Erreur AUROC: {e} => AUROC=0.5")
            auc = 0.5
    else:
        print("Pas de probas ou scores disponibles => AUROC=0.5")
        auc = 0.5

    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    return {"auc": auc, "accuracy": acc}
