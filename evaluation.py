import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

def evaluate_feature_subset(
    data: pd.DataFrame, 
    selected_features: list[str], 
    target_column: str
) -> float:
    """
    Entraîne et évalue un modèle de classification binaire
    sur un sous-ensemble de caractéristiques numériques.
    
    Returns:
        float: Le score AUROC obtenu.
    """
    if not selected_features:
        print("Aucune feature sélectionnée => AUROC=0.5")
        return 0.5

    # Garder seulement les features numériques
    numeric_features = [
        f for f in selected_features if pd.api.types.is_numeric_dtype(data[f])
    ]
    if not numeric_features:
        print("Aucune feature numérique => AUROC=0.5")
        return 0.5

    X = data[numeric_features].copy()
    y = data[target_column]

    # Nettoyage inf et NaN
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y = y.loc[X.index]

    if X.empty:
        print("Plus de lignes après nettoyage => AUROC=0.5")
        return 0.5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if X_train.empty or X_test.empty:
        print("Train ou test vide => AUROC=0.5")
        return 0.5

    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        print("Train ou test mono-classe => AUROC=0.5")
        return 0.5
    
    

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)

    if predictions.shape[1] == 2:
        # Cas binaire => prends juste proba de la classe positive
        y_scores = predictions[:, 1]
        auc = roc_auc_score(y_test, y_scores)
    else:
        # Cas multi-classes => utilise multi_class
        auc = roc_auc_score(y_test, predictions, multi_class="ovr")

    return auc

