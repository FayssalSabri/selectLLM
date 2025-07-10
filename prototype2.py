import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.base import BaseEstimator
import umap.umap_ as umap
import missingno as msno
import io
import base64
import h2o
from h2o.automl import H2OAutoML
import time
import json
import re
from evaluation import evaluate_feature_subset_with_time_P2 
from feature_selector import FeatureSelectorP2
from ollamaInterface import OllamaInterface



# --- H2O AutoML Section ---
def automl_section(data, target_column, preprocessed_data=None):
    # ... (code for this section remains unchanged)
    pass

# --- Main App Configuration ---
st.set_page_config(page_title='Auto-Prep: Prétraitement & Réduction de Dimension', layout='wide')

# --- Main App Interface ---
st.title('Auto-Prep: Prétraitement Automatique & Réduction de Dimension')
st.markdown("""
Cette application a été améliorée pour se concentrer sur la **réduction de dimension**, un aspect clé de l'optimisation des pipelines de Machine Learning, maintenant avec une **assistance LLM intégrée et une évaluation en aval**.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Étapes:",
    ["1. Chargement des données",
     "2. Exploration des données",
     "3. Gestion des valeurs manquantes",
     "4. Encodage des variables catégorielles",
     "5. Réduction de Dimension & Sélection de Features",
     "6. Modélisation AutoML"])

# Initialize session_state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'llm_scores' not in st.session_state:
    st.session_state.llm_scores = None
if 'classic_selected_features' not in st.session_state:
    st.session_state.classic_selected_features = None


# --- Page 1: Data Loading ---
if page == "1. Chargement des données":
    st.header("1. Chargement de vos données")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            # Reset state for new data
            st.session_state.data = df.copy()
            st.session_state.original_data = df.copy()
            st.session_state.llm_scores = None
            st.session_state.classic_selected_features = None
            if 'evaluation_results_llm' in st.session_state:
                del st.session_state.evaluation_results_llm
            if 'evaluation_results_classic' in st.session_state:
                del st.session_state.evaluation_results_classic

            st.success(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes.")
            st.dataframe(df.head())

            st.subheader("Sélection de la colonne cible (essentiel pour les méthodes supervisées)")
            target_col = st.selectbox("Choisissez la variable cible:", options=[""] + list(df.columns))
            if target_col:
                st.session_state.target_column = target_col
                st.info(f"Colonne cible sélectionnée: **{target_col}**")
        except Exception as e:
            st.error(f"Erreur lors du chargement: {e}")

    if st.button("Ou utilisez des données de démonstration (Cancer)"):
        from sklearn.datasets import load_breast_cancer
        cancer = load_breast_cancer(as_frame=True)
        df = cancer.frame
        st.session_state.data = df.copy()
        st.session_state.original_data = df.copy()
        st.session_state.target_column = 'target'
        st.session_state.llm_scores = None
        st.session_state.classic_selected_features = None
        st.success("Données de démonstration (Cancer) chargées!")
        st.dataframe(df.head())
        # st.rerun()


# --- Other Pages (2, 3, 4) ---
elif page == "2. Exploration des données":
    if st.session_state.data is not None:
        df = st.session_state.data
        st.header("2. Exploration et Analyse des Données")
        st.dataframe(df.describe(include='all').T)
    else:
        st.warning("Veuillez charger des données à l'étape 1.")

elif page == "3. Gestion des valeurs manquantes":
     if st.session_state.data is not None:
        st.header("3. Gestion des Valeurs Manquantes")
        df = st.session_state.data
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            st.write("Colonnes avec des valeurs manquantes :")
            st.dataframe(missing_data)
        else:
            st.success("Aucune valeur manquante détectée.")
     else:
        st.warning("Veuillez charger des données à l'étape 1.")

elif page == "4. Encodage des variables catégorielles":
    if st.session_state.data is not None:
        st.header("4. Encodage des Variables Catégorielles")
        df = st.session_state.data
        _, categorical_cols = detect_data_types(df)
        if categorical_cols:
             st.write(f"Variables catégorielles trouvées : {categorical_cols}")
        else:
            st.success("Aucune variable catégorielle à encoder.")
    else:
        st.warning("Veuillez charger des données à l'étape 1.")


# --- Page 5: Dimensionality Reduction ---
elif page == "5. Réduction de Dimension & Sélection de Features":
    if st.session_state.data is not None:
        df = st.session_state.data
        target = st.session_state.target_column
        st.header("5. Réduction de Dimension & Sélection de Features")

        numeric_df = df.select_dtypes(include=np.number)
        if target and target in numeric_df.columns:
            features = numeric_df.drop(columns=[target])
        else:
            features = numeric_df

        if features.isnull().sum().sum() > 0:
            st.error("Veuillez imputer les valeurs manquantes à l'étape 3 avant de procéder.")
            st.stop()

        tab_titles = [
            "🤖 Sélection & Évaluation par LLM",
            "⚙️ Sélection & Évaluation Classique",
            "🌀 Extraction (Projection)",
            "🎨 Visualisation de Variétés",
        ]
        tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

        with tab1:
            st.subheader("✨ Sélection de Features Assistée par LLM (Approche Robuste)")
            st.info("Cette approche envoie toutes les caractéristiques au LLM en une seule fois pour une meilleure contextualisation et peut être exécutée plusieurs fois pour fiabiliser les scores.")

            if not target:
                st.warning("Veuillez sélectionner une colonne cible à l'étape 1 pour utiliser cette fonctionnalité.")
                st.stop()

            with st.form("llm_robust_form"):
                task_description = st.text_input(
                    "Task:",
                    f"classifie the traffic as malicious or benign"
                )
                llm_name = st.text_input("Nom du modèle Ollama à utiliser:", "mistral")
                n_runs = st.slider("Nombre d'exécutions pour la fiabilisation (auto-consistance):", 1, 5, 1)
                submitted = st.form_submit_button("1. Lancer l'analyse robuste par le LLM")

            if submitted:
                with st.spinner(f"Le modèle '{llm_name}' analyse les features (x{n_runs})..."):
                    try:
                        ollama_llm = OllamaInterface(model=llm_name)
                        selector = FeatureSelectorP2(llm_interface=ollama_llm)
                        concepts = features.columns.tolist()
                        data_preview = st.session_state.original_data.head().to_markdown()
                        
                        feature_scores = selector.get_scores_robust(
                            concepts=concepts, 
                            task_description=task_description,
                            data_preview=data_preview,
                            n_runs=n_runs
                        )

                        if not feature_scores:
                             st.error("L'analyse LLM n'a retourné aucun score. Veuillez vérifier la console et la réponse du LLM.")
                        else:
                            st.success("Analyse terminée!")
                            st.session_state.llm_scores = feature_scores
                            # Clear previous evaluation results
                            if 'evaluation_results_llm' in st.session_state:
                                del st.session_state.evaluation_results_llm
                            st.rerun()

                    except Exception as e:
                        st.error(f"Une erreur est survenue lors de la communication avec le LLM: {e}")
                        st.info("Assurez-vous que le service Ollama est en cours d'exécution et que le modèle spécifié est disponible.")

            if st.session_state.llm_scores:
                st.markdown("---")
                st.subheader("Résultats de l'analyse et Évaluation")
                scores_df = pd.DataFrame(list(st.session_state.llm_scores.items()), columns=['Feature', 'Score']).sort_values('Score', ascending=False)
                
                st.markdown("#### 1. Visualisation des scores")
                st.dataframe(scores_df)
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x='Score', y='Feature', data=scores_df, ax=ax, orient='h')
                ax.set_title("Pertinence des Features (Scores LLM Moyennés)")
                st.pyplot(fig)

                # Evaluation Section
                st.markdown("#### 2. Évaluer la performance par seuil de score")
                
                with st.form("llm_eval_form"):
                    models_to_evaluate = {
                        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
                        "SVC": SVC(probability=True, random_state=42),
                    }
                    selected_models = st.multiselect("Choisissez les modèles pour l'évaluation:", list(models_to_evaluate.keys()), default=["RandomForest"])
                    score_Scores = st.multiselect(
                        "Choisissez les seuils de score minimum à tester:", 
                        options=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0], 
                        default=[0.1, 0.4, 0.7, 0.9]
                    )
                    eval_submitted = st.form_submit_button("Lancer l'évaluation par seuil")

                if eval_submitted:
                    evaluation_results = []
                    with st.spinner("Évaluation en cours..."):
                        for Score in sorted(score_Scores, reverse=True):
                            subset = scores_df[scores_df['Score'] >= Score]['Feature'].tolist()
                            if not subset:
                                st.write(f"Aucune feature avec un score >= {Score}. Saut de l'évaluation.")
                                continue

                            for model_name in selected_models:
                                model = models_to_evaluate[model_name]
                                results = evaluate_feature_subset_with_time_P2(
                                    data=st.session_state.original_data,
                                    selected_features=subset,
                                    target_column=target,
                                    model=model
                                )
                                # Ajout des infos complémentaires au dictionnaire
                                results['Model'] = model_name
                                results['Score'] = Score
                                results['Num_Features'] = len(subset)

                                evaluation_results.append(results)

                    # Conversion de la liste de dicts en DataFrame
                    df_results = pd.DataFrame(evaluation_results)
                    st.success("Évaluation terminée!")


                    # Réorganiser les colonnes pour mettre 'Model', 'Score', 'Num_Features' en premier
                    cols = df_results.columns.tolist()
                    ordered_cols = ['Model', 'Score', 'Num_Features'] + [col for col in cols if col not in ['Model', 'Score', 'Num_Features']]
                    df_results = df_results[ordered_cols]

                    st.session_state.evaluation_results_llm = df_results


                if 'evaluation_results_llm' in st.session_state:
                    st.markdown("---")
                    st.subheader("Résultats de l'évaluation (LLM)")
                    results_df = st.session_state.evaluation_results_llm
                    st.dataframe(results_df)

                    # Convertir le seuil en string pour l'axe X
                    results_df['Score_str'] = results_df.apply(lambda row: f"{row['Score']} ({row['Num_Features']})", axis=1)
                    

                    # === FIGURE 1 : AUC & Accuracy ===
                    fig1, (ax_auc, ax_acc) = plt.subplots(1, 2, figsize=(20, 7))
                    sns.barplot(data=results_df, x='Score_str', y='auc', hue='Model', ax=ax_auc)
                    ax_auc.set_title('Performance (AUC) par Seuil de Score')
                    ax_auc.set_ylabel('AUC Score')
                    ax_auc.set_xlabel('Seuil de score minimum (Nombre de features)')

                    sns.barplot(data=results_df, x='Score_str', y='accuracy', hue='Model', ax=ax_acc)
                    ax_acc.set_title('Performance (Accuracy) par Seuil de Score')
                    ax_acc.set_ylabel('Accuracy Score')
                    ax_acc.set_xlabel('Seuil de score minimum (Nombre de features)')

                    st.pyplot(fig1)

                    # === FIGURE 2 : Training Time & Inference Time ===
                    fig2, (ax_train, ax_inf) = plt.subplots(1, 2, figsize=(20, 7))
                    sns.barplot(data=results_df, x='Score_str', y='training_time', hue='Model', ax=ax_train)
                    ax_train.set_title("Temps d'entraînement par Seuil")
                    ax_train.set_ylabel('Temps d\'entraînement')
                    ax_train.set_xlabel('Seuil de score minimum (Nombre de features)')

                    sns.barplot(data=results_df, x='Score_str', y='inference_time', hue='Model', ax=ax_inf)
                    ax_inf.set_title("Temps d'inférence par Seuil")
                    ax_inf.set_ylabel('Temps d\'inférence')
                    ax_inf.set_xlabel('Seuil de score minimum (Nombre de features)')

                    st.pyplot(fig2)

                    # === FIGURE 3 : F1-Score ===
                    fig3, ax_f1 = plt.subplots(figsize=(10, 7))
                    sns.barplot(data=results_df, x='Score_str', y='f1', hue='Model', ax=ax_f1)
                    ax_f1.set_title("Performance (F1-Score) par Seuil de Score")
                    ax_f1.set_ylabel('F1-Score')
                    ax_f1.set_xlabel('Seuil de score minimum (Nombre de features)')

                    st.pyplot(fig3)



        with tab2:
            st.subheader("Sélection de Features (Méthodes Filtre & Wrapper)")
            # ... (This tab's code remains unchanged)

        with tab3:
            st.subheader("🌀 Extraction de Features (Projection)")
            # ... (This tab's code remains unchanged)

        with tab4:
            st.subheader("🎨 Apprentissage de Variétés (Manifold Learning pour la Visualisation)")
            # ... (This tab's code remains unchanged)

    else:
        st.warning("Veuillez charger des données à l'étape 1.")


# --- Page 6: AutoML Modeling ---
elif page == "6. Modélisation AutoML":
    if st.session_state.data is not None and st.session_state.target_column:
        automl_section(
            data=st.session_state.original_data,
            target_column=st.session_state.target_column,
            preprocessed_data=st.session_state.data
        )
    else:
        st.warning("Veuillez charger des données et sélectionner une colonne cible pour la modélisation.")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
        <p>Développé par SABRI Fayssal</p>
        <p>Thème: Optimisation des pipelines ML par réduction de dimension</p>
    </div>
    """,
    unsafe_allow_html=True
)
