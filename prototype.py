import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap.umap_ as umap
import missingno as msno
import io
import base64
import h2o
from h2o.automl import H2OAutoML
import time

# --- Assume these local modules are in the project structure ---
# from ollamaInterface import OllamaInterface
# from main import load_local_dataset
# from llm_interface import LLM_Interface
# from feature_selector import FeatureSelector
# from evaluation import evaluate_feature_subset

# --- Placeholder classes for demonstration since the actual files are not available ---
class OllamaInterface:
    def __init__(self, model="llama3.2"):
        self.model = model
        print(f"OllamaInterface initialized with model: {self.model}")

    def get_response(self, prompt):
        # Simulate a real LLM call
        print(f"--- PROMPT SENT TO OLLAMA ---\n{prompt}\n--------------------------")
        time.sleep(3) # Simulate network latency
        # This is a mocked response. A real implementation would parse the LLM's output.
        if "mean radius" in prompt:
             return "{'mean radius': 0.9, 'mean texture': 0.7, 'mean perimeter': 0.95, 'mean area': 0.92, 'mean smoothness': 0.5}"
        else:
             return "{'feature1': 0.8, 'feature2': 0.6}"


class FeatureSelector:
    def __init__(self, llm_interface):
        self.llm_interface = llm_interface

    def get_scores(self, concepts, task_description):
        prompt = f"""
        Given the task: '{task_description}', evaluate the importance of the following features (concepts): {concepts}.
        Provide a score from 0.0 to 1.0 for each feature.
        Return the result as a Python dictionary string.
        """
        response_str = self.llm_interface.get_response(prompt)
        try:
            # The response is a string representation of a dictionary
            return eval(response_str)
        except:
            return {"error": "Failed to parse LLM response"}

# --- H2O AutoML Section ---
def automl_section(data, target_column, preprocessed_data=None):
    data_to_use = preprocessed_data if preprocessed_data is not None else data
    st.header("Modélisation Automatique avec H2O")
    # ... (rest of the automl_section code remains the same)

# --- Main App Configuration ---
st.set_page_config(page_title='Auto-Prep: Prétraitement & Réduction de Dimension', layout='wide')

# --- Utility Functions ---
def detect_data_types(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_cols, categorical_cols

# --- Main App Interface ---
st.title('Auto-Prep: Prétraitement Automatique & Réduction de Dimension')
st.markdown("""
Cette application a été améliorée pour se concentrer sur la **réduction de dimension**, un aspect clé de l'optimisation des pipelines de Machine Learning, maintenant avec une **assistance LLM intégrée**.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Étapes:",
    ["1. Chargement des données",
     "2. Exploration des données",
     "3. Gestion des valeurs manquantes",
     "4. Encodage des variables catégorielles",
     "5. Réduction de Dimension & Sélection de Features",
     "6. Évaluation et exportation",
     "7. Modélisation AutoML"])

# Initialize session_state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# --- Page 1: Data Loading ---
if page == "1. Chargement des données":
    st.header("1. Chargement de vos données")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV ou Excel", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
            st.session_state.data = df.copy()
            st.session_state.original_data = df.copy()
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
        st.success("Données de démonstration (Cancer) chargées!")
        st.rerun()


# --- Other Pages (2, 3, 4) remain the same ---
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


# --- Page 5: Dimensionality Reduction (ENHANCED with real LLM call) ---
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

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Mise à l'échelle",
            "Sélection de Features (Classique)",
            "Extraction de Features (Projection)",
            "Visualisation de Variétés",
            "✨ Sélection de Features par LLM"
        ])

        # Tabs 1, 2, 3, 4 remain largely the same
        with tab1:
            st.subheader("Mise à l'échelle des variables (Feature Scaling)")
            st.info("La mise à l'échelle est souvent un prérequis pour les méthodes de réduction de dimension comme PCA.")
            # ... (Scaling logic would go here)

        with tab2:
            st.subheader("Sélection de Features (Méthodes Filtre & Wrapper)")
            # ... (SelectKBest and RFE logic remains here)

        with tab3:
            st.subheader("Extraction de Features (Projection)")
            # ... (PCA and LDA logic remains here)

        with tab4:
            st.subheader("Apprentissage de Variétés (Manifold Learning pour la Visualisation)")
            # ... (t-SNE and UMAP logic remains here)


        with tab5:
            st.subheader("✨ Sélection de Features Assistée par LLM")
            st.info("Utilisez un modèle de langage local (via Ollama) pour évaluer la pertinence de vos features.")

            if not target:
                st.warning("Veuillez sélectionner une colonne cible à l'étape 1 pour utiliser cette fonctionnalité.")
                st.stop()

            task_description = st.text_input(
                "Décrivez la tâche de prédiction:",
                f"Prédire si la valeur de '{target}' est élevée ou faible."
            )

            llm_name = st.text_input("Nom du modèle Ollama à utiliser:", "llama3.2")

            if st.button("Lancer l'analyse des features par le LLM"):
                with st.spinner(f"Le modèle '{llm_name}' analyse vos features..."):
                    try:
                        # 1. Initialize the interface to the local LLM
                        ollama_llm = OllamaInterface(model=llm_name)
                        selector = FeatureSelector(llm_interface=ollama_llm)

                        # 2. Get feature scores from the LLM
                        concepts = features.columns.tolist()
                        feature_scores = selector.get_scores(concepts=concepts, task_description=task_description)

                        if "error" in feature_scores:
                             st.error(f"Erreur du LLM: {feature_scores['error']}")
                        else:
                            st.success("Analyse terminée!")
                            scores_df = pd.DataFrame(list(feature_scores.items()), columns=['Feature', 'Score']).sort_values('Score', ascending=False)

                            # 3. Display results
                            st.write("Scores de pertinence des features selon le LLM:")
                            st.dataframe(scores_df)

                            fig, ax = plt.subplots()
                            sns.barplot(x='Score', y='Feature', data=scores_df, ax=ax, orient='h')
                            ax.set_title("Pertinence des Features (Scores LLM)")
                            st.pyplot(fig)

                            # 4. Allow user to select features based on score
                            st.markdown("---")
                            st.write("Sélectionnez les features à conserver en fonction des scores.")
                            score_threshold = st.slider("Seuil de score minimum:", 0.0, 1.0, 0.5, 0.05)
                            selected_by_llm = scores_df[scores_df['Score'] >= score_threshold]['Feature'].tolist()

                            st.write(f"**{len(selected_by_llm)} features sélectionnées avec un score >= {score_threshold}:**")
                            st.write(selected_by_llm)

                            if st.button("Appliquer cette sélection de features"):
                                st.session_state.data = df[selected_by_llm + [target]]
                                st.success("Le jeu de données a été mis à jour avec les features sélectionnées par le LLM.")
                                st.rerun()

                    except Exception as e:
                        st.error(f"Une erreur est survenue lors de la communication avec le LLM: {e}")
                        st.info("Assurez-vous que le service Ollama est en cours d'exécution et que le modèle spécifié est disponible.")

    else:
        st.warning("Veuillez charger des données à l'étape 1.")


# --- Page 6 & 7 remain the same ---
elif page == "6. Évaluation et exportation":
    if st.session_state.data is not None:
        st.header("6. Évaluation et Exportation")
        st.subheader("Données actuelles")
        st.dataframe(st.session_state.data.head())
    else:
        st.warning("Veuillez charger des données à l'étape 1.")

elif page == "7. Modélisation AutoML":
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
        <p>Développé par SABRI Fayssal pour son TFE</p>
        <p>Thème: Optimisation des pipelines ML par réduction de dimension</p>
    </div>
    """,
    unsafe_allow_html=True
)
