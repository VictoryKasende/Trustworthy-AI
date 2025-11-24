# Trustworthy AI - Projet de Classification Faciale

## 📋 Description du Projet

Projet d'éthique en Intelligence Artificielle développé par une équipe de 3 étudiants. L'objectif est de construire un modèle d'IA intégrant les caractéristiques d'un Trustworthy AI model avec :

- **Classification faciale** des 3 membres du groupe
- **Apprentissage fédéré** pour la protection de la confidentialité
- **Techniques d'IA explicable** (LIME, SHAP, Grad-CAM)
- **Sécurité et confidentialité** des données

## 🎯 Objectifs

1. Entraîner un modèle global avec Federated Learning
2. Appliquer des techniques d'Explainable AI
3. Garantir la protection de la confidentialité et sécurité
4. Maximiser la précision du modèle

## 🏗️ Structure du Projet

```
trustworthy-ai/
├── data/
│   ├── raw/                    # Photos brutes des 3 membres
│   │   ├── member1/
│   │   ├── member2/
│   │   └── member3/
│   ├── processed/              # Images préprocessées
│   └── federated/             # Données distribuées pour FL
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py   # Préparation des données
│   │   └── data_loader.py     # Chargement des données
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_model.py       # Architecture CNN
│   │   ├── federated_client.py# Client FL
│   │   └── federated_server.py# Serveur FL
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── lime_explainer.py  # LIME implementation
│   │   ├── shap_explainer.py  # SHAP implementation
│   │   └── gradcam_explainer.py# Grad-CAM implementation
│   ├── privacy/
│   │   ├── __init__.py
│   │   ├── differential_privacy.py
│   │   └── encryption.py      # Chiffrement des paramètres
│   └── utils/
│       ├── __init__.py
│       ├── config.py          # Configuration
│       ├── metrics.py         # Métriques d'évaluation
│       └── visualization.py   # Visualisations
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_federated_learning.ipynb
│   ├── 04_explainability_analysis.ipynb
│   └── 05_privacy_evaluation.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_explainability.py
│   └── test_privacy.py
├── docs/
│   ├── ethics_report.md       # Rapport éthique
│   ├── model_documentation.md
│   └── privacy_analysis.md
├── config/
│   ├── model_config.yaml
│   ├── federated_config.yaml
│   └── privacy_config.yaml
├── requirements.txt
├── setup.py
└── .gitignore
```

## 🛠️ Technologies Utilisées

- **Deep Learning**: TensorFlow/Keras, PyTorch
- **Federated Learning**: TensorFlow Federated (TFF)
- **Explainable AI**: LIME, SHAP, tf-explain
- **Privacy**: TensorFlow Privacy, PySyft
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly

## 📦 Installation

```bash
# Cloner le projet
git clone <repository-url>
cd trustworthy-ai

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 🚀 Utilisation

### 1. Préparation des Données

```bash
python src/data/preprocessing.py
```

### 2. Entraînement Fédéré

```bash
python src/models/federated_server.py
```

### 3. Analyse d'Explicabilité

```bash
jupyter notebook notebooks/04_explainability_analysis.ipynb
```

## 📊 Critères d'Évaluation

1. **Précision des Modèles**: Confiance vs hasard
2. **Protection de la Confidentialité**:
   - Division sécurisée des données
   - Chiffrement des paramètres
   - Agrégation sécurisée
3. **Explainabilité**: Application de techniques d'XAI
4. **Documentation Éthique**: Rapport complet

## 👥 Équipe

- Membre 1: [Nom]
- Membre 2: [Nom]
- Membre 3: [Nom]

## 📅 Timeline

- **Date de remise**: 20-11-2025
- **Type d'évaluation**: Moyenne et Examen

## 📄 License

Ce projet est développé dans un cadre académique pour le cours d'éthique en IA.
