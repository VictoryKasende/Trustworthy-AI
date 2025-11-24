# Instructions pour Démarrer le Projet

## 🚀 Démarrage Rapide

### 1. Collecter les Photos

Chaque membre doit collecter 20-30 photos de son visage et les placer dans :

- `data/raw/member1/` (pour le premier membre)
- `data/raw/member2/` (pour le deuxième membre)
- `data/raw/member3/` (pour le troisième membre)

### 2. Activer l'Environnement Virtuel

```bash
source .venv/bin/activate
```

### 3. Lancer Jupyter Notebook

```bash
jupyter notebook
```

Ou utiliser la tâche VS Code : `Ctrl+Shift+P` → "Tasks: Run Task" → "Lancer Jupyter Notebook"

### 4. Suivre les Notebooks dans l'Ordre

1. `01_data_exploration.ipynb` - Exploration des données
2. `02_model_training.ipynb` - Entraînement du modèle (à créer)
3. `03_federated_learning.ipynb` - Apprentissage fédéré (à créer)
4. `04_explainability_analysis.ipynb` - Techniques d'explicabilité (à créer)
5. `05_privacy_evaluation.ipynb` - Évaluation de la confidentialité (à créer)

## 📝 Notes Importantes

- **Photos** : Utilisez des photos de bonne qualité (min 224x224 pixels)
- **Diversité** : Variez les angles, éclairages, expressions
- **Confidentialité** : Les photos restent sur vos machines locales
- **Format** : JPG, JPEG ou PNG acceptés

## 🔧 Dépannage

### Erreur de packages manquants

```bash
.venv/bin/python -m pip install [package_name]
```

### Problème avec OpenCV

```bash
.venv/bin/python -m pip install opencv-python-headless
```

### Kernel Jupyter non trouvé

```bash
.venv/bin/python -m ipykernel install --user --name=trustworthy-ai
```
