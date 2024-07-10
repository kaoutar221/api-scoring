# API Scoring Project

## Objectif du Projet

L'objectif principal de cette mission est de concevoir et de mettre en place un modèle de scoring(API) crédit efficace. Ce modèle devra être capable de prédire avec précision la probabilité de remboursement des clients et de déterminer si une demande de crédit doit être acceptée ou refusée.

## Structure des Dossiers

La structure du projet est organisée comme suit :

- **.github**
  - **workflows**
    - `test.yml` : Fichier de configuration pour les workflows GitHub Actions, permettant l'automatisation des tests.

- **data**
  - `sampled_df1.csv` : Exemple de données utilisées pour tester et entraîner le modèle.

- **model**
  - `best_model_fbeta_gb.pkl` : Modèle de machine learning pré-entraîné, sauvegardé au format pickle.
  - `feature_names.txt` : Fichier contenant les noms des features utilisés par le modèle.

- **api.py** : Script principal de l'API qui gère les requêtes et les réponses.
- **Fonction.py** : Fichier contenant des fonctions auxiliaires utilisées dans le projet.
- **load_data.py** : Script pour charger et prétraiter les données.
- **Procfile** : Fichier utilisé par Heroku pour exécuter l'application.
- **README.md** : Documentation principale du projet, contenant des informations sur l'installation, l'utilisation et la contribution.
- **requirements.txt** : Liste des dépendances Python nécessaires pour exécuter le projet.
- **test_app.py** : Script de test pour vérifier le bon fonctionnement de l'API.
- **train_model.py** : Script utilisé pour entraîner le modèle de machine learning.

## Guide de Démarrage

### Prérequis

- Python 3.x
- Pip

### Installation

1. Clonez le dépôt :
    ```bash
    git clone <url-du-repo>
    cd api-scoring
    ```

2. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

3. Lancez l'application :
    ```bash
    python api.py
    ```

## Utilisation

Pour utiliser l'API, envoyez une requête POST à l'endpoint approprié avec les données nécessaires. Consultez `README.md` pour des exemples de requêtes et de réponses.

---

## Contribuer

Les contributions sont les bienvenues ! Pour des changements majeurs, veuillez d'abord ouvrir une issue pour discuter de ce que vous aimeriez changer.

---

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
