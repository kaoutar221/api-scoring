import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

# Charger les données
file_path = 'data/sampled_df1.csv'
df = pd.read_csv(file_path)

# Séparer les features et la cible
X = df.drop(columns=['TARGET'])
y = df['TARGET']

# Définir le pipeline pour GradientBoostingClassifier
gb_pipeline = ImbPipeline([
    ('smote', SMOTE()),
    ('classifier', GradientBoostingClassifier())
])

# Définir les paramètres pour la recherche en grille pour GradientBoostingClassifier
gb_param_dist = {
    'classifier__learning_rate': [0.0001, 0.001, 0.005],  # Réduire le taux d'apprentissage
    'classifier__max_depth': [1, 2],  # Réduire la profondeur
    'classifier__n_estimators': [5, 10],  # Réduire le nombre d'estimators
    'classifier__min_samples_split': [50, 100],  # Augmenter le nombre minimum d'échantillons pour diviser un nœud
    'classifier__min_samples_leaf': [20, 30],  # Augmenter le nombre minimum d'échantillons dans une feuille
    'classifier__subsample': [0.1, 0.3]  # Réduire le sous-échantillonnage
}

# Définir les scorers
fbeta_scorer = make_scorer(fbeta_score, beta=10, average='macro')
roc_auc_scorer = 'roc_auc'

# Créer la première GridSearchCV pour F-beta
gb_grid_search_fbeta = GridSearchCV(estimator=gb_pipeline,
                                    param_grid=gb_param_dist,
                                    scoring=fbeta_scorer,
                                    refit=True,
                                    cv=5,
                                    n_jobs=-1,
                                    verbose=1)

# Créer la deuxième GridSearchCV pour AUC ROC
gb_grid_search_roc_auc = GridSearchCV(estimator=gb_pipeline,
                                      param_grid=gb_param_dist,
                                      scoring=roc_auc_scorer,
                                      refit=True,
                                      cv=5,
                                      n_jobs=-1,
                                      verbose=1)

# Adapter chaque GridSearchCV aux données d'entraînement
gb_grid_search_fbeta.fit(X, y)
gb_grid_search_roc_auc.fit(X, y)

# Récupérer les meilleurs paramètres et les meilleurs modèles
best_model_fbeta_gb = gb_grid_search_fbeta.best_estimator_
best_model_roc_auc_gb = gb_grid_search_roc_auc.best_estimator_

# Sauvegarder le meilleur modèle
model_path = 'model/best_model_fbeta_gb.pkl'
joblib.dump(best_model_fbeta_gb, model_path)

# Sauvegarder les noms des caractéristiques
feature_names_path = 'model/feature_names.txt'
with open(feature_names_path, 'w') as f:
    for feature in X.columns:
        f.write(f"{feature}\n")

