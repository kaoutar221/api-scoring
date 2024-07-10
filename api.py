from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import logging
import os
from load_data import load_data

# Configurer le logging
logging.basicConfig(level=logging.INFO)

# Charger le modèle
model_path = 'model/best_model_fbeta_gb.pkl'
model = joblib.load(model_path)

# Charger les noms des fonctionnalités
feature_names_path = 'model/feature_names.txt'
with open(feature_names_path, 'r') as f:
    feature_names = [line.strip() for line in f]

# Charger les données
data_path = 'data/sampled_df1 (1).csv'  # Utilisez un chemin relatif
df = load_data(data_path)

# Vérifiez que les données sont chargées correctement
if df is None:
    raise FileNotFoundError(f"Le fichier {data_path} est introuvable ou ne peut pas être chargé.")

# Obtenir quelques exemples d'identifiants
identifiant_exemples = df['SK_ID_CURR'].sample(4, random_state=60).astype(int).tolist()

# Créer l'application Flask
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    exemples = ", ".join(map(str, identifiant_exemples))
    return render_template('home.html', exemples=exemples)


@app.route('/predict', methods=['POST'])
def predict():
    client_id = request.form.get('client_id')

    if not client_id:
        return jsonify({'error': 'Client ID is required'}), 400

    try:
        client_id = int(client_id)
        logging.info(f"Requête reçue pour le client {client_id}")

        # Extraire les données du client
        if client_id not in df['SK_ID_CURR'].values:
            logging.error(f"Client ID {client_id} n'est pas dans le dataset")
            return jsonify({'error': f"Client ID {client_id} n'est pas dans le dataset"}), 400

        client_data = df[df['SK_ID_CURR'] == client_id][feature_names]
        logging.info(f"Données pour le client {client_id}: {client_data.to_dict(orient='records')}")

        # Vérifier que les données du client ne sont pas vides
        if client_data.empty:
            logging.error(f"Données du client {client_id} sont vides")
            return jsonify({'error': f"Données du client {client_id} sont vides"}), 400

        # Calculer la probabilité de défaut et la classe prédite
        probability = model.predict_proba(client_data)[:, 1][0]  # Probabilité de défaut
        logging.info(f"Probabilité de défaut pour le client {client_id}: {probability}")

        threshold = 0.45  # Vous pouvez ajuster ce seuil selon vos besoins
        prediction = int((probability >= threshold).astype(int))  # Convertir en type sérialisable
        status = 'accepté' if prediction == 0 else 'refusé'  # Utiliser la syntaxe ternaire correcte

        # Préparer la réponse
        result = {
            'client_id': client_id,
            'probability_of_default': float(probability),  # Convertir en type sérialisable
            'status': status
        }

        logging.info(f"Résultat pour le client {client_id}: {result}")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Erreur lors du traitement de la requête pour le client {client_id}: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
