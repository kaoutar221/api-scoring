from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging

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
data_path = 'data/sampled_df1.csv'
df = pd.read_csv(data_path)

# Créer l'application Flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "API is running. Send a GET request to /predict/<client_id> to get predictions."

@app.route('/predict/<int:client_id>', methods=['GET'])
def predict(client_id):
    logging.info(f"Requête reçue pour le client {client_id}")

    try:
        # Extraire les données du client
        if client_id not in df.index:
            logging.error(f"Client ID {client_id} n'est pas dans le dataset")
            return jsonify({'error': f"Client ID {client_id} n'est pas dans le dataset"}), 400

        client_data = df.loc[client_id, feature_names].to_frame().T
        logging.info(f"Données pour le client {client_id}: {client_data}")

        # Calculer la probabilité de défaut et la classe prédite
        probability = model.predict_proba(client_data)[:, 1][0]  # Probabilité de défaut
        threshold = 0.5  # Vous pouvez ajuster ce seuil selon vos besoins
        prediction = int((probability >= threshold).astype(int))  # Convertir en type sérialisable
        status = 'accepté' if prediction == 0 else 'refusé'

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
