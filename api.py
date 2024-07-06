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
with open('model/feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f]

# Créer l'application Flask
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return "API is running. Send a POST request to /predict to get predictions."


@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Requête reçue pour /predict")

    try:
        # Obtenir les données du client depuis la requête POST
        data = request.json
        logging.info(f"Données reçues: {data}")

        # Convertir les données en DataFrame
        input_data = pd.DataFrame(data, index=[0])
        logging.info(f"Données converties en DataFrame: {input_data}")

        # Vérifier les colonnes manquantes
        missing_cols = set(feature_names) - set(input_data.columns)
        if missing_cols:
            logging.error(f"Colonnes manquantes: {missing_cols}")
            return jsonify({'error': f"Les colonnes suivantes sont manquantes: {missing_cols}"}), 400

        # Réordonner les colonnes
        input_data = input_data[feature_names]
        logging.info(f"Données réordonnées en DataFrame: {input_data}")

        # Calculer la probabilité de défaut et la classe prédite
        probability = model.predict_proba(input_data)[:, 1][0]  # Probabilité de défaut
        threshold = 0.5  # Vous pouvez ajuster ce seuil selon vos besoins
        prediction = int((probability >= threshold).astype(int))  # Convertir en type sérialisable

        # Préparer la réponse
        result = {
            'probability_of_default': float(probability),  # Convertir en type sérialisable
            'prediction': prediction
        }

        logging.info(f"Résultat: {result}")
        return jsonify(result)
    except Exception as e:
        logging.error(f"Erreur lors du traitement de la requête: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

