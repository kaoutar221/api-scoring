from flask import Flask, request, jsonify, render_template_string
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
identifiant_exemples = df['SK_ID_CURR'].sample(4, random_state=1).astype(int).tolist()

# Créer l'application Flask
app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    exemples = ", ".join(map(str, identifiant_exemples))
    return render_template_string(f'''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Client Prediction</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }}
                .container {{
                    background-color: #fff;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                    text-align: center;
                }}
                h1 {{
                    margin-bottom: 20px;
                }}
                input[type="number"] {{
                    padding: 10px;
                    width: 80%;
                    margin-bottom: 10px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }}
                input[type="submit"] {{
                    padding: 10px 20px;
                    background-color: #28a745;
                    border: none;
                    border-radius: 4px;
                    color: #fff;
                    cursor: pointer;
                }}
                input[type="submit"]:hover {{
                    background-color: #218838;
                }}
                .result {{
                    margin-top: 20px;
                    padding: 10px;
                    background-color: #e9ecef;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Entrez l'identifiant du client</h1>
                <p>Voici quelques exemples d'identifiants pour tester : {exemples}</p>
                <form id="predictionForm">
                    <input type="number" name="client_id" id="client_id" required>
                    <input type="submit" value="Predict">
                </form>
                <div class="result" id="result"></div>
            </div>
            <script>
                document.getElementById('predictionForm').addEventListener('submit', function(event) {{
                    event.preventDefault();
                    const client_id = document.getElementById('client_id').value;
                    fetch('/predict', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/x-www-form-urlencoded',
                        }},
                        body: 'client_id=' + client_id
                    }})
                    .then(response => response.json())
                    .then(data => {{
                        if (data.error) {{
                            document.getElementById('result').innerHTML = '<p style="color: red;">' + data.error + '</p>';
                        }} else {{
                            document.getElementById('result').innerHTML = '<p>Client ID: ' + data.client_id + '</p>'
                                + '<p>Probability of Default: ' + data.probability_of_default + '</p>'
                                + '<p>Status: ' + data.status + '</p>';
                        }}
                    }})
                    .catch(error => {{
                        document.getElementById('result').innerHTML = '<p style="color: red;">Erreur: ' + error + '</p>';
                    }});
                }});
            </script>
        </body>
        </html>
    ''')


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
        logging.info(f"Données pour le client {client_id}: {client_data}")

        # Calculer la probabilité de défaut et la classe prédite
        probability = model.predict_proba(client_data)[:, 1][0]  # Probabilité de défaut
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
