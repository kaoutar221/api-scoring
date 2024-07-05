from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return "Application, model, and data have been successfully loaded."

@app.route('/retrieve_score/')
def retrieve_score():
    client_id = request.args.get('id')
    # Implémenter la logique pour récupérer le score basé sur client_id
    scores = {
        "200100": 5,
        "200105": 10,
        "300200": -5,
        "300300": 15,
        "300305": -10,
        "400400": 20
    }
    score = scores.get(client_id, 0)
    return jsonify(score=score)

@app.route('/retrieve_features/')
def retrieve_features():
    num = request.args.get('num')
    # Implémenter la logique pour récupérer les caractéristiques basées sur num
    features = {
        "2": ["TOTAL_CREDIT"],
        "4": ["TOTAL_CREDIT", "AGE_YEARS", "EDUCATION_LEVEL_Bachelor / bachelor special"],
        "6": ["TOTAL_CREDIT", "AGE_YEARS", "EDUCATION_LEVEL_Bachelor / bachelor special", "ANNUITY_TOTAL", "CITY_RATING_CUSTOMER"]
    }
    feature_list = features.get(num, [])
    return jsonify(features=feature_list)

if __name__ == "__main__":
    app.run(debug=True)