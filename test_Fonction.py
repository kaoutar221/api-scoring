import pytest
import json
from Fonction import app

def test_homepage():
    with app.test_client() as client:
        response = client.get('/')
        assert response.data.decode('utf-8') == "Application, model, and data have been successfully loaded."

# Test data for ID scoring
id_test_data = [
    ("200100", 5),
    ("200105", 10),
    ("300200", -5),
]

@pytest.mark.parametrize("client_id, expected_score", id_test_data)
def test_retrieve_score(client_id, expected_score):
    with app.test_client() as client:
        response = client.get(f'/retrieve_score/?id={client_id}')
        result = json.loads(response.data.decode('utf-8')).get("score")
        assert result == expected_score

# Test data for feature retrieval
feature_test_data = [
    ("2", ["TOTAL_CREDIT"]),
    ("4", ["TOTAL_CREDIT", "AGE_YEARS", "EDUCATION_LEVEL_Bachelor / bachelor special"]),
    ("6", ["TOTAL_CREDIT", "AGE_YEARS", "EDUCATION_LEVEL_Bachelor / bachelor special", "ANNUITY_TOTAL", "CITY_RATING_CUSTOMER"]),
]

@pytest.mark.parametrize("num_features, expected_features", feature_test_data)
def test_retrieve_features(num_features, expected_features):
    with app.test_client() as client:
        response = client.get(f'/retrieve_features/?num={num_features}')
        result = json.loads(response.data.decode('utf-8')).get("features")
        assert result == expected_features

# Additional test data for more coverage
additional_test_data = [
    ("300300", 15),
    ("300305", -10),
    ("400400", 20),
]

@pytest.mark.parametrize("client_id, expected_score", additional_test_data)
def test_additional_scores(client_id, expected_score):
    with app.test_client() as client:
        response = client.get(f'/retrieve_score/?id={client_id}')
        result = json.loads(response.data.decode('utf-8')).get("score")
        assert result == expected_score
