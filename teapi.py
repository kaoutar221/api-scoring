import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    'EXT_SOURCE_3': 0.5,
    'EXT_SOURCE_2': 0.5,
    'EXT_SOURCE_1': 0.5,
    'CC_CNT_DRAWINGS_ATM_CURRENT_MEAN': 0,
    'CC_CNT_DRAWINGS_CURRENT_MAX': 0,
    'CC_CNT_DRAWINGS_CURRENT_MEAN': 0,
    'BURO_DAYS_CREDIT_MEAN': -1000,
    'CC_AMT_BALANCE_MEAN': 0,
    'CC_AMT_TOTAL_RECEIVABLE_MEAN': 0,
    'CC_COUNT': 0,
    'REFUSED_COUNT': 0,
    'DAYS_BIRTH': -12000,
    'PREV_NAME_CONTRACT_STATUS_Refused_MEAN': 0.1,
    'BURO_CREDIT_ACTIVE_Closed_MEAN': 0.2,
    'DAYS_EMPLOYED': -2000,
    'BURO_CREDIT_ACTIVE_Active_MEAN': 0.3,
    'PREV_CODE_REJECT_REASON_XAP_MEAN': 0.4,
    'BURO_DAYS_CREDIT_MIN': -1500,
    'BURO_DAYS_CREDIT_UPDATE_MEAN': -500,
    'DAYS_EMPLOYED_PERC': 0.6,
    'PREV_NAME_CONTRACT_STATUS_Approved_MEAN': 0.7,
    'CLOSED_DAYS_CREDIT_MIN': -1000,
    'ACTIVE_DAYS_CREDIT_MEAN': -800
}

response = requests.post(url, json=data)

# Vérifiez le code de statut de la réponse
if response.status_code == 200:
    print(response.json())
else:
    print(f"Erreur: {response.status_code}")
    print(response.text)
