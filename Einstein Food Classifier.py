import requests
import pandas as pd



def get_output_schema():
    return pd.DataFrame({
        'label':prep_string(),
        'probability':prep_decimal(),
        'source':prep_string()
    })

def get_tags(url):
        headers = {
            'Authorization': 'Bearer <insert token>',
            'Cache-Control': 'no-cache'
        }
        files = {
            'sampleLocation': url,
            'modelId': 'FoodImageClassifier'
        }
        response = requests.post(
            'https://api.einstein.ai/v2/vision/predict', 
            headers=headers, 
            files=files
        )
        return response.json().get('probabilities', [{'label': 'ERROR', 'probability': 1.0}])

def get_it(df):
    urls = df['source'].tolist()
    results = [{'source': u, 'probabilities': get_tags(u)} for u in urls]
    df = pd.json_normalize(results, 'probabilities', ['source'])
    return df
