from flask import Flask, render_template
import json
import requests
import time
from datetime import datetime
import psycopg2
import pickle
from scripts import read_to_df, feat_engineer, prep_data

rfc = pickle.load(open('rfc.p','rb'))
encoders = pickle.load(open('encoders.pkl','rb'))

app = Flask(__name__)
PORT = 80
DATA = []
TIMESTAMP = []


def get_datapoint():
    r = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
    DATA.append(r.json())
    #DATA.append(json.dumps(r.json(), sort_keys=True, indent=4, separators=(',', ': ')))
    TIMESTAMP.append(time.time())

def transform(data, encoders):
    df = read_to_df(data)
    df_trans = feat_engineer(df, encoders)
    return df_trans

def predict(df, estimator):
    # train model and predict
    y_hat = estimator.predict(df.values)
    y_prob = estimator.predict_proba(df.values)

    return y_hat, y_prob

''' flask app'''

@app.route('/')
def home_page():
    return '''
        <center>
            <h1>
            Fraud testing
            </h1>
        </center>
        <center>
            <a href = '/dashboard'> Predict Fraud </a>
        </center>
        '''

# prediction app
@app.route('/dashboard')
def dashboard():
    get_datapoint()
    df = transform(DATA[-1], encoders)
    y_hat, y_prob = predict(df, rfc)
    threshold = 0.2
    pred = 'Fraud' if y_prob[0][1] > threshold else 'Not fraud'

    # page:
    page = '''
            <center>
                {}
            </center>

            <center>
                Prediction: {}
            </center>
            <center>
                Risk of fraud: {} %
            </center>
            '''
    return page.format(df.head().to_html(), str(pred), str(y_prob[0][1]*100))

if __name__ == '__main__':
    # load fitted model and fitted encoders
    rfc = pickle.load(open('rfc.p','rb'))
    encoders = pickle.load(open('encoders.pkl','rb'))
    app.run(host='0.0.0.0', port=8080, debug=True)
