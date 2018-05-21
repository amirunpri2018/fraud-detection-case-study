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

def read_entry(example_path):
    '''
    Read single entry from http://galvanize-case-study-on-fraud.herokuapp.com/data_point
    '''
    with open(example_path) as data_file:
        d = json.load(data_file)
    df = pd.DataFrame()
    df_ = pd.DataFrame(dict([(k, pd.Series(d[k])) for k in d if (
        k != 'ticket_types') and (k != 'previous_payouts')]))
    df_['ticket_types'] = str(d['ticket_types'])
    df_['previous_payouts_total'] = len(d['previous_payouts'])
    df = df.append(df_)
    df.reset_index(drop=1, inplace=1)
    example = df
    return df

def transform(data, encoders):
    df = read_to_df(data)
    df_trans = feat_engineer(df, encoders)
    return df_trans

def predict(df, estimator):
    # name feature, oversample, split

    # train model and predict
    y_hat = estimator.predict(df.values)

    return y_hat


''' flask app'''
# home page
# @app.route("/")
# def index():
#     return render_template('index.html')


@app.route('/')
def home_page():
    return '''
        <center>
            <h1>
            Fraud testing
            </h1>
        </center>
        <center>
            <a href = '/prediction'> Predict Fraud </a>
        </center>
        '''
# submit page
@app.route('/submit')
def submit():
    return '''
        <center>
        <form action="/prediction" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        </center>
        '''
# prediction app
@app.route('/prediction', methods=['POST'] )
def prediction():
    # load fitted model and fitted encoders
    rfc = pickle.load(open('rfc.p','rb'))
    encoders = pickle.load(open('encoders.pkl','rb'))
    # get data from source, transform and predict
    get_datapoint()
    df = transform(DATA[0], encoders)
    y_hat = predict(df, rfc)
    # page:
    page = 'Prediction: {}'
    return page.format(str(y_hat))

if __name__ == '__main__':
    get_datapoint()
    df = transform(DATA[0], encoders)
    y_hat = predict(df, rfc)
    print(y_hat)
    app.run(host='0.0.0.0', port=8080, debug=True)
