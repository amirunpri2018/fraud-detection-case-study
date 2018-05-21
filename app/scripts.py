import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

def read_to_df(data):
    ''' imput: data (JSON nested dictionary)
    '''
    import pandas as pd
    df_ = pd.DataFrame(dict([(k, pd.Series(data[k])) for k in data if (
        k != 'ticket_types') and (k != 'previous_payouts')]))
    df_['n_tickets'] = len(data['ticket_types'])
    df_['n_payouts'] = len(data['previous_payouts'])
    #df.reset_index(drop=1, inplace=True)
    return df_

def fill_nans(df):
    for col in ['currency']:
        df[col] = df[col].replace(np.nan, 'USD', regex=True)

    for col in ['country','venue_country', 'payout_type']:
        df[col] = df[col].replace(np.nan, '', regex=True)

    for col in ['delivery_method','has_header','sale_duration', 'venue_latitude', 'venue_longitude',\
'body_length', 'delivery_method', 'fb_published', 'gts', 'has_analytics', 'has_header', 'has_logo',\
'sale_duration', 'sale_duration2', 'show_map', 'user_age', 'user_created', 'n_payouts', 'name_length',\
'object_id']:
        df[col] = df[col].replace(np.nan, 0, regex=True)
    for col in ['listed']:
        df[col] = df[col].replace(np.nan, 'n', regex=True)
    for col in ['n_tickets', 'num_order', 'num_payouts']:
        df[col] = df[col].replace(np.nan, 1, regex=True)

    return df

def feat_engineer(df,encoders):

    # columns to remove
    ts_cols = ['approx_payout_date','event_created', 'event_end', 'event_published', 'event_start']
    topic_cols = ['org_desc','org_name','name','venue_name','description','email_domain']
    string_cols = ['payee_name','venue_address','venue_state']
    cat_cols = ['channels','user_type','org_facebook','org_twitter']

    # drop unwanted/ list/ cat columns
    cols_to_drop = ts_cols + topic_cols + string_cols + cat_cols
    df.drop(cols_to_drop, axis=1, inplace=True)
    ''' fix this!'''

    # fill nans with mode or ''
    df = fill_nans(df)

    # label encoding
    cols = ['currency','country','venue_country','payout_type','listed']
    for col, encoder in zip(cols, encoders):
        encoded = encoder.transform(df[col])
        df[col] = encoded

    return df

    #df.fillna(0)

# def categorical_encode(df):
#     ds = pickle.load(open('dicts.pkl','rb'))
#
#     df.currency = df.currency.apply(lambda x: ds[0]][x])
#     df.country = df.country.apply(lambda x: ds[1][x])
#     df.venue_country = df.venue_country.apply(lambda x: ds[2][x])
#     df.payout_type = df.payout_type.apply(lambda x: ds[3][x])
#     df.listed = df.listed.apply(lambda x: ds[4][x])
#
#     return df

def prep_data(df):
    # def x, y
    y = df.pop('FRAUD')
    X = df

    # oversample
    X_smoted, y_smoted = smote(X.values, y.values, 0.33, k=None)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X_smoted, y_smoted)

    return X_train, X_test, y_train, y_test
