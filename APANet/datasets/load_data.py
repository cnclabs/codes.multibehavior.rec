import pandas as pd
import random

def load_df(dataset):
    if dataset == 'Kkbox':
        sess_, index_, item_, act_ = 'msno', 'index', 'song_id', 'source_system_tab'
        index_gap = 50

        train_dataset = pd.read_csv('raw/'+dataset+'/train.csv')
        test_dataset = pd.read_csv('raw/'+dataset+'/test.csv')

        del train_dataset['target']
        del test_dataset['id']
        df = pd.concat([train_dataset, test_dataset])
        df.dropna(inplace=True)
        df['index'] = range(len(df))

        test_samples = random.choices(list(set(list(df['msno']))), k=int(len(set(list(df['msno'])))*0.1))
        train_dataset = df[~df['msno'].isin(test_samples)].copy()
        test_dataset = df[df['msno'].isin(test_samples)].copy()
            

    if dataset == 'yoochoose':
        sess_, index_, item_, act_ = 'sess_id', 'timestamp', 'item_id', 'action'
        index_gap = None

        yo_df = pd.read_csv('raw/'+ dataset + '/yoochoose-clicks.dat', names=['sess_id', 'timestamp', 'item_id', 'cat_id'], dtype={'cat_id': 'str'}, header=None)
        yo_df['action'] = 'click'
        yo_df_buy = pd.read_csv('raw/' + dataset + '/yoochoose-buys.dat', names=['sess_id', 'timestamp', 'item_id', 'price', 'quantity'], header=None)
        yo_df_buy['action'] = 'buy'
        
        df = pd.concat([yo_df[['sess_id', 'timestamp', 'item_id', 'action']], yo_df_buy[['sess_id', 'timestamp', 'item_id', 'action']]])
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%dT%H:%M:%S.%f').values.tolist()
        df[['sess_id', 'item_id']] = df[['sess_id', 'item_id']].astype(str) 
        
        df = df.sort_values(by='timestamp')[:int(len(df)*(1/64))]
        train_dataset = df[:int(len(df)*0.9)].copy()
        test_dataset = df[int(len(df)*0.9):].copy()
    
    if dataset == 'Rrocket' or dataset == 'sample':
        sess_, index_, item_, act_ = 'visitorid', 'timestamp', 'itemid', 'event'
        index_gap = 500

        df = pd.read_csv('raw/'+dataset+'/events.csv')

        test_samples = random.choices(list(df['visitorid']), k=int(len(df['visitorid'])*0.1))
        train_dataset = df[~df['visitorid'].isin(test_samples)].copy()
        test_dataset = df[df['visitorid'].isin(test_samples)].copy()
        
        train_dataset['visitorid'] = train_dataset['visitorid'].astype(str) 
        test_dataset['visitorid'] = test_dataset['visitorid'].astype(str) 

    return sess_, index_, item_, act_, index_gap, train_dataset, test_dataset

