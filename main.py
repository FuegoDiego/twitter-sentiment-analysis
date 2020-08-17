import pandas as pd

# load the data
training_df = pd.read_csv('./data/tweets_training_data.csv',
                          header=None,
                          names=['target', 'id', 'date', 'flag', 'user', 'text'],
                          usecols=['target', 'id', 'date', 'text'],
                          encoding='latin-1')

training_df = training_df.groupby('target').head(1000).reset_index(drop=True)
