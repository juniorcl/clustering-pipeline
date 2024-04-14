"""
This is a boilerplate pipeline 'music_recommendation'
generated using Kedro 0.19.3
"""
import re

from category_encoders import CountEncoder

from sklearn.cluster       import KMeans
from sklearn.compose       import ColumnTransformer
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def preprocessing_data(df):

    df.loc[:, 'artists_main'] = df['artists'].apply(lambda i: re.search(r"([\w|\s]+)", i).group(0))
    df.loc[:, 'artists_num'] = df['artists'].str.count("(['\w\s\d]+)+")

    df = df.loc[df['year'] >= 1950, :]

    return df


def train_model(df):

    df = df.drop(columns=['id', 'key', 'name', 'release_date', 'artists'])

    min_max_scaler_columns = [
        'valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 
        'instrumentalness', 'liveness', 'loudness', 'popularity', 'speechiness', 'tempo'
    ]
    
    onehot_encoder_columns = ['artists_main']

    ct = ColumnTransformer(
        transformers=[
            ('minmax_scaler', MinMaxScaler(), min_max_scaler_columns),
            ('count_encoder', CountEncoder(return_df=False, normalize=True), onehot_encoder_columns)
        ], 
        remainder='passthrough'
    )

    pipe = Pipeline(
        [
            ('column_transformer', ct),
            ('pca', PCA(n_components=2, random_state=42)),
            ('kmeans', KMeans(n_init='auto', n_clusters=4, random_state=42))
        ]
    )

    pipe.fit(df)

    return pipe
