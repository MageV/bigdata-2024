import warnings
import pandas as pd
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")
from scipy.sparse import csr_matrix

movies = pd.read_csv('/home/master/Documents/bd/movies.csv')
ratings = pd.read_csv('/home/master/Documents/bd/ratings.csv')
movies.drop(['genres'], axis=1, inplace=True)
ratings.drop(['timestamp'], axis=1, inplace=True)
usex_movies_matrix = ratings.pivot(index='movieId', columns='userId', values='rating')
usex_movies_matrix.fillna(0, inplace=True)
user_rates = ratings.groupby('userId')['rating'].agg('count')
movie_rates = ratings.groupby('movieId')['rating'].agg('count')
user_filter = user_rates[user_rates > 50].index
movie_filter = movie_rates[movie_rates > 10].index
usex_movies_matrix = usex_movies_matrix.loc[:, user_filter]
usex_movies_matrix = usex_movies_matrix.loc[movie_filter, :]
csr_data = csr_matrix(usex_movies_matrix.values)
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)
criteria = movies[movies['title'].str.contains('Before')]['movieId'].index[0]
var = usex_movies_matrix.iloc[criteria]
dist, idx = knn.kneighbors(csr_data[criteria], n_neighbors=10)
dist = list(*dist)
idx = list(*idx)
zipped = list(zip(idx, map(lambda x: round(x, 6), dist)))
recom = sorted(zipped, key=min)[1:]
for item in recom:
    movid = usex_movies_matrix.iloc[item[0]]['movieId']
    title = movies[movies['movieId'] == movid]['title'].values[0]
    print(title)
