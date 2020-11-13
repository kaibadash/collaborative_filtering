import pickle

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# based on https://www.codexa.net/collaborative-filtering-k-nearest-neighbor/
class Recommendation:
    MODEL_DATA = "model.data"

    def search_title(self, title):
        anime_pivot = self.generate_learning_data()
        return anime_pivot[anime_pivot.index.str.contains(title)].index[0:][0]

    def recommend(self, string):
        model_knn = self.load()
        title = self.search_title(string)
        print(title)
        anime_pivot = self.generate_learning_data()
        distance, index = model_knn.kneighbors(anime_pivot.iloc[anime_pivot.index == title].values.reshape(1, -1),
                                               n_neighbors=11)
        for i in range(0, len(distance.flatten())):
            if i == 0:
                print(
                    'Recommendations if you like the anime {0}:\n'.format(
                        anime_pivot[anime_pivot.index == title].index[0]))
            else:
                print('{0}: {1} ({2})'.format(i, anime_pivot.index[index.flatten()[i]], distance.flatten()[i]))

    # get csv from https://www.kaggle.com/CooperUnion/anime-recommendations-database
    def generate_learning_data(self):
        ratings = pd.read_csv('rating.csv')
        anime = pd.read_csv('anime.csv')
        anime = anime[anime['members'] > 10000]
        # 欠損データをdropna()でデータセットから取り除く
        anime = anime.dropna()
        # ratingの値が0以上のみ残す
        ratings = ratings[ratings.rating >= 0]

        # animeとratingsの2つのデータフレームをマージさせる
        merged = ratings.merge(anime, left_on='anime_id', right_on='anime_id', suffixes=['_user', ''])
        merged = merged[['user_id', 'name', 'rating_user']]
        merged = merged.drop_duplicates(['user_id', 'name'])

        # pivot. userごとのアニメ評価データを作成する
        anime_pivot = merged.pivot(index='name', columns='user_id', values='rating_user').fillna(0)
        return anime_pivot

    def learn(self):
        print("start leaning...")
        knn = NearestNeighbors(n_neighbors=9, algorithm='brute', metric='cosine')
        anime_pivot_sparse = csr_matrix(self.generate_learning_data().values)
        # 前処理したデータセットでモデルを訓練
        model = knn.fit(anime_pivot_sparse)
        with open(self.MODEL_DATA, "wb") as f:
            pickle.dump(model, f)

    def load(self):
        with open(self.MODEL_DATA, "rb") as f:
            return pickle.load(f)
