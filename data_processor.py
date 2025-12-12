import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import os
import warnings
warnings.filterwarnings('ignore')

class MovieLensDataset(Dataset):
    """自定义数据集类"""
    
    def __init__(self, users, movies, ratings, user_features, movie_features):
        self.users = users
        self.movies = movies
        self.ratings = ratings
        self.user_features = user_features
        self.movie_features = movie_features
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return {
            'user_id': torch.tensor(self.users[idx], dtype=torch.long),
            'movie_id': torch.tensor(self.movies[idx], dtype=torch.long),
            'rating': torch.tensor(self.ratings[idx], dtype=torch.float),
            'user_features': torch.tensor(self.user_features[idx], dtype=torch.float),
            'movie_features': torch.tensor(self.movie_features[idx], dtype=torch.float)
        }

class DataProcessor:
    """数据预处理类"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.feature_scaler = MinMaxScaler()
        
        # 初始化空属性
        self.ratings_df = None
        self.users_df = None
        self.movies_df = None
        self.user_features_df = None
        self.movie_features_df = None
        
    def load_data(self):
        """从文件夹加载MovieLens数据集"""
        try:
            # 检查数据文件是否存在
            if not os.path.exists(self.data_path):
                raise FileNotFoundError(f"数据路径不存在: {self.data_path}")
            
            # 尝试不同的文件格式
            self._load_ratings()
            self._load_users()
            self._load_movies()
            
            print("数据加载成功!")
            print(f"用户数: {len(self.users_df)}")
            print(f"电影数: {len(self.movies_df)}")
            print(f"评分记录数: {len(self.ratings_df)}")
            
            return True
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def _load_ratings(self):
        """加载评分数据"""
        ratings_files = ['ratings.dat', 'ratings.csv', 'ratings.txt']
        for file in ratings_files:
            file_path = os.path.join(self.data_path, file)
            if os.path.exists(file_path):
                if file.endswith('.dat'):
                    self.ratings_df = pd.read_csv(file_path, sep='::', 
                                                engine='python', 
                                                names=['user_id', 'movie_id', 'rating', 'timestamp'])
                else:
                    self.ratings_df = pd.read_csv(file_path)
                break
        else:
            raise FileNotFoundError("未找到评分数据文件")
    
    def _load_users(self):
        """加载用户数据"""
        users_files = ['users.dat', 'users.csv', 'users.txt']
        for file in users_files:
            file_path = os.path.join(self.data_path, file)
            if os.path.exists(file_path):
                if file.endswith('.dat'):
                    self.users_df = pd.read_csv(file_path, sep='::', 
                                              engine='python', 
                                              names=['user_id', 'gender', 'age', 'occupation', 'zipcode'])
                else:
                    self.users_df = pd.read_csv(file_path)
                break
        else:
            # 如果没有用户文件，从评分数据中提取用户ID
            user_ids = self.ratings_df['user_id'].unique()
            self.users_df = pd.DataFrame({'user_id': user_ids})
    
    def _load_movies(self):
        """加载电影数据"""
        movies_files = ['movies.dat', 'movies.csv', 'movies.txt']
        for file in movies_files:
            file_path = os.path.join(self.data_path, file)
            if os.path.exists(file_path):
                if file.endswith('.dat'):
                    self.movies_df = pd.read_csv(file_path, sep='::', 
                                               engine='python', 
                                               names=['movie_id', 'title', 'genres'],
                                               encoding='latin-1')
                else:
                    self.movies_df = pd.read_csv(file_path)
                break
        else:
            # 如果没有电影文件，从评分数据中提取电影ID
            movie_ids = self.ratings_df['movie_id'].unique()
            self.movies_df = pd.DataFrame({'movie_id': movie_ids})
    
    def preprocess_data(self, min_ratings_per_user=5, min_ratings_per_movie=10):
        """数据预处理"""
        print("开始数据预处理...")
        
        # 过滤稀疏数据
        user_rating_counts = self.ratings_df['user_id'].value_counts()
        movie_rating_counts = self.ratings_df['movie_id'].value_counts()
        
        active_users = user_rating_counts[user_rating_counts >= min_ratings_per_user].index
        popular_movies = movie_rating_counts[movie_rating_counts >= min_ratings_per_movie].index
        
        self.ratings_df = self.ratings_df[
            self.ratings_df['user_id'].isin(active_users) & 
            self.ratings_df['movie_id'].isin(popular_movies)
        ]
        
        # 编码用户和电影ID
        self.ratings_df['user_id_encoded'] = self.user_encoder.fit_transform(self.ratings_df['user_id'])
        self.ratings_df['movie_id_encoded'] = self.movie_encoder.fit_transform(self.ratings_df['movie_id'])
        
        # 创建用户特征
        self._create_user_features()
        
        # 创建电影特征
        self._create_movie_features()
        
        print(f"预处理后数据: {len(self.ratings_df)} 条评分记录")
    
    def _create_user_features(self):
        """创建用户特征"""
        # 用户基本特征
        if 'age' in self.users_df.columns:
            age_scaled = (self.users_df['age'] - self.users_df['age'].min()) / (self.users_df['age'].max() - self.users_df['age'].min())
        else:
            age_scaled = pd.Series([0.5] * len(self.users_df))
        
        if 'gender' in self.users_df.columns:
            gender_encoded = pd.get_dummies(self.users_df['gender'], prefix='gender')
        else:
            gender_encoded = pd.DataFrame({'gender_0': [1] * len(self.users_df)})
        
        if 'occupation' in self.users_df.columns:
            occupation_encoded = pd.get_dummies(self.users_df['occupation'], prefix='occupation')
        else:
            occupation_encoded = pd.DataFrame({'occupation_0': [1] * len(self.users_df)})
        
        # 用户行为特征
        user_stats = self.ratings_df.groupby('user_id_encoded').agg({
            'rating': ['mean', 'std', 'count']
        }).fillna(0)
        user_stats.columns = ['rating_mean', 'rating_std', 'rating_count']
        
        # 合并所有用户特征
        user_features_list = []
        for user_id_encoded in range(len(self.user_encoder.classes_)):
            original_user_id = self.user_encoder.inverse_transform([user_id_encoded])[0]
            user_data = self.users_df[self.users_df['user_id'] == original_user_id].iloc[0]
            
            features = []
            # 年龄
            if 'age' in user_data:
                features.append(age_scaled[self.users_df['user_id'] == original_user_id].values[0])
            else:
                features.append(0.5)
            
            # 性别 (one-hot)
            if 'gender' in user_data:
                gender_features = gender_encoded[self.users_df['user_id'] == original_user_id].values[0]
                features.extend(gender_features)
            else:
                features.extend([1, 0])  # 默认值
            
            # 职业 (one-hot)
            if 'occupation' in user_data:
                occupation_features = occupation_encoded[self.users_df['user_id'] == original_user_id].values[0]
                features.extend(occupation_features)
            else:
                features.extend([1] + [0] * (len(occupation_encoded.columns) - 1))
            
            # 评分统计
            if user_id_encoded in user_stats.index:
                features.extend(user_stats.loc[user_id_encoded].values)
            else:
                features.extend([3.0, 0.0, 0.0])  # 默认值
            
            user_features_list.append(features)
        
        self.user_features_df = pd.DataFrame(user_features_list)
        self.user_features_df = pd.DataFrame(
            self.feature_scaler.fit_transform(self.user_features_df),
            columns=self.user_features_df.columns
        )
    
    def _create_movie_features(self):
        """创建电影特征"""
        # 电影评分统计
        movie_stats = self.ratings_df.groupby('movie_id_encoded').agg({
            'rating': ['mean', 'std', 'count']
        }).fillna(0)
        movie_stats.columns = ['rating_mean', 'rating_std', 'rating_count']
        
        # 电影类型特征
        if 'genres' in self.movies_df.columns:
            # 提取所有可能的类型
            all_genres = set()
            for genres in self.movies_df['genres'].dropna():
                for genre in genres.split('|'):
                    all_genres.add(genre)
            
            # 创建类型特征矩阵
            genre_features = []
            for movie_id_encoded in range(len(self.movie_encoder.classes_)):
                original_movie_id = self.movie_encoder.inverse_transform([movie_id_encoded])[0]
                movie_data = self.movies_df[self.movies_df['movie_id'] == original_movie_id]
                
                if len(movie_data) > 0 and 'genres' in movie_data.iloc[0]:
                    genres = movie_data.iloc[0]['genres'].split('|')
                    genre_vector = [1 if genre in genres else 0 for genre in all_genres]
                else:
                    genre_vector = [0] * len(all_genres)
                
                genre_features.append(genre_vector)
            
            genre_df = pd.DataFrame(genre_features, columns=[f'genre_{g}' for g in all_genres])
        else:
            genre_df = pd.DataFrame({'genre_unknown': [1] * len(self.movie_encoder.classes_)})
        
        # 合并电影特征
        movie_features_list = []
        for movie_id_encoded in range(len(self.movie_encoder.classes_)):
            features = []
            
            # 评分统计
            if movie_id_encoded in movie_stats.index:
                features.extend(movie_stats.loc[movie_id_encoded].values)
            else:
                features.extend([3.0, 0.0, 0.0])  # 默认值
            
            # 类型特征
            features.extend(genre_df.iloc[movie_id_encoded].values)
            
            movie_features_list.append(features)
        
        self.movie_features_df = pd.DataFrame(movie_features_list)
        self.movie_features_df = pd.DataFrame(
            self.feature_scaler.fit_transform(self.movie_features_df),
            columns=self.movie_features_df.columns
        )
    
    def get_data_loaders(self, batch_size=256, test_size=0.2):
        """获取数据加载器"""
        # 准备训练数据
        user_ids = self.ratings_df['user_id_encoded'].values
        movie_ids = self.ratings_df['movie_id_encoded'].values
        ratings = self.ratings_df['rating'].values
        
        user_features = self.user_features_df.iloc[user_ids].values
        movie_features = self.movie_features_df.iloc[movie_ids].values
        
        # 分割训练集和测试集
        train_idx, test_idx = train_test_split(
            np.arange(len(ratings)), test_size=test_size, random_state=42
        )
        
        # 创建数据集
        train_dataset = MovieLensDataset(
            user_ids[train_idx], movie_ids[train_idx], ratings[train_idx],
            user_features[train_idx], movie_features[train_idx]
        )
        
        test_dataset = MovieLensDataset(
            user_ids[test_idx], movie_ids[test_idx], ratings[test_idx],
            user_features[test_idx], movie_features[test_idx]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def get_user_mapping(self):
        """获取用户ID映射"""
        return self.user_encoder
    
    def get_movie_mapping(self):
        """获取电影ID映射"""
        return self.movie_encoder
    
    def get_feature_dimensions(self):
        """获取特征维度"""
        user_feat_dim = self.user_features_df.shape[1]
        movie_feat_dim = self.movie_features_df.shape[1]
        return user_feat_dim, movie_feat_dim