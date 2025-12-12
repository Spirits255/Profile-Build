import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class UserProfilingModel(nn.Module):
    """用户画像深度学习模型（原始版本，用于基准对比）"""
    
    def __init__(self, num_users, num_movies, user_feat_dim, movie_feat_dim, 
                 embedding_dim=64, hidden_dim=128):
        super(UserProfilingModel, self).__init__()
        
        # 嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # 特征编码层
        self.user_feature_encoder = nn.Sequential(
            nn.Linear(user_feat_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        
        self.movie_feature_encoder = nn.Sequential(
            nn.Linear(movie_feat_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        
        # 深度神经网络
        self.deep_layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(self, user_ids, movie_ids, user_features, movie_features):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        user_feat_emb = self.user_feature_encoder(user_features)
        movie_feat_emb = self.movie_feature_encoder(movie_features)
        
        combined = torch.cat([
            user_emb, 
            movie_emb, 
            user_feat_emb, 
            movie_feat_emb
        ], dim=1)
        
        output = self.deep_layers(combined)
        return output.squeeze()

class BaselineModel(nn.Module):
    """基线模型 - 矩阵分解"""
    
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super(BaselineModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1)
        
    def forward(self, user_ids, movie_ids, user_features=None, movie_features=None):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        interaction = user_emb * movie_emb
        output = self.output_layer(interaction)
        
        return output.squeeze()
    


from torch.nn import TransformerEncoderLayer, TransformerEncoder # 新增导入

class TransformerRecommender(nn.Module):
    """
    基于Transformer编码器的推荐模型 (Rating Prediction)
    
    它将用户ID、电影ID、用户特征和电影特征的嵌入连接起来，
    作为一个单一的“token”输入到Transformer Encoder Layer进行特征交互学习。
    """
    
    def __init__(self, num_users, num_movies, user_feat_dim, movie_feat_dim, 
                 embedding_dim=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerRecommender, self).__init__()
        
        # 1. 嵌入层
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # 2. 特征编码层 (保持与UserProfilingModel一致)
        half_dim = embedding_dim // 2
        
        self.user_feature_encoder = nn.Sequential(
            nn.Linear(user_feat_dim, half_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(half_dim, half_dim)
        )
        
        self.movie_feature_encoder = nn.Sequential(
            nn.Linear(movie_feat_dim, half_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(half_dim, half_dim)
        )
        
        # 3. Transformer 结构
        # 输入维度 D_model = user_emb + movie_emb + user_feat_emb + movie_feat_emb = 3 * embedding_dim
        d_model = embedding_dim * 3
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True # PyTorch Transformer默认第一个维度是Batch Size
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # 4. 预测层
        self.output_layer = nn.Linear(d_model, 1)
        
        # 5. 权重初始化
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)

    def forward(self, user_ids, movie_ids, user_features, movie_features):
        # 1. 获取嵌入
        user_emb = self.user_embedding(user_ids) # (B, E)
        movie_emb = self.movie_embedding(movie_ids) # (B, E)
        
        # 2. 编码特征
        user_feat_emb = self.user_feature_encoder(user_features) # (B, E/2)
        movie_feat_emb = self.movie_feature_encoder(movie_features) # (B, E/2)
        
        # 3. 拼接所有特征
        combined = torch.cat([
            user_emb, 
            movie_emb, 
            user_feat_emb, 
            movie_feat_emb
        ], dim=1) # (B, 3*E)
        
        # 4. 准备Transformer输入 (seq_len=1)
        x = combined.unsqueeze(1) # (B, 1, 3*E)
        
        # 5. 通过Transformer Encoder
        transformer_output = self.transformer_encoder(x) # (B, 1, 3*E)
        
        # 6. 恢复形状并预测
        output = transformer_output.squeeze(1) # (B, 3*E)
        output = self.output_layer(output) # (B, 1)
        
        return output.squeeze() # (B)