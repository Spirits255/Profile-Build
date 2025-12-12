
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
import seaborn as sns
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelTrainer:
    """高级模型训练器，支持多种优化策略"""
    
    def __init__(self, model, train_loader, test_loader, 
                 model_name='transformer', use_amp=True):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model_name = model_name
        self.use_amp = use_amp  # 自动混合精度训练
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 训练历史
        self.train_losses = []
        self.test_losses = []
        self.learning_rates = []
        self.predictions = []
        self.targets = []
        
        # 优化器设置
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
        # 梯度裁剪
        self.grad_clip = 1.0
        
        # 早停设置
        self.patience = 10
        self.best_loss = float('inf')
        self.early_stop_counter = 0
        
        print(f"使用设备: {self.device}")
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            print("启用混合精度训练")
    
    def setup_optimizer(self, lr=0.001, weight_decay=1e-5):
        """设置优化器和学习率调度器"""
        # 不同参数使用不同的学习率
        param_groups = [
            {'params': self.model.user_embedding.parameters(), 'lr': lr * 0.5},
            {'params': self.model.movie_embedding.parameters(), 'lr': lr * 0.5},
            {'params': [p for n, p in self.model.named_parameters() 
                       if 'embedding' not in n], 'lr': lr}
        ]
        
        self.optimizer = optim.AdamW(
            param_groups, 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 动态学习率调度
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
        )
        
        # 可选：余弦退火
        # self.scheduler = CosineAnnealingWarmRestarts(
        #     self.optimizer, 
        #     T_0=10, 
        #     T_mult=2,
        #     eta_min=1e-6
        # )
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch in progress_bar:
            self.optimizer.zero_grad()
            
            # 移动到设备
            user_ids = batch['user_id'].to(self.device)
            movie_ids = batch['movie_id'].to(self.device)
            ratings = batch['rating'].to(self.device)
            user_features = batch['user_features'].to(self.device)
            movie_features = batch['movie_features'].to(self.device)
            
            # 混合精度训练
            if self.use_amp and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    predictions = self.model(user_ids, movie_ids, user_features, movie_features)
                    loss = self.criterion(predictions, ratings)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(user_ids, movie_ids, user_features, movie_features)
                loss = self.criterion(predictions, ratings)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/batch_count:.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def validate(self, save_predictions=False):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='验证中'):
                user_ids = batch['user_id'].to(self.device)
                movie_ids = batch['movie_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                user_features = batch['user_features'].to(self.device)
                movie_features = batch['movie_features'].to(self.device)
                
                predictions = self.model(user_ids, movie_ids, user_features, movie_features)
                loss = self.criterion(predictions, ratings)
                
                total_loss += loss.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        self.test_losses.append(avg_loss)
        
        if save_predictions:
            self.predictions = all_predictions
            self.targets = all_targets
        
        return avg_loss, all_predictions, all_targets
    
    def train(self, epochs=50, lr=0.001, save_best=True):
        """训练模型"""
        print(f"\n开始训练 {self.model_name} 模型...")
        print(f"总epoch数: {epochs}")
        print(f"初始学习率: {lr}")
        
        self.setup_optimizer(lr=lr)
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, _, _ = self.validate(save_predictions=(epoch == epochs-1))
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # 打印训练信息
            print(f'\nEpoch {epoch+1}/{epochs}:')
            print(f'  训练损失: {train_loss:.4f}')
            print(f'  验证损失: {val_loss:.4f}')
            print(f'  学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # 保存最佳模型
            if save_best and val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model(f'models/best_{self.model_name}_model.pth')
                self.early_stop_counter = 0
                print(f'  保存最佳模型 (损失: {val_loss:.4f})')
            else:
                self.early_stop_counter += 1
            
            # 早停检查
            if self.early_stop_counter >= self.patience:
                print(f'早停触发，停止训练')
                break
        
        # 加载最佳模型
        self.load_model(f'models/best_{self.model_name}_model.pth')
        print(f'训练完成，最佳验证损失: {self.best_loss:.4f}')
    
    def calculate_detailed_metrics(self):
        """计算详细的评估指标"""
        if len(self.predictions) == 0 or len(self.targets) == 0:
            _, self.predictions, self.targets = self.validate(save_predictions=True)
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        # 基础回归指标
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # 分类指标（将评分视为离散类别）
        targets_discrete = np.round(targets).astype(int)
        predictions_discrete = np.clip(np.round(predictions), 1, 5).astype(int)
        
        # 精度、召回率、F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_discrete, predictions_discrete, 
            labels=[1, 2, 3, 4, 5], average='weighted', zero_division=0
        )
        
        # 准确率
        accuracy = np.mean(targets_discrete == predictions_discrete)
        
        # 创建详细的评分统计
        detailed_stats = self._calculate_rating_statistics(targets, predictions)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detailed_stats': detailed_stats
        }
    
    def _calculate_rating_statistics(self, targets, predictions):
        """计算详细的评分统计信息"""
        stats_df = pd.DataFrame({
            'actual': targets,
            'predicted': predictions,
            'error': predictions - targets,
            'abs_error': np.abs(predictions - targets)
        })
        
        # 按实际评分的统计
        actual_stats = stats_df.groupby(
            pd.cut(stats_df['actual'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        ).agg({
            'predicted': ['mean', 'std', 'count'],
            'error': ['mean', 'std'],
            'abs_error': ['mean', 'std']
        }).round(4)
        
        # 预测评分的统计
        pred_stats = stats_df.groupby(
            pd.cut(stats_df['predicted'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        ).agg({
            'actual': ['mean', 'std', 'count'],
            'error': ['mean', 'std'],
            'abs_error': ['mean', 'std']
        }).round(4)
        
        # 误差分布统计
        error_stats = {
            'mean_error': stats_df['error'].mean(),
            'std_error': stats_df['error'].std(),
            'mae': stats_df['abs_error'].mean(),
            'rmse': np.sqrt((stats_df['error'] ** 2).mean()),
            'correlation': stats_df['actual'].corr(stats_df['predicted'])
        }
        
        # 极端情况分析
        large_errors = stats_df[stats_df['abs_error'] > 2]
        perfect_predictions = stats_df[stats_df['abs_error'] < 0.5]
        
        return {
            'overall_stats': error_stats,
            'by_actual_rating': actual_stats,
            'by_predicted_rating': pred_stats,
            'extreme_cases': {
                'large_errors_count': len(large_errors),
                'large_errors_percentage': len(large_errors) / len(stats_df) * 100,
                'perfect_predictions_count': len(perfect_predictions),
                'perfect_predictions_percentage': len(perfect_predictions) / len(stats_df) * 100
            },
            'raw_data': stats_df
        }
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Train / Validation loss
        axes[0, 0].plot(self.train_losses, label='Train Loss', alpha=0.8)
        axes[0, 0].plot(self.test_losses, label='Validation Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training History')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 1].plot(self.learning_rates, color='green', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Predictions vs Actual
        axes[0, 2].scatter(self.targets, self.predictions, alpha=0.3, s=10)
        axes[0, 2].plot([0.5, 5.5], [0.5, 5.5], 'r--', alpha=0.7)
        axes[0, 2].set_xlabel('Actual Rating')
        axes[0, 2].set_ylabel('Predicted Rating')
        axes[0, 2].set_title('Predicted vs Actual')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Error distribution
        errors = np.array(self.predictions) - np.array(self.targets)
        axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mean error by rating
        stats_df = pd.DataFrame({'actual': self.targets, 'error': errors})
        error_by_rating = stats_df.groupby(
            pd.cut(stats_df['actual'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        )['error'].agg(['mean', 'std']).dropna()
        
        x_pos = np.arange(len(error_by_rating))
        axes[1, 1].bar(x_pos, error_by_rating['mean'], 
                      yerr=error_by_rating['std'], 
                      alpha=0.7, capsize=5)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Actual Rating')
        axes[1, 1].set_ylabel('Mean Error')
        axes[1, 1].set_title('Mean Error by Actual Rating')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(['1 star', '2 star', '3 star', '4 star', '5 star'])
        axes[1, 1].grid(True, alpha=0.3)
        
        # Rating distribution comparison
        actual_counts = pd.Series(self.targets).value_counts().sort_index()
        pred_counts = pd.Series(np.round(self.predictions)).value_counts().reindex(
            actual_counts.index, fill_value=0
        )
        
        x = np.arange(len(actual_counts))
        width = 0.35
        axes[1, 2].bar(x - width/2, actual_counts.values, width, 
                      alpha=0.7, label='实际')
        axes[1, 2].bar(x + width/2, pred_counts.values, width, 
                      alpha=0.7, label='预测')
        axes[1, 2].set_xlabel('Rating')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Rating Distribution Comparison')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels([f'{i} star' for i in actual_counts.index])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'training_history_{self.model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_loss_curve(self, save_path=None):
        """Plot training and validation loss curve and save to file.

        Args:
            save_path (str|None): file path to save the figure. If None, saves to
                `plots/training_loss_{model_name}.png`.
        """
        if len(self.train_losses) == 0:
            print('No training loss history to plot.')
            return

        if save_path is None:
            os.makedirs('plots', exist_ok=True)
            save_path = os.path.join('plots', f'training_loss_{self.model_name}.png')

        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label='Train Loss', marker='o')
        if len(self.test_losses) > 0:
            plt.plot(self.test_losses, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'best_loss': self.best_loss
        }, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"模型已从 {path} 加载")
        else:
            print(f"警告: 模型文件 {path} 不存在")

# 保持向后兼容性
class ModelTrainer(AdvancedModelTrainer):
    """向后兼容的模型训练器"""
    pass

class RecommenderSystem:
    """推荐系统（更新版）"""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # 缓存训练时见过的电影ID集合
        self.train_movie_ids = set(processor.movie_encoder.classes_)
    
    def recommend_for_user(self, user_id, top_k=10, min_rating=3.5):
        """为用户生成推荐"""
        try:
            # 获取用户已评分的电影
            user_ratings = self.processor.ratings_df[
                self.processor.ratings_df['user_id'] == user_id
            ]
            rated_movies = set(user_ratings['movie_id'].values)
            
            # 只考虑训练时见过的电影
            unrated_movies = self.train_movie_ids - rated_movies
            
            if len(unrated_movies) == 0:
                print(f"用户 {user_id} 已评分所有训练集中的电影")
                return []
            
            # 准备预测数据
            predictions = []
            
            # 批量预测以提高效率
            batch_size = 512
            unrated_movies_list = list(unrated_movies)
            
            # 获取用户编码和特征
            user_encoded = self.processor.user_encoder.transform([user_id])[0]
            user_feature = self.processor.user_features_df.iloc[user_encoded].values
            
            with torch.no_grad():
                for i in range(0, len(unrated_movies_list), batch_size):
                    batch_movies = unrated_movies_list[i:i+batch_size]
                    
                    # 准备batch数据
                    user_ids = torch.tensor([user_encoded] * len(batch_movies), dtype=torch.long).to(self.device)
                    
                    # 确保电影ID可以被编码（应该是可以的，因为我们在训练集中）
                    try:
                        movie_encodings = self.processor.movie_encoder.transform(batch_movies)
                    except ValueError as e:
                        # 如果还有编码问题，跳过这一批
                        print(f"警告: 跳过无法编码的电影: {e}")
                        continue
                    
                    movie_ids = torch.tensor(movie_encodings, dtype=torch.long).to(self.device)
                    
                    user_features = torch.tensor(
                        [user_feature] * len(batch_movies), dtype=torch.float
                    ).to(self.device)
                    
                    movie_features = torch.tensor(
                        self.processor.movie_features_df.iloc[movie_encodings].values, 
                        dtype=torch.float
                    ).to(self.device)
                    
                    # 预测
                    batch_predictions = self.model(user_ids, movie_ids, user_features, movie_features)
                    
                    # 收集结果
                    for movie_id, pred_rating in zip(batch_movies, batch_predictions.cpu().numpy()):
                        predictions.append({
                            'movie_id': movie_id,
                            'predicted_rating': float(pred_rating)
                        })
            
            # 过滤和排序
            predictions = [p for p in predictions if p['predicted_rating'] >= min_rating]
            predictions.sort(key=lambda x: x['predicted_rating'], reverse=True)
            
            # 添加电影信息
            recommendations = []
            for pred in predictions[:top_k]:
                # 查找电影信息
                movie_mask = self.processor.movies_df['movie_id'] == pred['movie_id']
                if movie_mask.any():
                    movie_info = self.processor.movies_df[movie_mask].iloc[0]
                    
                    recommendation = {
                        'title': movie_info.get('title', '未知电影'),
                        'genres': movie_info.get('genres', '未知类型'),
                        'predicted_rating': pred['predicted_rating']
                    }
                    
                    # 如果可能，添加实际平均评分
                    if 'movie_id' in self.processor.ratings_df.columns:
                        movie_ratings = self.processor.ratings_df[
                            self.processor.ratings_df['movie_id'] == pred['movie_id']
                        ]
                        if len(movie_ratings) > 0:
                            recommendation['actual_avg_rating'] = movie_ratings['rating'].mean()
                    
                    recommendations.append(recommendation)
                else:
                    # 如果电影信息不存在，使用基本信息
                    recommendations.append({
                        'movie_id': pred['movie_id'],
                        'title': f'电影ID: {pred["movie_id"]}',
                        'genres': '未知类型',
                        'predicted_rating': pred['predicted_rating']
                    })
            
            return recommendations
            
        except Exception as e:
            print(f"为用户 {user_id} 生成推荐时出错: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def safe_recommend_for_user(self, user_id, top_k=10, min_rating=3.5):
        """安全版本的用户推荐，避免编码问题"""
        try:
            return self.recommend_for_user(user_id, top_k, min_rating)
        except Exception as e:
            print(f"安全推荐捕获错误: {e}")
            # 返回一个简单的推荐结果
            return [{
                'title': '推荐系统暂时不可用',
                'genres': '系统错误',
                'predicted_rating': 0.0,
                'error': str(e)
            }]