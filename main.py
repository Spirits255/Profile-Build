import torch
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

from config import Config
from data_processor import DataProcessor
from models import UserProfilingModel, BaselineModel,TransformerRecommender
from trainer import AdvancedModelTrainer, RecommenderSystem
from evaluation import ModelEvaluator

def setup_experiment():
    """设置实验"""
    print("="*80)
    print("             社交网络用户画像技术研究")
    print("="*80)
    
    # 设置配置
    config_dict = Config.setup_experiment()
    
    print(f"\n实验配置:")
    print(f"  模式: {config_dict['mode']}")
    print(f"  数据路径: {config_dict['data_path']}")
    print(f"  实验名称: {config_dict['experiment_name']}")
    print(f"  实验路径: {config_dict['experiment_path']}")
    
    # 保存配置
    Config.save_config()
    
    return config_dict

def load_data(config_dict):
    """加载和预处理数据"""
    print("\n" + "="*60)
    print("数据加载和预处理")
    print("="*60)
    
    processor = DataProcessor(config_dict['data_path'])
    if not processor.load_data():
        print("数据加载失败，请检查数据路径")
        return None
    
    processor.preprocess_data(
        min_ratings_per_user=config_dict['data_preprocessing']['min_ratings_per_user'],
        min_ratings_per_movie=config_dict['data_preprocessing']['min_ratings_per_movie']
    )
    
    # 获取数据加载器
    train_loader, test_loader = processor.get_data_loaders(
        batch_size=config_dict['training']['batch_size'],
        test_size=config_dict['evaluation']['test_size']
    )
    
    # 获取特征维度
    user_feat_dim, movie_feat_dim = processor.get_feature_dimensions()
    num_users = len(processor.get_user_mapping().classes_)
    num_movies = len(processor.get_movie_mapping().classes_)
    
    print(f"\n数据统计:")
    print(f"  - 用户数量: {num_users}")
    print(f"  - 电影数量: {num_movies}")
    print(f"  - 用户特征维度: {user_feat_dim}")
    print(f"  - 电影特征维度: {movie_feat_dim}")
    print(f"  - 评分记录数: {len(processor.ratings_df)}")
    
    # 保存数据统计
    data_stats = {
        'num_users': num_users,
        'num_movies': num_movies,
        'user_feature_dim': user_feat_dim,
        'movie_feature_dim': movie_feat_dim,
        'total_ratings': len(processor.ratings_df),
        'rating_distribution': processor.ratings_df['rating'].value_counts().to_dict()
    }
    
    stats_file = os.path.join(config_dict['experiment_path'], 'reports', 'data_statistics.json')
    with open(stats_file, 'w') as f:
        json.dump(data_stats, f, indent=2)
    
    return processor, train_loader, test_loader, user_feat_dim, movie_feat_dim, num_users, num_movies

def train_models(config_dict, processor, train_loader, test_loader, user_feat_dim, movie_feat_dim, num_users, num_movies):
    """训练模型"""
    print("\n" + "="*60)
    print("训练模型")
    print("="*60)
    
    all_metrics = {}
    all_trainers = {}
    
    models_to_train = config_dict['evaluation']['models_to_evaluate']
    
    for model_type in models_to_train:
        try:
            print(f"\n{'='*40}")
            print(f"训练 {model_type} 模型")
            print(f"{'='*40}")
            
            # 初始化模型
            if model_type == 'deep':
                model = UserProfilingModel(
                    num_users, num_movies, user_feat_dim, movie_feat_dim,
                    embedding_dim=config_dict['models']['deep']['embedding_dim'],
                    hidden_dim=config_dict['models']['deep']['hidden_dim']
                )
            elif model_type == 'baseline':
                model = BaselineModel(
                    num_users, num_movies,
                    embedding_dim=config_dict['models']['baseline']['embedding_dim']
                )
            elif model_type == 'transformer':
                model = TransformerRecommender(num_users, num_movies, user_feat_dim, movie_feat_dim,
                    embedding_dim=config_dict['models']['transformer']['embedding_dim'],
                    nhead=config_dict['models']['transformer']['nhead'],
                    num_layers=config_dict['models']['transformer']['num_layers'],
                    dim_feedforward=config_dict['models']['transformer']['dim_feedforward'],
                    dropout=config_dict['models']['transformer']['dropout']
                    )
            else:
                print(f"未知模型类型: {model_type}")
                continue
            
            # 打印模型信息
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"总参数: {total_params:,}")
            print(f"可训练参数: {trainable_params:,}")
            
            # 创建训练器
            trainer = AdvancedModelTrainer(
                model, 
                train_loader, 
                test_loader, 
                model_name=model_type,
                use_amp=config_dict['training']['use_amp']
            )
            
            # 训练模型
            trainer.train(
                epochs=config_dict['training']['epochs'],
                lr=config_dict['training']['learning_rate'],
                save_best=True
            )

            # 保存训练/验证损失曲线到实验目录下的 plots 文件夹
            plots_dir = os.path.join(config_dict['experiment_path'], 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            loss_plot_path = os.path.join(plots_dir, f'training_loss_{model_type}.png')
            try:
                trainer.plot_loss_curve(save_path=loss_plot_path)
            except Exception:
                # 如果单独的 loss 绘制失败，继续不会中断训练流程
                pass
            
            # 计算指标
            metrics = trainer.calculate_detailed_metrics()
            
            # 绘制训练历史
            trainer.plot_training_history()
            
            # 保存模型
            model_path = os.path.join(config_dict['experiment_path'], 'models', f'{model_type}_model.pth')
            trainer.save_model(model_path)
            
            # 保存预测结果
            predictions_df = pd.DataFrame({
                'actual': trainer.targets,
                'predicted': trainer.predictions,
                'error': np.array(trainer.predictions) - np.array(trainer.targets),
                'abs_error': np.abs(np.array(trainer.predictions) - np.array(trainer.targets))
            })
            
            predictions_file = os.path.join(config_dict['experiment_path'], 'predictions', f'{model_type}_predictions.csv')
            predictions_df.to_csv(predictions_file, index=False)
            
            # 保存指标
            metrics_to_save = {
                'basic_metrics': {
                    'rmse': metrics.get('rmse', 0),
                    'mae': metrics.get('mae', 0),
                    'r2': metrics.get('r2', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': metrics.get('f1', 0)
                },
                'model_info': {
                    'parameters': total_params,
                    'model_type': model_type,
                    'embedding_dim': config_dict['models'][model_type]['embedding_dim'] if model_type in config_dict['models'] else 64
                }
            }
            
            metrics_file = os.path.join(config_dict['experiment_path'], 'reports', f'{model_type}_metrics.json')
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)
            
            all_metrics[model_type] = metrics_to_save
            all_trainers[model_type] = trainer
            
            print(f"\n{model_type} 模型训练完成!")
            print(f"RMSE: {metrics_to_save['basic_metrics']['rmse']:.4f}")
            print(f"MAE: {metrics_to_save['basic_metrics']['mae']:.4f}")
            
        except Exception as e:
            print(f"训练 {model_type} 模型时出错: {e}")
            import traceback
            traceback.print_exc()
    
    return all_metrics, all_trainers

def evaluate_existing_models(config_dict, processor, test_loader, user_feat_dim, movie_feat_dim, num_users, num_movies):
    """评估已存在的模型"""
    print("\n" + "="*60)
    print("评估现有模型")
    print("="*60)
    
    evaluator = ModelEvaluator(processor)
    all_metrics = {}
    
    models_to_evaluate = config_dict['evaluation']['models_to_evaluate']
    experiment_path = config_dict['experiment_path']
    
    # 首先获取训练数据加载器来分析数据集划分情况
    # 注意：这里我们需要重新获取训练数据加载器，因为test_loader已经传入
    # 我们需要同时有训练和测试数据加载器才能分析
    train_loader, _ = processor.get_data_loaders(
        batch_size=config_dict['training']['batch_size'],
        test_size=config_dict['evaluation']['test_size']
    )
    
    # 分析数据集划分情况
    evaluator.analyze_dataset_split(train_loader, test_loader, experiment_path)
    
    # 继续原有的评估逻辑
    for model_type in models_to_evaluate:
        try:
            print(f"\n评估 {model_type} 模型...")
            
            # 初始化模型
            if model_type == 'deep':
                model = UserProfilingModel(
                    num_users, num_movies, user_feat_dim, movie_feat_dim,
                    embedding_dim=config_dict['models']['deep']['embedding_dim'],
                    hidden_dim=config_dict['models']['deep']['hidden_dim']
                )
            elif model_type == 'baseline':
                model = BaselineModel(
                    num_users, num_movies,
                    embedding_dim=config_dict['models']['baseline']['embedding_dim']
                )
            elif model_type == 'transformer':
                model = TransformerRecommender(num_users, num_movies, user_feat_dim, movie_feat_dim,
                    embedding_dim=config_dict['models']['transformer']['embedding_dim'],
                    nhead=config_dict['models']['transformer']['nhead'],
                    num_layers=config_dict['models']['transformer']['num_layers'],
                    dim_feedforward=config_dict['models']['transformer']['dim_feedforward'],
                    dropout=config_dict['models']['transformer']['dropout']
                    )
            else:
                print(f"未知模型类型: {model_type}")
                continue
            
            # 加载模型权重
            model_path = os.path.join(experiment_path, 'models', f'{model_type}_model.pth')
            if not evaluator.load_model(model, model_path):
                print(f"无法加载 {model_type} 模型，跳过...")
                continue
            
            # 评估模型
            metrics = evaluator.evaluate_model(model, model_type, test_loader, experiment_path)
            all_metrics[model_type] = metrics
            
            print(f"{model_type} 模型评估完成!")
            print(f"RMSE: {metrics['basic_metrics']['rmse']:.4f}")
            print(f"MAE: {metrics['basic_metrics']['mae']:.4f}")
            
        except Exception as e:
            print(f"评估 {model_type} 模型时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成比较报告
    if evaluator.results:
        comparison_df = evaluator.generate_comparison_report(experiment_path)
        
        # 使用最佳模型进行推荐演示
        if comparison_df is not None and not comparison_df.empty:
            best_model_type = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
            print(f"\n最佳模型: {best_model_type}")
            
            # 加载最佳模型进行推荐演示
            try:
                if best_model_type == 'deep':
                    best_model = UserProfilingModel(
                        num_users, num_movies, user_feat_dim, movie_feat_dim
                    )
                elif best_model_type == 'baseline':
                    best_model = BaselineModel(num_users, num_movies)
                elif best_model_type == 'transformer':
                    best_model = TransformerRecommender(num_users, num_movies, user_feat_dim, movie_feat_dim)
                
                model_path = os.path.join(experiment_path, 'models', f'{best_model_type}_model.pth')
                if evaluator.load_model(best_model, model_path):
                    # 使用安全的推荐方法
                    from trainer import RecommenderSystem
                    recommender = RecommenderSystem(best_model, processor)
                    
                    # 为示例用户生成推荐（只使用训练时见过的用户）
                    train_user_ids = set(processor.user_encoder.classes_)
                    available_users = [uid for uid in processor.ratings_df['user_id'].unique() 
                                     if uid in train_user_ids]
                    
                    if len(available_users) > 0:
                        sample_users = np.random.choice(available_users, min(2, len(available_users)), replace=False)
                        
                        print(f"\n使用最佳模型 ({best_model_type}) 为 {len(sample_users)} 个示例用户生成推荐:")
                        
                        for user_id in sample_users:
                            print(f"\n用户 {user_id} 的推荐:")
                            
                            # 使用安全推荐方法
                            recommendations = recommender.safe_recommend_for_user(user_id, top_k=3)
                            
                            if recommendations and 'error' not in recommendations[0]:
                                for i, rec in enumerate(recommendations, 1):
                                    print(f"  {i}. {rec.get('title', '未知电影')[:50]}...")
                                    print(f"     类型: {rec.get('genres', '未知类型')}")
                                    print(f"     预测评分: {rec.get('predicted_rating', 0):.2f}")
                                    if 'actual_avg_rating' in rec:
                                        print(f"     实际平均评分: {rec['actual_avg_rating']:.2f}")
                                    print()
                            else:
                                print("  " + (recommendations[0].get('error', '没有推荐结果') if recommendations else '没有推荐结果'))
                    else:
                        print("没有可用的用户进行推荐演示")
                        
            except Exception as e:
                print(f"推荐演示时出现错误: {e}")
                import traceback
                traceback.print_exc()
    
    return all_metrics
def main():
    """主函数"""
    try:
        # 设置实验
        config_dict = setup_experiment()
        
        # 设置随机种子
        torch.manual_seed(config_dict['evaluation']['random_seed'])
        np.random.seed(config_dict['evaluation']['random_seed'])
        
        # 加载数据
        data_result = load_data(config_dict)
        if data_result is None:
            return
        
        processor, train_loader, test_loader, user_feat_dim, movie_feat_dim, num_users, num_movies = data_result
        
        if config_dict['mode'] == 'train':
            # 训练模式
            all_metrics, all_trainers = train_models(
                config_dict, processor, train_loader, test_loader,
                user_feat_dim, movie_feat_dim, num_users, num_movies
            )
            
            print("\n" + "="*80)
            print("训练完成!")
            print(f"所有结果已保存到: {config_dict['experiment_path']}")
            print("="*80)
            
        elif config_dict['mode'] == 'evaluate':
            # 评估模式
            all_metrics = evaluate_existing_models(
                config_dict, processor, test_loader,
                user_feat_dim, movie_feat_dim, num_users, num_movies
            )
            
            print("\n" + "="*80)
            print("评估完成!")
            print(f"评估结果已保存到: {config_dict['experiment_path']}")
            print("="*80)
        
    except Exception as e:
        print(f"程序执行时出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()