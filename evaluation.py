import torch
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')
from matplotlib import font_manager as fm

# 指定字体路径
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'  # 改为实际路径
prop = fm.FontProperties(fname=font_path)

class ModelEvaluator:
    """Model Evaluator"""
    
    def __init__(self, processor, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.processor = processor
        self.device = device
        self.results = {}
    
    def load_model(self, model, model_path):
        """Load model weights"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check if state_dict exists in checkpoint
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Warning: Model file {model_path} does not exist")
            return False
    
    def analyze_dataset_split(self, train_loader, test_loader, experiment_path):
        """Analyze and output dataset split information"""
        print("\n" + "="*60)
        print("Dataset Split Analysis")
        print("="*60)
        
        # Collect training set information
        train_users = []
        train_movies = []
        train_ratings = []
        
        for batch in train_loader:
            train_users.extend(batch['user_id'].numpy())
            train_movies.extend(batch['movie_id'].numpy())
            train_ratings.extend(batch['rating'].numpy())
        
        # Collect test set information
        test_users = []
        test_movies = []
        test_ratings = []
        
        for batch in test_loader:
            test_users.extend(batch['user_id'].numpy())
            test_movies.extend(batch['movie_id'].numpy())
            test_ratings.extend(batch['rating'].numpy())
        
        # Convert to numpy arrays
        train_users = np.array(train_users)
        train_movies = np.array(train_movies)
        train_ratings = np.array(train_ratings)
        test_users = np.array(test_users)
        test_movies = np.array(test_movies)
        test_ratings = np.array(test_ratings)
        
        # Calculate statistics
        split_info = {
            'overall': {
                'total_samples': len(train_ratings) + len(test_ratings),
                'train_samples': len(train_ratings),
                'test_samples': len(test_ratings),
                'train_ratio': len(train_ratings) / (len(train_ratings) + len(test_ratings)),
                'test_ratio': len(test_ratings) / (len(train_ratings) + len(test_ratings))
            },
            'users': {
                'total_unique_users': len(np.unique(np.concatenate([train_users, test_users]))),
                'train_unique_users': len(np.unique(train_users)),
                'test_unique_users': len(np.unique(test_users)),
                'users_only_in_train': len(set(train_users) - set(test_users)),
                'users_only_in_test': len(set(test_users) - set(train_users)),
                'users_in_both': len(set(train_users) & set(test_users)),
                'user_overlap_ratio': len(set(train_users) & set(test_users)) / len(np.unique(np.concatenate([train_users, test_users])))
            },
            'movies': {
                'total_unique_movies': len(np.unique(np.concatenate([train_movies, test_movies]))),
                'train_unique_movies': len(np.unique(train_movies)),
                'test_unique_movies': len(np.unique(test_movies)),
                'movies_only_in_train': len(set(train_movies) - set(test_movies)),
                'movies_only_in_test': len(set(test_movies) - set(train_movies)),
                'movies_in_both': len(set(train_movies) & set(test_movies)),
                'movie_overlap_ratio': len(set(train_movies) & set(test_movies)) / len(np.unique(np.concatenate([train_movies, test_movies])))
            },
            'ratings': {
                'train_rating_stats': {
                    'mean': float(train_ratings.mean()),
                    'std': float(train_ratings.std()),
                    'min': float(train_ratings.min()),
                    'max': float(train_ratings.max()),
                    'median': float(np.median(train_ratings))
                },
                'test_rating_stats': {
                    'mean': float(test_ratings.mean()),
                    'std': float(test_ratings.std()),
                    'min': float(test_ratings.min()),
                    'max': float(test_ratings.max()),
                    'median': float(np.median(test_ratings))
                },
                'rating_distribution_train': {str(k): int(v) for k, v in zip(*np.unique(train_ratings, return_counts=True))},
                'rating_distribution_test': {str(k): int(v) for k, v in zip(*np.unique(test_ratings, return_counts=True))}
            },
            'user_activity': {
                'train_users_avg_ratings': float(len(train_ratings) / len(np.unique(train_users))),
                'test_users_avg_ratings': float(len(test_ratings) / len(np.unique(test_users))),
                'train_movies_avg_ratings': float(len(train_ratings) / len(np.unique(train_movies))),
                'test_movies_avg_ratings': float(len(test_ratings) / len(np.unique(test_movies)))
            }
        }
        
        # Output to console
        print(f"\nOverall Statistics:")
        print(f"  Total Samples: {split_info['overall']['total_samples']:,}")
        print(f"  Train Samples: {split_info['overall']['train_samples']:,} ({split_info['overall']['train_ratio']:.1%})")
        print(f"  Test Samples: {split_info['overall']['test_samples']:,} ({split_info['overall']['test_ratio']:.1%})")
        
        print(f"\nUser Statistics:")
        print(f"  Total Unique Users: {split_info['users']['total_unique_users']:,}")
        print(f"  Train Unique Users: {split_info['users']['train_unique_users']:,}")
        print(f"  Test Unique Users: {split_info['users']['test_unique_users']:,}")
        print(f"  Users Only in Train: {split_info['users']['users_only_in_train']:,}")
        print(f"  Users Only in Test: {split_info['users']['users_only_in_test']:,}")
        print(f"  Users in Both Sets: {split_info['users']['users_in_both']:,}")
        print(f"  User Overlap Ratio: {split_info['users']['user_overlap_ratio']:.1%}")
        
        print(f"\nMovie Statistics:")
        print(f"  Total Unique Movies: {split_info['movies']['total_unique_movies']:,}")
        print(f"  Train Unique Movies: {split_info['movies']['train_unique_movies']:,}")
        print(f"  Test Unique Movies: {split_info['movies']['test_unique_movies']:,}")
        print(f"  Movies Only in Train: {split_info['movies']['movies_only_in_train']:,}")
        print(f"  Movies Only in Test: {split_info['movies']['movies_only_in_test']:,}")
        print(f"  Movies in Both Sets: {split_info['movies']['movies_in_both']:,}")
        print(f"  Movie Overlap Ratio: {split_info['movies']['movie_overlap_ratio']:.1%}")
        
        print(f"\nRating Statistics:")
        print(f"  Train Rating Mean: {split_info['ratings']['train_rating_stats']['mean']:.2f}")
        print(f"  Test Rating Mean: {split_info['ratings']['test_rating_stats']['mean']:.2f}")
        print(f"  Train Rating Distribution: {split_info['ratings']['rating_distribution_train']}")
        print(f"  Test Rating Distribution: {split_info['ratings']['rating_distribution_test']}")
        
        print(f"\nUser Activity:")
        print(f"  Avg Ratings per User (Train): {split_info['user_activity']['train_users_avg_ratings']:.2f}")
        print(f"  Avg Ratings per User (Test): {split_info['user_activity']['test_users_avg_ratings']:.2f}")
        print(f"  Avg Ratings per Movie (Train): {split_info['user_activity']['train_movies_avg_ratings']:.2f}")
        print(f"  Avg Ratings per Movie (Test): {split_info['user_activity']['test_movies_avg_ratings']:.2f}")
        
        # Save to file
        split_info_file = os.path.join(experiment_path, 'reports', 'dataset_split_info.json')
        with open(split_info_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)
        
        # Generate visualization charts
        self.plot_dataset_split_analysis(split_info, experiment_path)
        
        return split_info
    
    def plot_dataset_split_analysis(self, split_info, experiment_path):
        """Plot dataset split analysis charts"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Sample Distribution
        labels = ['Train Set', 'Test Set']
        sizes = [split_info['overall']['train_samples'], split_info['overall']['test_samples']]
        colors = ['#66b3ff', '#ff9999']
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Sample Distribution')
        axes[0, 0].axis('equal')
        
        # User Distribution
        user_data = [
            split_info['users']['users_only_in_train'],
            split_info['users']['users_in_both'],
            split_info['users']['users_only_in_test']
        ]
        user_labels = ['Only in Train', 'In Both Sets', 'Only in Test']
        axes[0, 1].bar(user_labels, user_data, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_title('User Distribution')
        axes[0, 1].set_ylabel('Number of Users')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Movie Distribution
        movie_data = [
            split_info['movies']['movies_only_in_train'],
            split_info['movies']['movies_in_both'],
            split_info['movies']['movies_only_in_test']
        ]
        movie_labels = ['Only in Train', 'In Both Sets', 'Only in Test']
        axes[0, 2].bar(movie_labels, movie_data, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 2].set_title('Movie Distribution')
        axes[0, 2].set_ylabel('Number of Movies')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Rating Distribution Comparison
        train_ratings = list(split_info['ratings']['rating_distribution_train'].keys())
        train_counts = list(split_info['ratings']['rating_distribution_train'].values())
        test_ratings = list(split_info['ratings']['rating_distribution_test'].keys())
        test_counts = list(split_info['ratings']['rating_distribution_test'].values())
        
        x = np.arange(len(train_ratings))
        width = 0.35
        axes[1, 0].bar(x - width/2, train_counts, width, label='Train Set', color='#1f77b4')
        axes[1, 0].bar(x + width/2, test_counts, width, label='Test Set', color='#ff7f0e')
        axes[1, 0].set_xlabel('Rating')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Rating Distribution Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(train_ratings)
        axes[1, 0].legend()
        
        # Statistical Metrics Comparison
        metrics = ['Mean', 'Std Dev', 'Min', 'Max', 'Median']
        train_metrics = [
            split_info['ratings']['train_rating_stats']['mean'],
            split_info['ratings']['train_rating_stats']['std'],
            split_info['ratings']['train_rating_stats']['min'],
            split_info['ratings']['train_rating_stats']['max'],
            split_info['ratings']['train_rating_stats']['median']
        ]
        test_metrics = [
            split_info['ratings']['test_rating_stats']['mean'],
            split_info['ratings']['test_rating_stats']['std'],
            split_info['ratings']['test_rating_stats']['min'],
            split_info['ratings']['test_rating_stats']['max'],
            split_info['ratings']['test_rating_stats']['median']
        ]
        
        x = np.arange(len(metrics))
        axes[1, 1].bar(x - width/2, train_metrics, width, label='Train Set', color='#1f77b4')
        axes[1, 1].bar(x + width/2, test_metrics, width, label='Test Set', color='#ff7f0e')
        axes[1, 1].set_xlabel('Metric')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Rating Statistics Comparison')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend()
        
        # Activity Metrics
        activity_metrics = ['Ratings per User', 'Ratings per Movie']
        train_activity = [
            split_info['user_activity']['train_users_avg_ratings'],
            split_info['user_activity']['train_movies_avg_ratings']
        ]
        test_activity = [
            split_info['user_activity']['test_users_avg_ratings'],
            split_info['user_activity']['test_movies_avg_ratings']
        ]
        
        x = np.arange(len(activity_metrics))
        axes[1, 2].bar(x - width/2, train_activity, width, label='Train Set', color='#1f77b4')
        axes[1, 2].bar(x + width/2, test_activity, width, label='Test Set', color='#ff7f0e')
        axes[1, 2].set_xlabel('Activity Metric')
        axes[1, 2].set_ylabel('Average Value')
        axes[1, 2].set_title('User/Movie Activity Comparison')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(activity_metrics)
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plot_file = os.path.join(experiment_path, 'plots', 'dataset_split_analysis.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nDataset split analysis plot saved to: {plot_file}")
    
    def evaluate_model(self, model, model_type, test_loader, experiment_path):
        """Evaluate model"""
        print(f"\nEvaluating {model_type} model...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_user_ids = []
        all_movie_ids = []
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids = batch['user_id'].to(self.device)
                movie_ids = batch['movie_id'].to(self.device)
                ratings = batch['rating'].to(self.device)
                user_features = batch['user_features'].to(self.device)
                movie_features = batch['movie_features'].to(self.device)
                
                # Predict
                predictions = model(user_ids, movie_ids, user_features, movie_features)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
                all_user_ids.extend(user_ids.cpu().numpy())
                all_movie_ids.extend(movie_ids.cpu().numpy())
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_targets, all_predictions)
        
        # Save prediction results
        predictions_df = pd.DataFrame({
            'user_id_encoded': all_user_ids,
            'movie_id_encoded': all_movie_ids,
            'actual': all_targets,
            'predicted': all_predictions,
            'error': np.array(all_predictions) - np.array(all_targets),
            'abs_error': np.abs(np.array(all_predictions) - np.array(all_targets))
        })
        
        predictions_file = os.path.join(experiment_path, 'predictions', f'{model_type}_predictions.csv')
        predictions_df.to_csv(predictions_file, index=False)
        
        # Save metrics
        metrics_file = os.path.join(experiment_path, 'reports', f'{model_type}_evaluation_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"Evaluation results saved to: {predictions_file}")
        print(f"Metrics saved to: {metrics_file}")
        
        self.results[model_type] = {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'dataframe': predictions_df
        }
        
        return metrics
    
    def calculate_metrics(self, targets, predictions):
        """Calculate evaluation metrics"""
        targets = np.array(targets)
        predictions = np.array(predictions)
        
        # Basic regression metrics
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        # Classification metrics (treat ratings as discrete categories)
        targets_discrete = np.round(targets).astype(int)
        predictions_discrete = np.clip(np.round(predictions), 1, 5).astype(int)
        
        # Precision, recall, F1 score
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_discrete, predictions_discrete, 
            labels=[1, 2, 3, 4, 5], average='weighted', zero_division=0
        )
        
        # Accuracy
        accuracy = np.mean(targets_discrete == predictions_discrete)
        
        # Error statistics
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        error_stats = {
            'mean_error': float(errors.mean()),
            'std_error': float(errors.std()),
            'mae': float(mae),
            'rmse': float(rmse),
            'max_error': float(abs_errors.max()),
            'min_error': float(abs_errors.min()),
            'perfect_predictions': int((abs_errors < 0.5).sum()),
            'perfect_predictions_percentage': float((abs_errors < 0.5).sum() / len(targets) * 100)
        }
        
        return {
            'basic_metrics': {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1)
            },
            'error_stats': error_stats,
            'sample_count': len(targets)
        }
    
    def generate_comparison_report(self, experiment_path):
        """Generate model comparison report"""
        if len(self.results) == 0:
            print("No evaluation results to compare")
            return None
        
        # Create comparison data
        comparison_data = []
        for model_type, result in self.results.items():
            metrics = result['metrics']['basic_metrics']
            comparison_data.append({
                'Model': model_type,
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R2': metrics['r2'],
                'Accuracy': metrics['accuracy'],
                'F1': metrics['f1']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        report_file = os.path.join(experiment_path, 'reports', 'model_comparison_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("                    Model Performance Comparison Report\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of models evaluated: {len(self.results)}\n\n")
            
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            
            # Best model
            if len(comparison_data) > 1:
                best_model = min(comparison_data, key=lambda x: x['RMSE'])
                f.write(f"Best model: {best_model['Model']} (RMSE: {best_model['RMSE']:.4f})\n\n")
            
            # Detailed analysis
            for model_type, result in self.results.items():
                f.write(f"{'='*40}\n")
                f.write(f"{model_type} model detailed results:\n")
                f.write(f"{'='*40}\n")
                
                metrics = result['metrics']['basic_metrics']
                error_stats = result['metrics']['error_stats']
                
                for key, value in metrics.items():
                    f.write(f"  {key}: {value:.4f}\n")
                
                f.write(f"  Error statistics:\n")
                f.write(f"    Mean error: {error_stats['mean_error']:.4f}\n")
                f.write(f"    Error std: {error_stats['std_error']:.4f}\n")
                f.write(f"    Max error: {error_stats['max_error']:.4f}\n")
                f.write(f"    Perfect prediction percentage: {error_stats['perfect_predictions_percentage']:.2f}%\n")
                f.write(f"    Perfect prediction count: {error_stats['perfect_predictions']}\n")
        
        print(f"Comparison report saved to: {report_file}")
        
        # Visualization comparison
        self.plot_comparison(comparison_df, experiment_path)
        
        return comparison_df
    
    def plot_comparison(self, comparison_df, experiment_path):
        """Plot model comparison charts"""
        if comparison_df.empty:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # RMSE comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['RMSE'])
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['MAE'])
        axes[0, 1].set_title('MAE Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R^2 comparison
        axes[0, 2].bar(comparison_df['Model'], comparison_df['R2'])
        axes[0, 2].set_title('R² Comparison')
        axes[0, 2].set_ylabel('R²')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Accuracy comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['Accuracy'])
        axes[1, 0].set_title('Accuracy Comparison')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # F1 score comparison
        axes[1, 1].bar(comparison_df['Model'], comparison_df['F1'])
        axes[1, 1].set_title('F1 Score Comparison')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Radar chart
        ax_radar = axes[1, 2]
        metrics_to_plot = ['RMSE', 'MAE', 'R2', 'Accuracy', 'F1']
        num_vars = len(metrics_to_plot)
        
        # Normalize data
        plot_data = comparison_df[metrics_to_plot].copy()
        for metric in ['RMSE', 'MAE']:
            plot_data[metric] = 1 - (plot_data[metric] / plot_data[metric].max())
        for metric in ['R2', 'Accuracy', 'F1']:
            plot_data[metric] = plot_data[metric] / plot_data[metric].max()
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        for idx, row in plot_data.iterrows():
            values = row[metrics_to_plot].values.tolist()
            values += values[:1]
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=comparison_df.iloc[idx]['Model'])
            ax_radar.fill(angles, values, alpha=0.25)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics_to_plot)
        ax_radar.set_title('Model Comparison Radar Chart')
        ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        plot_file = os.path.join(experiment_path, 'plots', 'model_comparison.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Comparison plot saved to: {plot_file}")