import os
from datetime import datetime

class Config:
    """配置类"""
    
    # 实验模式: 'train' 或 'evaluate'
    MODE = 'evaluate'  # 更改为 'train' 进行训练
    
    # 实验相关配置
    DATA_PATH = "./ml-1m"  # 数据路径
    OUTPUT_DIR = "experiment_results"  # 输出目录
    EXPERIMENT_NAME = None  # 实验名称（自动生成或指定）
    
    # 训练配置
    TRAINING = {
        'epochs': 50,
        'batch_size': 128,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'use_amp': True,  # 使用混合精度训练
    }
    
    # 评估配置
    EVALUATION = {
        'test_size': 0.2,
        'random_seed': 42,
        'models_to_evaluate': ['deep','baseline','transformer'],  # 要评估的模型
    }
    
    # 数据预处理配置
    DATA_PREPROCESSING = {
        'min_ratings_per_user': 5,
        'min_ratings_per_movie': 10,
    }
    
    # 模型配置
    MODELS = {
        'deep': {
            'embedding_dim': 64,
            'hidden_dim': 128,
        },
        'baseline': {
            'embedding_dim': 64,
        },
# ========== 新增 Transformer 模型配置 ==========
        'transformer': {
            'embedding_dim': 64,      # 嵌入维度
            'nhead': 4,               # 多头注意力头数
            'num_layers': 2,          # 编码器层数
            'dim_feedforward': 256,   # 前馈网络维度
            'dropout': 0.1,           # Dropout率
        }
    }

    @classmethod
    def setup_experiment(cls, mode=None):
        """设置实验配置"""
        if mode:
            cls.MODE = mode
        
        # 如果是训练模式，创建新的实验名称
        if cls.MODE == 'train':
            if not cls.EXPERIMENT_NAME:
                cls.EXPERIMENT_NAME = f'exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            print(f"训练模式 - 创建新实验: {cls.EXPERIMENT_NAME}")
        
        # 如果是评估模式，需要指定实验名称
        elif cls.MODE == 'evaluate':
            if not cls.EXPERIMENT_NAME:
                # 获取最新的实验文件夹
                experiments = [d for d in os.listdir(cls.OUTPUT_DIR) 
                             if os.path.isdir(os.path.join(cls.OUTPUT_DIR, d)) and d.startswith('exp_')]
                if experiments:
                    cls.EXPERIMENT_NAME = sorted(experiments)[-1]  # 使用最新的实验
                    print(f"评估模式 - 使用最新实验: {cls.EXPERIMENT_NAME}")
                else:
                    raise ValueError("没有找到实验文件夹，请指定 EXPERIMENT_NAME")
        
        # 创建实验目录
        cls.EXPERIMENT_PATH = os.path.join(cls.OUTPUT_DIR, cls.EXPERIMENT_NAME)
        if cls.MODE == 'train':
            os.makedirs(cls.EXPERIMENT_PATH, exist_ok=True)
            subdirs = ['models', 'plots', 'reports', 'predictions']
            for subdir in subdirs:
                os.makedirs(os.path.join(cls.EXPERIMENT_PATH, subdir), exist_ok=True)
        
        return cls.get_config_dict()
    
    @classmethod
    def get_config_dict(cls):
        """获取配置字典"""
        return {
            'mode': cls.MODE,
            'data_path': cls.DATA_PATH,
            'experiment_name': cls.EXPERIMENT_NAME,
            'experiment_path': cls.EXPERIMENT_PATH,
            'training': cls.TRAINING,
            'evaluation': cls.EVALUATION,
            'data_preprocessing': cls.DATA_PREPROCESSING,
            'models': cls.MODELS,
        }
    
    @classmethod
    def save_config(cls, path=None):
        """保存配置到文件"""
        if not path:
            path = os.path.join(cls.EXPERIMENT_PATH, 'config.json')
        
        import json
        config_dict = cls.get_config_dict()
        config_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"配置已保存到: {path}")
    
    @classmethod
    def load_config(cls, config_file):
        """从文件加载配置"""
        import json
        with open(config_file, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # 更新配置
        for key, value in config_dict.items():
            if hasattr(cls, key.upper()):
                setattr(cls, key.upper(), value)
            elif key in ['training', 'evaluation', 'data_preprocessing', 'models']:
                getattr(cls, key.upper()).update(value)
        
        cls.EXPERIMENT_PATH = os.path.join(cls.OUTPUT_DIR, cls.EXPERIMENT_NAME)
        return config_dict