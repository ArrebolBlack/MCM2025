import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from q1_2_make_dataset import train_dataset, test_dataset
import torch.nn as nn
import torch.optim as optim
from utils import train, eval
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

# 初始化 W&B
wandb.init(project="MCM2025Q1", entity="tbsi", name="APINet")

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # 每个 batch 处理一个 (n, 7) 的矩阵
test_loader = DataLoader(test_dataset, batch_size=1)

# 定义模型
APM_list = ["XGBoost", "Random Forest", "MLP"]
Regression_model_list = ["MLP482"]

xgboost = joblib.load('xgb_model.pkl')
from dl_model import APINet, MLP
rg_model = MLP(4, 8, 2)
model = APINet(apm_model=xgboost, regression_model=rg_model)

learning_rate = 0.001

# criterion = nn.MSELoss()
criterion = nn.L1Loss()
# criterion = nn.SmoothL1Loss()
# criterion = nn.HuberLoss()


optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 定义余弦退火学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

# 训练和评估
train(model, train_loader, criterion, optimizer, num_epochs=20, device='cuda' if torch.cuda.is_available() else 'cpu', save_path="checkpoints/", scheduler=scheduler, task_type="regression")
eval(model, test_loader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu', task_type="regression")

wandb.finish()