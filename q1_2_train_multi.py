import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset
from q1_2_make_dataset import train_dataset, test_dataset
import torch.nn as nn
import torch.optim as optim
from utils import train, eval
from torch.optim.lr_scheduler import CosineAnnealingLR
from dl_model import APINet, MLP
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# 定义设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)


# 定义模型选择器
def create_apm_model(model_name):
    if model_name == "XGBoost":
        return joblib.load('xgb_model.pkl')
    elif model_name == "Random Forest":
        return RandomForestClassifier()  # 假设已经训练好
    elif model_name == "MLP":
        return MLP(7, 16, 4)  # 输入特征为 7，隐藏层为 16，输出 4 类


def create_regression_model(model_name):
    if model_name == "MLP482":
        return MLP(4, 8, 2)  # 输入为 4，隐藏层为 8，输出为 2
    # 可以在这里添加更多模型
    raise ValueError(f"Unknown regression model: {model_name}")


# 自动化实验设置
APM_choices = ["XGBoost", "Random Forest", "MLP"]
Regression_model_choices = ["MLP482"]

# 选择模型组合
for apm_name in APM_choices:
    for reg_name in Regression_model_choices:
        print(f"Running experiment with APM: {apm_name}, Regression: {reg_name}")

        # 创建模型实例
        apm_model = create_apm_model(apm_name)
        regression_model = create_regression_model(reg_name)
        model = APINet(apm_model=apm_model, regression_model=regression_model).to(device)

        # 设置学习参数
        learning_rate = 0.001
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

        # 训练和评估
        train(model, train_loader, criterion, optimizer, num_epochs=10, device=device,
              save_path=f"model_{apm_name}_{reg_name}", scheduler=scheduler)
        eval(model, test_loader, criterion, device=device)

