import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# 加载处理后的数据集
X_train = pd.read_csv("Q1_X_train.csv")
y_train = pd.read_csv("Q1_y_train.csv").values.ravel()  # 使用 values.ravel() 转换为 1D 数组
X_test = pd.read_csv("Q1_X_test.csv")
y_test = pd.read_csv("Q1_y_test.csv").values.ravel()

# 将标准化的数据转换为 PyTorch 张量

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 检查加载器是否工作正常
for data, target in train_loader:
    print(data, target)
    break  # 仅打印第一批数据

#############################################################
import torch.nn as nn
import torch.optim as optim
from dl_model import MLP
from utils import train, eval
from torch.optim.lr_scheduler import CosineAnnealingLR

input_size = 7
hidden_size = 128
output_size = 4
learning_rate = 0.001

model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 定义余弦退火学习率调度器
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

# 训练和评估
train(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu', save_path="model", scheduler=scheduler)
eval(model, test_loader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')
