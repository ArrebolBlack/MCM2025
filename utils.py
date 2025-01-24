import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import os
import wandb

# 初始化 WandB
wandb.init(project="MCM2025Q1", entity="tbsi")

def train(model, train_loader, criterion, optimizer, num_epochs=100, device='cuda', save_path="model", scheduler=None):
    best_accuracy = 0.0
    lowest_loss = float('inf')
    model.to(device)

    # 数据并行
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # tqdm 用于进度条
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()  # 梯度清零
                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, targets)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重

                # 累积指标
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'accuracy': 100. * correct / total})
                pbar.update()

        # 计算每个 epoch 的指标
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

        # 将指标记录到 WandB
        wandb.log({'epoch': epoch + 1, 'loss': epoch_loss, 'accuracy': epoch_accuracy})

        if scheduler:
            scheduler.step()  # 更新学习率

        # 保存模型
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), f"{save_path}_best_accuracy_epoch_{epoch + 1}.pth")
            print(f"最佳准确率模型已保存，准确率: {best_accuracy:.2f}%")

        if epoch_loss < lowest_loss:
            lowest_loss = epoch_loss
            torch.save(model.state_dict(), f"{save_path}_lowest_loss_epoch_{epoch + 1}.pth")
            print(f"最低损失模型已保存，损失: {lowest_loss:.4f}")

def eval(model, test_loader, criterion, device='cuda'):
    model.to(device)
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():  # 禁用梯度计算
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total

    print(f'评估 - 损失: {loss:.4f}, 准确率: {accuracy:.2f}%')
    print("分类报告:")
    print(classification_report(all_targets, all_predictions))

    # 将评估指标记录到 WandB
    wandb.log({'eval_loss': loss, 'eval_accuracy': accuracy})

    return loss, accuracy

# 使用示例
# 假设 `train_loader` 和 `test_loader` 已定义，并且 `model` 是你的神经网络

# # 示例超参数
# input_size = 7
# hidden_size = 128
# output_size = 4
# learning_rate = 0.001
#
# model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # 训练和评估
# train(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu', save_path="model")
# eval(model, test_loader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu')

# 结束 WandB 运行
wandb.finish()
