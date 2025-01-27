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
# wandb.init(project="MCM2025Q1", entity="tbsi")

def train(model, train_loader, criterion, optimizer, num_epochs=100, device='cuda', save_path="checkpoints/",
          scheduler=None, task_type=None):
    assert task_type in ["classification", "regression"], "task_type must be either 'classification' or 'regression'"

    best_metric = 0.0 if task_type == 'classification' else float('inf')
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

                if task_type == 'classification':
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)
                    accuracy = 100. * correct / total
                    pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'accuracy': accuracy})
                else:
                    total += targets.size(0)
                    pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

                pbar.update()

        # 计算每个 epoch 的指标
        epoch_loss = running_loss / len(train_loader)
        if task_type == 'classification':
            epoch_metric = 100. * correct / total
            print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_metric:.2f}%')
        else:
            epoch_metric = epoch_loss
            print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}')

        # 将指标记录到 WandB
        log_data = {'epoch': epoch + 1, 'loss': epoch_loss}
        if task_type == 'classification':
            log_data['accuracy'] = epoch_metric
        wandb.log(log_data)

        if scheduler:
            scheduler.step()  # 更新学习率

        # 保存模型
        if (task_type == 'classification' and epoch_metric > best_metric) or (
                task_type == 'regression' and epoch_metric < best_metric):
            best_metric = epoch_metric
            model_path = f"{save_path}_best_model_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"最佳模型已保存，指标: {epoch_metric:.4f}")


def eval(model, test_loader, criterion, device='cuda', task_type=None):
    assert task_type in ["classification", "regression"], "task_type must be either 'classification' or 'regression'"

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

            if task_type == 'classification':
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
            else:
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(outputs.cpu().numpy())

    loss = running_loss / len(test_loader)
    if task_type == 'classification':
        accuracy = 100. * correct / total
        print(f'评估 - 损失: {loss:.4f}, 准确率: {accuracy:.2f}%')
        print("分类报告:")
        print(classification_report(all_targets, all_predictions))
        wandb.log({'eval_loss': loss, 'eval_accuracy': accuracy})
    else:
        # 计算回归的评估指标
        mse = np.mean((np.array(all_targets) - np.array(all_predictions)) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.array(all_targets) - np.array(all_predictions)))
        ss_total = np.sum((np.array(all_targets) - np.mean(all_targets)) ** 2)
        ss_res = np.sum((np.array(all_targets) - np.array(all_predictions)) ** 2)
        r_squared = 1 - (ss_res / ss_total)

        print(f'评估 - 损失: {loss:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r_squared:.4f}')
        wandb.log({'eval_loss': loss, 'eval_mse': mse, 'eval_rmse': rmse, 'eval_mae': mae, 'eval_r2': r_squared})

    return loss


# 使用示例
# 假设 `train_loader` 和 `test_loader` 已定义，并且 `model` 是你的神经网络

# 示例超参数
# input_size = 7
# hidden_size = 128
# output_size = 4
# learning_rate = 0.001
#
# model = MLP(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
# criterion = nn.CrossEntropyLoss() if task_type == 'classification' else nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # 训练和评估
# train(model, train_loader, criterion, optimizer, num_epochs=10, device='cuda' if torch.cuda.is_available() else 'cpu', save_path="model", task_type='classification')
# eval(model, test_loader, criterion, device='cuda' if torch.cuda.is_available() else 'cpu', task_type='classification')

# 结束 WandB 运行
wandb.finish()
