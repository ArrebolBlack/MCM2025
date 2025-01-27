import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Athlete Probability Integration Network (APINet)
class APINet(nn.Module):

    def __init__(self, apm_model, regression_model):
        super(APINet, self).__init__()
        self.APM = apm_model
        self.regression_model = regression_model

    def forward(self, x):
        # x.shape: [batch_size, seq_len, feature_dim]
        # 运动员概率预测
        if isinstance(self.APM, nn.Module):
            # TODO: 模型怎么是训练状态
            self.APM.eval()
            pred = self.APM(x)
            raise KeyError

        else:
            # 假设 APM 是一个来自 Sklearn/XGBoost 的模型
            # 将 PyTorch 张量转换为 NumPy 数组
            x_np = x.detach().cpu().numpy()
            # 调整 x_np 的形状以匹配 APM 的预期输入形状
            batch_size, seq_len, feature_dim = x_np.shape
            x_np_flat = x_np.reshape(-1, feature_dim)  # 展平为 (n_samples, n_features)
            # print("APM 的展平输入形状:", x_np_flat.shape) -> (n_samples, 7)

            # 进行概率预测   无、铜、银、金
            pred_np_flat = self.APM.predict_proba(x_np_flat)
            # print("展平预测结果形状:", pred_np_flat.shape) -> (n_samples, 4)

            # 确定结果维度
            # 分类模型，0,1,2,3  输出概率
            result_dim = pred_np_flat.shape[1] if len(pred_np_flat.shape) > 1 else 1

            # 将预测结果的形状调整回 (batch_size, seq_len, result_dim)
            pred_np = pred_np_flat.reshape(batch_size, seq_len, result_dim)

            # 将预测结果转换回 PyTorch 张量
            pred = torch.tensor(pred_np, dtype=torch.float32, device=x.device)

        # 聚合数据！
        # 示例处理：对序列长度维度进行求和
        team_pred = pred.sum(dim=1) # (batchsize, 4) #TODO:
        # print(team_pred)

        # 进行回归！
        if isinstance(self.regression_model, nn.Module):
            out = self.regression_model(team_pred)
        else:
            # 将 PyTorch 张量转换为 NumPy 数组
            team_pred_np = team_pred.detach().cpu().numpy()
            print("团队预测 NumPy 形状:", team_pred_np.shape)

            # 进行预测
            out_np = self.regression_model.predict(team_pred_np)
            print("回归预测结果形状:", out_np.shape)

            # 将预测结果转换回 PyTorch 张量
            out = torch.tensor(out_np, dtype=torch.float32, device=x.device)
        # print("输出形状:", out.shape)  # (batchsize, 2)
        return out


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MLPWithEmbeddings(nn.Module):
    def __init__(self, num_noc, num_sport, output_size):
        super(MLPWithEmbeddings, self).__init__()
        self.noc_embedding = nn.Embedding(num_embeddings=num_noc, embedding_dim=8)  # 假设8维嵌入
        self.sport_embedding = nn.Embedding(num_embeddings=num_sport, embedding_dim=8)

        # 其余特征的全连接层
        self.fc1 = nn.Linear(8 + 8 + 5, 128)  # 5个数值特征 + 16个嵌入特征
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x_noc, x_sport, x_numeric):
        noc_embedded = self.noc_embedding(x_noc)
        sport_embedded = self.sport_embedding(x_sport)
        x = torch.cat([noc_embedded, sport_embedded, x_numeric], dim=1)

        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class SharedMLP(nn.Module):
    def __init__(self):
        super(SharedMLP, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU()
        )
        self.output1 = nn.Linear(8, 1)
        self.output2 = nn.Linear(8, 1)

    def forward(self, x):
        shared_out = self.shared_layers(x)
        out1 = self.output1(shared_out)
        out2 = self.output2(shared_out)
        return out1, out2

# 需要将类别特征单独处理并传给嵌入层


# # 假设输入特征已经转换为张量 X_train_tensor 和 X_test_tensor
# # 和目标变量为 y_train_tensor 和 y_test_tensor
#
# model = MLP(input_size=7, hidden_size=128, output_size=4)  # 7个输入特征，128个隐藏神经元，4个输出类别
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练过程
# for epoch in range(100):  # 100个epoch
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train_tensor)
#     loss = criterion(outputs, y_train_tensor)
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}')


