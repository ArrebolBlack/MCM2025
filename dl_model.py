import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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
