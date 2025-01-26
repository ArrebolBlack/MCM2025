import pandas as pd
import numpy as np
import joblib
from read_data import year_list

# 文件路径设置
path_prefix = "E:/25MCM/"
# YC_data_path = path_prefix + "YearCountry-host-medal.csv"
# # YC_data = pd.read_csv(YC_data_path)
# #
# # q1_data_df = pd.read_csv("q1_dataset.csv")
# #
# # # 合并两个 DataFrame
# # # 'left_on' 和 'right_on' 参数用于指定两个 DataFrame 的合并列
# # merged_df = pd.merge(q1_data_df,
# #                      YC_data[['year', 'country', 'gold', 'total_medal']],
# #                      left_on=['Year', 'NOC'],
# #                      right_on=['year', 'country'],
# #                      how='left')
# #
# # # 删除多余的 'year' 和 'country' 列，因为它们已经在 q1_data_df 中以 'Year' 和 'NOC' 名称存在
# # merged_df.drop(columns=['year', 'country'], inplace=True)
# #
# # print("合并后的 DataFrame：")
# # print(merged_df.head())
# # print(merged_df.info())
# #
# # q1_data_df = merged_df
# # q1_data_df.to_csv('q1_dataset_new.csv')
#################################################################
# q1_data_df =pd.read_csv("q1_dataset_new.csv")
# # print(q1_data_df.info())
# ###########################################################################################
# # 导入 Q1_label_encoder
# NOC_label_encoder_path = path_prefix + "NOC_label_encoder.pkl"
# Sex_label_encoder_path = path_prefix + "Sex_label_encoder.pkl"
# Sport_label_encoder_path = path_prefix + "Sport_label_encoder.pkl"
#
# # 使用 joblib 导入已保存的 label encoders
# label_encoders = {
#     # 'NOC': joblib.load(NOC_label_encoder_path),
#     'Sport': joblib.load(Sport_label_encoder_path),
#     'Sex': joblib.load(Sex_label_encoder_path)
# }
#
# # 编码类别特征
# for col in ['Sport', 'Sex']:
#     le = label_encoders[col]
#     q1_data_df[col] = le.transform(q1_data_df[col])
#
# # 编码目标变量 `Medal` 和 `LastMedal`
# medal_map = {"No medal": 0, "Bronze": 1, "Silver": 2, "Gold": 3}
# q1_data_df['Medal'] = q1_data_df['Medal'].map(medal_map)
# q1_data_df['LastMedal'] = q1_data_df['LastMedal'].map(medal_map)

# 特征和目标
features = ['NOC', 'Sport', 'isHost', 'TeamNum', 'AvgGoldRate', 'AvgMedalRate', 'LastMedal']
target = ['gold', 'total_medal']

# # 导入 Q1_scaler
# Q1_scaler_path = path_prefix + "Q1_scaler.pkl"
# scaler = joblib.load(Q1_scaler_path)
#
# # 导入 NOC 的 LabelEncoder
# le_noc = joblib.load(NOC_label_encoder_path)
#
# # 初始化数据列表
# data = []
#
# # 遍历年份和国家
# for year in year_list:
#     # 获取该年参加奥运会的国家
#     noc_list = q1_data_df.loc[q1_data_df['Year'] == year]['NOC'].unique()
#     for noc in noc_list:
#         temp_data_df = q1_data_df.loc[(q1_data_df['Year'] == year) & (q1_data_df['NOC'] == noc)]
#         temp_X = temp_data_df[features].copy()
#         temp_X['NOC'] = le_noc.transform(temp_X['NOC'])
#         # 标准化 temp_X
#         # 转化为Numpy数组
#         temp_X_scaled = scaler.transform(temp_X).astype(np.float32)
#         # print(temp_X_scaled.dtype)  # (n, 7)   numpy.ndarray  float32
#         # print(type(temp_X_scaled))
#                     # # 获取该年该国的金牌和奖牌总数
#                     # gold_total = YC_data.loc[(YC_data['year'] == year) & (YC_data['country'] == noc), ['gold', 'total_medal']]
#                     # if not gold_total.empty:
#                     #     gold, total = gold_total.iloc[0]['gold'], gold_total.iloc[0]['total_medal']
#                     # else:
#                     #     gold, total = 0, 0  # 如果没有数据，默认值
#
#         gold_total = q1_data_df.loc[(q1_data_df['Year'] == year) & (q1_data_df['NOC'] == noc), ['gold', 'total_medal']]
#         if not gold_total.empty:
#             gold, total_medal = gold_total.iloc[0]['gold'], gold_total.iloc[0]['total_medal']
#         else:
#             gold, total_medal = 0, 0  # 如果没有数据，默认值
#         # 将数据附加到列表中
#         data.append([year, noc, temp_X_scaled, gold, total_medal])
#
#
# # 将数据转换为 DataFrame
# data_df = pd.DataFrame(data, columns=['Year', 'NOC', 'Features', 'gold', 'total_medal'])
#
#
#                         # 示例：如何使用这个数据集
#                         # # 通过 year 和 noc 查询某个国家某年的数据
#                         # year_example = 2020
#                         # noc_example = "USA"  # 使用原始字符串值
#                         # record = data_df[(data_df['Year'] == year_example) & (data_df['NOC'] == noc_example)]
#                         # print(record)
#
# # 随机打乱数据集
# shuffled_data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # # 保存数据集为 Pickle 格式以便后续使用
# shuffled_data_df.to_pickle("Q1_2_processed_data.pkl")

shuffled_data_df = pd.read_pickle("q1_2_processed_data.pkl")
print(shuffled_data_df.info())

# # 输出数据结构示例
# for index, row in data_df.iterrows():
#     print(f"Year: {row['Year']}, NOC: {row['NOC']}, Features shape: {row['Features'].shape}, gold: {row['gold']}, total_medal: {row['total_medal']}")
#     break  # 仅打印第一项

# 在这里，我们不再直接将 Features 列作为 X，因为它们是不同形状的矩阵
# 需要单独处理 Features 矩阵以保证处理过程的一致性

# 创建 PyTorch 数据加载器
feature_list = [row['Features'] for _, row in shuffled_data_df.iterrows()]
gold_total_list = shuffled_data_df[['gold', 'total_medal']].values


from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

class Q1_TotalDataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return feature, target


# 划分训练和测试集
train_features, test_features, train_targets, test_targets = train_test_split(feature_list, gold_total_list, test_size=0.2, random_state=42)

# 创建数据集和数据加载器
train_dataset = Q1_TotalDataset(train_features, train_targets)
test_dataset = Q1_TotalDataset(test_features, test_targets)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # 每个 batch 处理一个 (n, 7) 的矩阵
test_loader = DataLoader(test_dataset, batch_size=1)

# TODO：问题，机器学习方法预测的奖牌数不准

if "__name__" == "__main__":
    xgboost = joblib.load('xgb_model.pkl')
    from dl_model import APINet, MLP

    rg_model = MLP(4, 8, 2)

    model = APINet(apm_model=xgboost, regression_model=rg_model)

    # 检查加载器是否工作正常
    for data, target in train_loader:
        print(data.shape, target)
        output = model(data)
        print(output)
        break  # 仅打印第一批数据

#####################################################################################
# Test batch

#TODO:
# 没有批次能力

# # 设置批次大小为你想要测试的大小，例如 5
# batch_size = 5
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)
#
# # 加载 XGBoost 模型和回归模型
# xgboost = joblib.load('xgb_model.pkl')
# rg_model = MLP(4, 8, 2)  # 假设 MLP 是一个合适的 PyTorch 模型
#
# # 创建 APINet 模型实例
# model = APINet(apm_model=xgboost, regression_model=rg_model)
#
# # 检查加载器是否工作正常并测试多个批次数据
# for batch_idx, (data, target) in enumerate(train_loader):
#     print(f"Batch {batch_idx + 1}:")
#     print("Data shape:", data.shape, "Target shape:", target.shape)
#     output = model(data)
#     print("Output:", output)
#     if batch_idx >= 4:  # 测试前 5 个批次
#         break  # 仅打印前 5 批数据
# #

#

