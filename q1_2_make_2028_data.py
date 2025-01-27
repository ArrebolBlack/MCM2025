import pandas as pd
import numpy as np
from read_data import athlete_data, program_code_mapping
import joblib


path_prefix = "E:/25MCM/"
YC_data_path = path_prefix + "YearCountry-host-medal.csv"
YC_data = pd.read_csv(YC_data_path)



# 读取 CSV 文件
q1_dataset_new_df = pd.read_csv("q1_dataset_new.csv")
# print(q1_dataset_new_df.info())

# 提取 2020 年和 2024 年的数据
q1_2020_df = q1_dataset_new_df.loc[q1_dataset_new_df['Year'] == 2020].copy()
q1_2024_df = q1_dataset_new_df.loc[q1_dataset_new_df['Year'] == 2024].copy()

# 获取 2024 年中的唯一国家代码列表
country_2024_list = set(q1_2024_df['NOC'].unique())

# 假设 2028 年国家变化
# 移除 'AIN'，添加 'BLR' 和 'ROC'
country_2028 = (country_2024_list - {'AIN'}) | {'BLR', 'ROC'}
#TODO: 问题，如果这么做，没有运动员信息，后续会为空。检查2020年有没有BLR，ROC
# 20年有BLR，ROC ，底下做特殊情况处理吧
athlete_data_origin = pd.read_csv(path_prefix + "2025_Problem_C_Data/2025_Problem_C_Data\summerOly_athletes_origin.csv")
athlete_data_origin['Sport'] = athlete_data_origin['Sport'].replace(program_code_mapping)
temp_df = athlete_data_origin.loc[
    (athlete_data_origin['Year'] == 2020) & (athlete_data_origin['NOC'].isin(['BLR', 'ROC']))
]
temp_df = temp_df[temp_df['Sport'] != 'BOX']

# temp_df: Name Sex Team NOC Year City Sport Event Medal
temp_df = temp_df[['Name', 'Year', 'NOC', 'Sport', 'Sex', 'Medal']]
new_columns = ['isHost', 'TeamNum', 'AvgGoldRate', 'AvgMedalRate', 'LastMedal', 'gold', 'total_medal']
for col in new_columns:
    if col == 'LastMedal':
        temp_df[col] = 'No medal'
    else:
        temp_df[col] = 0
# q1_2028_df: Name Year NOC Sport Sex isHost TeamNum AvgGoldRate AvgMedalRate LastMedal Medal gold total_medal

print(temp_df.info())
print(temp_df['Sport'].unique())
print(temp_df['NOC'].unique())


# athlete_data_copy = pd.concat([athlete_data, temp_df], axis=0)
# print(athlete_data_copy.info())
print("2028 年参赛国家：", country_2028)

# 删除 'Sport' 列中包含 'BOX' 的行
q1_2024_df = q1_2024_df[q1_2024_df['Sport'] != 'BOX']
q1_2020_df = q1_2020_df[q1_2020_df['Sport'] != 'BOX']

# 创建 2028 年的数据，过滤 2024 年的数据以只保留 2028 年参赛的国家
q1_2028_df = q1_2024_df[q1_2024_df['NOC'].isin(country_2028)].copy()
# TODO: q1_2028_df 没有 ROC，BLR！
q1_2028_df = pd.concat([q1_2028_df, temp_df], axis=0)

print(q1_2028_df.info())

# 将 'isHost' 列的所有值清零
q1_2028_df['isHost'] = 0
q1_2028_df['gold'] = 0
q1_2028_df['total_medal'] = 0
q1_2028_df['Year'] = 2028

# 更新 'isHost' 列，将 'USA' 标记为主办国
q1_2028_df.loc[q1_2028_df['NOC'] == 'USA', 'isHost'] = 1

# 在 q1_2028_df 中计算 AvgGoldRate 和 AvgMedalRate
q1_2028_df['AvgGoldRate'] = 0.0
q1_2028_df['AvgMedalRate'] = 0.0

for noc in country_2028:
    if noc == 'ROC' or noc == 'BLR':
        program_list = temp_df.loc[temp_df['NOC'] == noc]['Sport'].unique()
        print("NOC == :", noc)
        print("Sport:", program_list)
        # ===============================================================================
        for program in program_list:
            # print(noc,program)
            # print(athlete_data_copy.info())
            past_medals_sport = temp_df.loc[
                (temp_df['Year'] == 2020) &  # 仅取上两届
                # (athlete_data['Year'] <= 2024) &  # 早于当前年份
                (temp_df['NOC'] == noc) &  # 国家匹配
                (temp_df['Sport'] == program)  # 运动项目匹配
                ]
            if not past_medals_sport.empty:  # 有历史记录
                gold_num = (past_medals_sport['Medal'] == 'Gold').sum()
                medal_num = (past_medals_sport['Medal'] != 'No medal').sum()
                team_num_2 = len(past_medals_sport)
            else:
                print("from YC_data")
                spe = YC_data.loc[(YC_data['year'] == 2020) & (YC_data['country'] == noc)]
                print(spe.info())
                print(spe.describe())
                gold_num = YC_data.loc[(YC_data['year'] == 2020) &
                                       # (YC_data['year'] <= 2024) &
                                       (YC_data['country'] == noc)]['gold'].sum()
                medal_num = YC_data.loc[(YC_data['year'] == 2020) &
                                        # (YC_data['year'] <= 2024) &
                                        (YC_data['country'] == noc)]['total_medal'].sum()
                team_num_2 = len(athlete_data.loc[
                                     (athlete_data['Year'] == 2020) &  # 仅取上两届
                                     # (athlete_data['Year'] <= 2024) &  # 早于当前年份
                                     (athlete_data['NOC'] == noc)  # 国家匹配
                                     ])
                print(team_num_2)
            avgGoldRate = gold_num / team_num_2 if team_num_2 > 0 else 0
            avgMedalRate = medal_num / team_num_2 if team_num_2 > 0 else 0
            # print("(((******(((")
            print(avgGoldRate, avgMedalRate)
            q1_2028_df.loc[(q1_2028_df['Year'] == 2028) &
                           (q1_2028_df['NOC'] == noc) &
                           (q1_2028_df['Sport'] == program), ['TeamNum', 'AvgGoldRate',
                                                              'AvgMedalRate']] = team_num_2, avgGoldRate, avgMedalRate
    # ===========================================================================
    else:
        program_list = q1_2024_df.loc[q1_2024_df['NOC'] == noc]['Sport'].unique()
        #===============================================================================
        for program in program_list:
            # print(noc,program)
            # print(athlete_data_copy.info())
            past_medals_sport = athlete_data.loc[
                (athlete_data['Year'] >= 2020) &  # 仅取上两届
                (athlete_data['Year'] <= 2024) &  # 早于当前年份
                (athlete_data['NOC'] == noc) &  # 国家匹配
                (athlete_data['Sport'] == program)  # 运动项目匹配
            ].sort_values(by='Year', ascending=False)
            if not past_medals_sport.empty:  # 有历史记录
                gold_num = (past_medals_sport['Medal'] == 'Gold').sum()
                medal_num = (past_medals_sport['Medal'] != 'No medal').sum()
                team_num_2 = len(past_medals_sport)
            else:
                gold_num = YC_data.loc[(YC_data['year'] >= 2020) &
                                       (YC_data['year'] <= 2024) &
                                       (YC_data['country'] == noc)]['gold'].sum()
                medal_num = YC_data.loc[(YC_data['year'] >= 2020) &
                                        (YC_data['year'] <= 2024) &
                                        (YC_data['country'] == noc)]['total_medal'].sum()
                team_num_2 = len(athlete_data.loc[
                                     (athlete_data['Year'] >= 2020) &  # 仅取上两届
                                     (athlete_data['Year'] <= 2024) &  # 早于当前年份
                                     (athlete_data['NOC'] == noc)  # 国家匹配
                                     ])
            avgGoldRate = gold_num / team_num_2 if team_num_2 > 0 else 0
            avgMedalRate = medal_num / team_num_2 if team_num_2 > 0 else 0
            q1_2028_df.loc[(q1_2028_df['Year'] == 2028) &
                           (q1_2028_df['NOC'] == noc) &
                           (q1_2028_df['Sport'] == program), ['AvgGoldRate', 'AvgMedalRate']] = avgGoldRate, avgMedalRate
#===========================================================================
# 获取第 0 和第 1 列的列名
columns_to_drop = q1_2028_df.columns[[0, 1]] # Name也删了
# 使用 drop 方法删除这些列
q1_2028_df = q1_2028_df.drop(columns=columns_to_drop)
print(q1_2028_df.info())
print(q1_2028_df.iloc[:5, 4:-2])
print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
print(len(q1_2028_df.loc[(q1_2028_df['NOC'] == 'BLR')]))
print(len(q1_2028_df.loc[(q1_2028_df['NOC'] == 'ROC')]))
print(q1_2028_df.loc[(q1_2028_df['NOC'] == 'BLR')])
print(q1_2028_df.loc[(q1_2028_df['NOC'] == 'ROC')])
#############################################
q1_2028_df.to_csv('q1_2028_df.csv')
q1_2028_df = pd.read_csv('q1_2028_df.csv')
# 以上更新完2028年数据，以下是转换为ML格式
##############################################################################################

# 导入 Q1_label_encoder
NOC_label_encoder_path = path_prefix + "NOC_label_encoder.pkl"
Sex_label_encoder_path = path_prefix + "Sex_label_encoder.pkl"
Sport_label_encoder_path = path_prefix + "Sport_label_encoder.pkl"

# 使用 joblib 导入已保存的 label encoders
label_encoders = {
    # 'NOC': joblib.load(NOC_label_encoder_path),
    'Sport': joblib.load(Sport_label_encoder_path),
    'Sex': joblib.load(Sex_label_encoder_path)
}

# 编码类别特征
for col in ['Sport', 'Sex']:
    le = label_encoders[col]
    q1_2028_df[col] = le.transform(q1_2028_df[col])

# 编码目标变量 `Medal` 和 `LastMedal`
medal_map = {"No medal": 0, "Bronze": 1, "Silver": 2, "Gold": 3}
q1_2028_df['Medal'] = q1_2028_df['Medal'].map(medal_map)
q1_2028_df['LastMedal'] = q1_2028_df['LastMedal'].map(medal_map)

# # 特征和目标
features = ['NOC', 'Sport', 'isHost', 'TeamNum', 'AvgGoldRate', 'AvgMedalRate', 'LastMedal']
target = ['gold', 'total_medal']


# 导入 Q1_scaler
Q1_scaler_path = path_prefix + "Q1_scaler.pkl"
scaler = joblib.load(Q1_scaler_path)

# 导入 NOC 的 LabelEncoder
le_noc = joblib.load(NOC_label_encoder_path)
# 假设 le_noc 是已经训练好的 LabelEncoder 对象
# 这里确保 'ROC' 不在 le_noc.classes_ 中
if 'ROC' not in le_noc.classes_:
    # 手动扩展 le_noc.classes_
    le_noc.classes_ = np.append(le_noc.classes_, 'ROC')
    joblib.dump(le_noc, f"NOC_label_encoder.pkl")
# 初始化数据列表
data = []
year = 2028

for noc in country_2028:
    # if noc == 'ROC' or noc == "BLR":
    temp_data_df = q1_2028_df.loc[(q1_2028_df['Year'] == year) & (q1_2028_df['NOC'] == noc)]
    temp_X = temp_data_df[features].copy()
    # 利用RUS之前的训练的权重。如果直接用没出现过的ROC，无法学习到俄罗斯之前的特征

    # 将 'ROC' 替换为 'RUS' 在 temp_X['NOC'] 列中
    temp_X['NOC'] = temp_X['NOC'].replace('ROC', 'RUS') # APM , 100 year  RUS

    temp_X['NOC'] = le_noc.transform(temp_X['NOC'])

    # 标准化 temp_X
    # 转化为Numpy数组
    temp_X_scaled = scaler.transform(temp_X).astype(np.float32)
    # print(temp_X_scaled.dtype)  # (n, 7)   numpy.ndarray  float32
    # print(type(temp_X_scaled))

    gold, total_medal = 0, 0
    # 将数据附加到列表中
    data.append([year, noc, temp_X_scaled, gold, total_medal])

# 将数据转换为 DataFrame
data_df = pd.DataFrame(data, columns=['Year', 'NOC', 'Features', 'gold', 'total_medal'])

data_df.to_pickle("2028_data_for_apm.pkl")
###########################################################################################

# 在这里，我们不再直接将 Features 列作为 X，因为它们是不同形状的矩阵
# 需要单独处理 Features 矩阵以保证处理过程的一致性

import torch
# 将特征数据转换为 PyTorch 张量列表
feature_list = [torch.tensor(row['Features'], dtype=torch.float32) for _, row in data_df.iterrows()]
gold_total_list = data_df[['gold', 'total_medal']].values


xgboost = joblib.load('xgb_model.pkl')
from dl_model import APINet, MLP
rg_model = MLP(4, 8, 2)
model = APINet(apm_model=xgboost, regression_model=rg_model)
model.load_state_dict(torch.load("checkpoints/_lowest_loss_epoch_10.pth")) # SmoothL1Loss 的 结果
model.to(device='cuda')
model.eval()

# result_df = data[['Year', 'NOC', 'gold', 'total_medal']].copy()
# 创建一个新的 DataFrame 来存储结果
result_df = data_df[['Year', 'NOC']].copy()
result_df['predicted_gold'] = 0.0
result_df['predicted_total_medal'] = 0.0

# 对每个特征进行预测并存储结果
for i, feature in enumerate(feature_list):
    # print(feature.shape)
    feature = feature.unsqueeze(0).to(device='cuda')
    # print(feature.shape)
    out = model(feature)
    # print(out.shape)
    out = out.squeeze(0)
    # # 假设 out 是一个张量，包含预测的金牌和总奖牌数
    result_df.loc[i, 'predicted_gold'] = out[0].item()
    result_df.loc[i, 'predicted_total_medal'] = out[1].item()

# 打印或保存结果
print(result_df)
result_df.to_csv('2028_result.csv')
