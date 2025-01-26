import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# 加载数据
data = pd.read_csv("q1_dataset.csv")
# columns = ['Name', 'Year', 'NOC', 'Sport', 'Sex', 'isHost', 'TeamNum',
# 'AvgGoldRate', 'AvgMedalRate', 'LastMedal', 'Medal']

# # 1. 数据清洗
# # 检查缺失值
# print("Missing values:\n", data.isnull().sum())
#
# # 填充或删除缺失值
# data.fillna(method='ffill', inplace=True)  # 可以根据需要选择合适的填充方法
print(data.info())
print(data.describe())

# 编码类别特征
label_encoders = {}
for col in ['NOC', 'Sport', 'Sex']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# 编码目标变量 `Medal` 和 `LastMedal`
medal_map = {"No medal": 0, "Bronze": 1, "Silver": 2, "Gold": 3}
data['Medal'] = data['Medal'].map(medal_map)
data['LastMedal'] = data['LastMedal'].map(medal_map)

# 特征和目标
# features = ['Year', 'NOC', 'Sport', 'Sex', 'TeamNum', 'AvgGoldRate', 'AvgMedalRate', 'LastMedal']
features = ['NOC', 'Sport', 'isHost', 'TeamNum', 'AvgGoldRate', 'AvgMedalRate', 'LastMedal']
target = 'Medal'

# 划分训练集和测试集
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将数据转换为 DataFrame 以便保存
X_train_df = pd.DataFrame(X_train_scaled, columns=features)
X_test_df = pd.DataFrame(X_test_scaled, columns=features)
y_train_df = pd.DataFrame(y_train, columns=[target])
y_test_df = pd.DataFrame(y_test, columns=[target])

# 保存处理后的数据
X_train_df.to_csv("Q1_X_train.csv", index=False)
X_test_df.to_csv("Q1_X_test.csv", index=False)
y_train_df.to_csv("Q1_y_train.csv", index=False)
y_test_df.to_csv("Q1_y_test.csv", index=False)

# 保存 LabelEncoders 和 Scaler 以便后续使用
import joblib
for col, le in label_encoders.items():
    joblib.dump(le, f"{col}_label_encoder.pkl")

joblib.dump(scaler, "Q1_scaler.pkl")



