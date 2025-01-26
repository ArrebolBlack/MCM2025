import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# 加载处理后的数据集
X_train = pd.read_csv("Q1_X_train.csv")
y_train = pd.read_csv("Q1_y_train.csv").values.ravel()  # 使用 values.ravel() 转换为 1D 数组
X_test = pd.read_csv("Q1_X_test.csv")
y_test = pd.read_csv("Q1_y_test.csv").values.ravel()

# 自动计算类别权重
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
# class_weights = [0.1, 10, 10, 10]
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
sample_weights = np.array([class_weights[y] for y in y_train])
print(class_weights_dict)

# 随机森林
rf_model = RandomForestClassifier(class_weight=class_weights_dict, n_estimators=350, max_depth=30, min_samples_split=12, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# 保存随机森林模型
joblib.dump(rf_model, 'random_forest_model.pkl')

# 加载随机森林模型（示例）
# loaded_rf_model = joblib.load('random_forest_model.pkl')
# y_pred_loaded_rf = loaded_rf_model.predict(X_test)
# print("Loaded Random Forest Accuracy:", accuracy_score(y_test, y_pred_loaded_rf))

# XGBoost
xgb_model = XGBClassifier(n_estimators=300, max_depth=12, learning_rate=0.05, objective='multi:softmax', num_class=len(set(y_train)))
xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
y_pred_xgb = xgb_model.predict(X_test)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("XGBoost Classification Report:\n", classification_report(y_test, y_pred_xgb))

# 保存XGBoost模型
joblib.dump(xgb_model, 'xgb_model.pkl')

# 加载XGBoost模型（示例）
# loaded_xgb_model = joblib.load('xgb_model.pkl')
# y_pred_loaded_xgb = loaded_xgb_model.predict(X_test)
# print("Loaded XGBoost Accuracy:", accuracy_score(y_test, y_pred_loaded_xgb))


            # # SVM
            # svm_model = SVC(kernel='linear', probability=True, random_state=42)
            # svm_model.fit(X_train, y_train)
            # y_pred_svm = svm_model.predict(X_test)
            # print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
            # print("SVM Classification Report:\n", classification_report(y_test, y_pred_svm))
