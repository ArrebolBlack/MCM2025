import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# 加载数据集
X_train = pd.read_csv("Q1_X_train.csv")
y_train = pd.read_csv("Q1_y_train.csv").values.ravel()
X_test = pd.read_csv("Q1_X_test.csv")
y_test = pd.read_csv("Q1_y_test.csv").values.ravel()

# 随机森林参数网格
rf_param_grid = {
    'n_estimators': [350],
    'max_depth': [30],
    'min_samples_split': [12]
}

# XGBoost参数网格
xgb_param_grid = {
    'n_estimators': [300],
    'max_depth': [12],
    'learning_rate': [0.05]
}

# SVM参数网格
svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}


# 定义模型
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(objective='multi:softmax', num_class=len(set(y_train)), random_state=42)
svm_model = SVC(probability=True, random_state=42)


# 使用GridSearch进行超参数调优
# rf_grid = GridSearchCV(estimator=rf_model, param_grid=rf_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
# xgb_grid = GridSearchCV(estimator=xgb_model, param_grid=xgb_param_grid, cv=3, scoring='accuracy', n_jobs=-1)
svm_grid = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=3, scoring='accuracy', n_jobs=-1)


# 训练和评估每个模型
def evaluate_model(grid, name):
    grid.fit(X_train, y_train)
    y_pred = grid.predict(X_test)
    print(f"{name} Best Params:", grid.best_params_)
    print(f"{name} Best Estimator Accuracy:", grid.best_score_)
    print(f"{name} Test Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))

# 评估所有模型
# evaluate_model(rf_grid, "Random Forest")
# evaluate_model(xgb_grid, "XGBoost")
evaluate_model(svm_grid, "SVM")

