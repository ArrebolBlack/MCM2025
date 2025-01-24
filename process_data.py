import pandas as pd
import numpy as np
from read_data import athlete_data, program_data

path_prefix = "E:/25MCM/"
YC_data_path = path_prefix + "YearCountry-host-medal.csv"
YC_data = pd.read_csv(YC_data_path)

# print(YC_data.info())

data = athlete_data
# 查询每列的唯一值
for column in data.columns:
    print(f"列 {column} 的唯一值：")
    print(data[column].unique())

list1 = athlete_data['Sport'].unique()
print(len(list1))

# 构建数据集

# Sex City 可取
year = 2016
noc = 'CHN'
program = "TTE"
temp = athlete_data.loc[(athlete_data['Year'] == year) & (athlete_data['NOC'] == noc) & (athlete_data['Sport'] == program), :]
print(temp)

