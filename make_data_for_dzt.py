import pandas as pd
import numpy as np
import joblib
from read_data import year_list

from tqdm import tqdm
from read_data import athlete_data, program_data

shuffled_data_df = pd.read_pickle("q1_2_processed_data.pkl")
print(shuffled_data_df.info())

q1_data_df =pd.read_csv("q1_dataset_new.csv")
print(q1_data_df.info())

features = ['Year', 'NOC', 'Sport', 'isHost', 'TeamNum', 'AvgGoldRate', 'AvgMedalRate']

dzt_dataset = q1_data_df[features]
print(dzt_dataset.info())

# Year x NOC x Sport
data = []

for year in tqdm(year_list, desc='Year Loop'):
    # 该年 有哪些国家参加奥运会
    noc_list = athlete_data.loc[athlete_data['Year'] == year]['NOC'].unique()
    for noc in tqdm(noc_list, desc=f'NOC Loop {year}', leave=False):
        # 该年 该国  参加了哪些项目
        program_list = athlete_data.loc[(athlete_data['Year'] == year) & (athlete_data['NOC'] == noc)]['Sport'].unique()
        # print(program_list)
        for program in program_list:
            if program is None: #['SWM' 'WRF' 'ATH' 'SHO' 'MPN' 'CRD' 'SAL' 'EDR' 'DIV' 'TEN' 'GAR' None] 很奇怪
                continue
            else:
                # 该年 该国 该项目 有哪些人参加
                # 转化为DataFrame，逐行处理：将一个人多项目视为多个人，每人只打一个项目
                name_df = athlete_data.loc[(athlete_data['Year'] == year) & (athlete_data['NOC'] == noc) & (athlete_data['Sport'] == program)]

                sport_num = program_data.loc[(program_data['Code'] == program), [str(year)]]
                if sport_num.empty:
                    raise ValueError(f"No data found for program code '{program}' and year '{year}'.")
                sport_num_value = sport_num.iloc[0, 0]

                medal_sport, gold_sport = 0, 0
                for medal in name_df['Medal']:
                    if medal == "Gold":
                        gold_sport += 1
                        medal_sport += 1
                    elif medal == "Silver" or medal == "Bronze":
                        medal_sport += 1
                temp_df = q1_data_df.loc[(q1_data_df['Year'] == year) & (q1_data_df['NOC'] == noc) & (q1_data_df['Sport'] == program)].head(1)
                isHost = temp_df['isHost'].values[0] if not temp_df.empty else None
                TeamNum = temp_df['TeamNum'].values[0] if not temp_df.empty else None
                AvgGoldRate = temp_df['AvgGoldRate'].values[0] if not temp_df.empty else None
                AvgMedalRate = temp_df['AvgMedalRate'].values[0] if not temp_df.empty else None
                if (isHost is None) or (TeamNum is None) or (AvgGoldRate is None) or (AvgMedalRate is None):
                    raise ValueError

                data.append([year, noc, program, sport_num_value, isHost, TeamNum, AvgGoldRate, AvgMedalRate, medal_sport, gold_sport])

columns = ["Year", "NOC", "Sport", "sport_num", "isHost", "TeamNum", "AvgGoldRate", "AvgMedalRate", "medal_sport", "gold_sport"]
dzt_dataset_df = pd.DataFrame(data, columns=columns)
print(dzt_dataset_df.info())
print(dzt_dataset_df.describe())


# print(dzt_dataset.info())
# # # # 'left_on' 和 'right_on' 参数用于指定两个 DataFrame 的合并列
# dzt_dataset = pd.merge(dzt_dataset,
#                      temp_data_df,
#                      left_on=['Year', 'NOC'],
#                      right_on=['Year', 'NOC'],
#                      how='left')

dzt_dataset_df.to_csv("dzt_dataset.csv")