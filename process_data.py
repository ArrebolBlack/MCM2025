import pandas as pd
import numpy as np
from tqdm import tqdm
from read_data import athlete_data, program_data
from read_data import program_code_list, year_list, country_list

path_prefix = "E:/25MCM/"
YC_data_path = path_prefix + "YearCountry-host-medal.csv"
YC_data = pd.read_csv(YC_data_path)

# print(YC_data.info())

# data = athlete_data
# # 查询每列的唯一值
# for column in data.columns:
#     print(f"列 {column} 的唯一值：")
#     print(data[column].unique())
#
# list1 = athlete_data['Sport'].unique()
# print(len(list1))

# 构建数据集
# 每一条（行）相当于一个样本
# Name, Year, NOC, Sport, Sex, isHost, TeamNum, AvgGoldRate, AvgMedalRate, LastMedal, Medal
data = []
last_year = year_list[0] - 4
last_2_year = last_year
for year in tqdm(year_list, desc='Year Loop'):
    # 该年 有哪些国家参加奥运会
    noc_list = athlete_data.loc[athlete_data['Year'] == year]['NOC'].unique()
    for noc in tqdm(noc_list, desc=f'NOC Loop {year}', leave=False):
        isHost_series = YC_data.loc[(YC_data['year'] == year) & (YC_data['country'] == noc), 'is_host']
        if not isHost_series.empty:  # 如果筛选结果不为空
            isHost = isHost_series.values[0]  # 提取第一行（假设只需要第一行的值）
        else:  # 如果筛选结果为空
            isHost = 0

        # 该年 该国  参加了哪些项目
        program_list = athlete_data.loc[(athlete_data['Year'] == year) & (athlete_data['NOC'] == noc)]['Sport'].unique()
        for program in program_list:
            # 该年 该国 该项目 有哪些人参加
            # 转化为DataFrame，逐行处理：将一个人多项目视为多个人，每人只打一个项目
            name_df = athlete_data.loc[(athlete_data['Year'] == year) & (athlete_data['NOC'] == noc) & (athlete_data['Sport'] == program)]
            TeamNum = len(name_df)
            for name, sex, medal in zip(name_df['Name'], name_df['Sex'], name_df['Medal']):

                # 计算LastMedal
                # 筛选出该运动员在过去年份的记录
                past_medals = athlete_data.loc[
                    (athlete_data['Year'] >= last_2_year) & # 仅取上两届
                    (athlete_data['Year'] < year) &  # 早于当前年份
                    (athlete_data['Name'] == name) &  # 运动员姓名匹配
                    (athlete_data['NOC'] == noc) &  # 国家匹配
                    (athlete_data['Sport'] == program)  # 运动项目匹配
                ].sort_values(by='Year', ascending=False) # 按年份倒序排列
                if not past_medals.empty:  # 如果有历史记录
                    # 按奖牌优先级返回最近的奖牌
                    if 'Gold' in past_medals['Medal'].values:
                        last_medal = 'Gold'
                    elif 'Silver' in past_medals['Medal'].values:
                        last_medal = 'Silver'
                    elif 'Bronze' in past_medals['Medal'].values:
                        last_medal = 'Bronze'
                    else:
                        last_medal = 'No medal'
                else:  # 如果没有历史记录
                    last_medal = 'No medal'

                # 计算AvgGold
                past_medals_sport = athlete_data.loc[
                    (athlete_data['Year'] >= last_2_year) &  # 仅取上两届
                    (athlete_data['Year'] < year) &  # 早于当前年份
                    (athlete_data['NOC'] == noc) &  # 国家匹配
                    (athlete_data['Sport'] == program)  # 运动项目匹配
                ].sort_values(by='Year', ascending=False).head(2)  # 按年份倒序排列, 仅取前两行（相当于上两届）
                if not past_medals_sport.empty: # 有历史记录
                    gold_num = (past_medals_sport['Medal'] == 'Gold').sum()
                    medal_num = (past_medals_sport['Medal'] != 'No medal').sum()
                    team_num_2 = len(past_medals_sport)
                else:
                    gold_num = YC_data.loc[(YC_data['year'] >= last_2_year) &
                                           (YC_data['year'] < year) &
                                           (YC_data['country'] == noc)]['gold'].sum()
                    medal_num = YC_data.loc[(YC_data['year'] >= last_2_year) &
                                           (YC_data['year'] < year) &
                                           (YC_data['country'] == noc)]['total_medal'].sum()
                    team_num_2 = len(athlete_data.loc[
                                            (athlete_data['Year'] >= last_2_year) &  # 仅取上两届
                                            (athlete_data['Year'] < year) &  # 早于当前年份
                                            (athlete_data['NOC'] == noc) # 国家匹配
                                     ])
                avgGoldRate = gold_num / team_num_2 if team_num_2 > 0 else 0
                avgMedalRate = medal_num / team_num_2 if team_num_2 > 0 else 0

                data.append([name, year, noc, program, sex, isHost, TeamNum, avgGoldRate, avgMedalRate, last_medal, medal])
    last_2_year = last_year
    last_year = year


columns = ['Name', 'Year', 'NOC', 'Sport', 'Sex', 'isHost', 'TeamNum', 'AvgGoldRate', 'AvgMedalRate', 'LastMedal', 'Medal']
q1_data_df = pd.DataFrame(data, columns=columns)
print(q1_data_df.info())
print(q1_data_df.describe())
q1_data_df.to_csv("q1_dataset.csv")
# # Sex City 可取
# year = 2016
# noc = 'CHN'
# program = "TTE"
# temp = athlete_data.loc[(athlete_data['Year'] == year) & (athlete_data['NOC'] == noc) & (athlete_data['Sport'] == program), :]
# print(temp)


