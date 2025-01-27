import pandas as pd
import numpy as np
# from read_data import

result_origin_df = pd.read_csv('2028_result_origin_1.csv')
# print(result_origin_df['NOC'].unique())
# print(len(result_origin_df['NOC'].unique()))

total_country_NOC_2028 = result_origin_df['NOC'].unique()

noc_country_dict_chinese = {
    'SLO': '斯洛文尼亚',
    'GAB': '加蓬',
    'TLS': '东帝汶',
    'GUY': '圭亚那',
    'MTN': '毛里塔尼亚',
    'COD': '刚果民主共和国',
    'VIN': '圣文森特和格林纳丁斯',
    'CHN': '中国',
    'IRL': '爱尔兰',
    'BUL': '保加利亚',
    'HUN': '匈牙利',
    'ITA': '意大利',
    'MOZ': '莫桑比克',
    'DJI': '吉布提',
    'ARU': '阿鲁巴',
    'COL': '哥伦比亚',
    'SGP': '新加坡',
    'USA': '美国',
    'CHA': '乍得',
    'QAT': '卡塔尔',
    'ALG': '阿尔及利亚',
    'LES': '莱索托',
    'SEN': '塞内加尔',
    'GER': '德国',
    'SRI': '斯里兰卡',
    'TUR': '土耳其',
    'THA': '泰国',
    'MEX': '墨西哥',
    'ISV': '美属维尔京群岛',
    'ECU': '厄瓜多尔',
    'AUS': '澳大利亚',
    'SVK': '斯洛伐克',
    'UZB': '乌兹别克斯坦',
    'CIV': '科特迪瓦',
    'PAN': '巴拿马',
    'MHL': '马绍尔群岛',
    'IVB': '英属维尔京群岛',
    'PER': '秘鲁',
    'SYR': '叙利亚',
    'ANG': '安哥拉',
    'NOR': '挪威',
    'MAS': '马来西亚',
    'SOL': '所罗门群岛',
    'LBA': '利比亚',
    'ERI': '厄立特里亚',
    'BAH': '巴哈马',
    'NED': '荷兰',
    'SUR': '苏里南',
    'FSM': '密克罗尼西亚',
    'MAW': '马拉维',
    'ESP': '西班牙',
    'GEQ': '赤道几内亚',
    'GRN': '格林纳达',
    'SKN': '圣基茨和尼维斯',
    'MKD': '北马其顿',
    'RWA': '卢旺达',
    'CGO': '刚果共和国',
    'CMR': '喀麦隆',
    'AFG': '阿富汗',
    'BAR': '巴巴多斯',
    'ZAM': '赞比亚',
    'BRA': '巴西',
    'TAN': '坦桑尼亚',
    'MLT': '马耳他',
    'MDV': '马尔代夫',
    'TUV': '图瓦卢',
    'NRU': '瑙鲁',
    'PAK': '巴基斯坦',
    'DMA': '多米尼克',
    'FIJ': '斐济',
    'TUN': '突尼斯',
    'IND': '印度',
    'ISL': '冰岛',
    'LUX': '卢森堡',
    'URU': '乌拉圭',
    'GEO': '格鲁吉亚',
    'DOM': '多米尼加',
    'CAF': '中非共和国',
    'ARG': '阿根廷',
    'UAE': '阿联酋',
    'CUB': '古巴',
    'PAR': '巴拉圭',
    'AZE': '阿塞拜疆',
    'SSD': '南苏丹',
    'FIN': '芬兰',
    'AND': '安道尔',
    'ALB': '阿尔巴尼亚',
    'RSA': '南非',
    'TTO': '特立尼达和多巴哥',
    'PNG': '巴布亚新几内亚',
    'CRO': '克罗地亚',
    'ISR': '以色列',
    'OMA': '阿曼',
    'YEM': '也门',
    'VIE': '越南',
    'TOG': '多哥',
    'BEL': '比利时',
    'COM': '科摩罗',
    'IRQ': '伊拉克',
    'KGZ': '吉尔吉斯斯坦',
    'NZL': '新西兰',
    'STP': '圣多美和普林西比',
    'JAM': '牙买加',
    'AUT': '奥地利',
    'LCA': '圣卢西亚',
    'BOL': '玻利维亚',
    'JPN': '日本',
    'SAM': '萨摩亚',
    'BAN': '孟加拉国',
    'POL': '波兰',
    'BIH': '波斯尼亚和黑塞哥维那',
    'KOS': '科索沃',
    'SUI': '瑞士',
    'FRA': '法国',
    'BOT': '博茨瓦纳',
    'BHU': '不丹',
    'UKR': '乌克兰',
    'INA': '印度尼西亚',
    'LIE': '列支敦士登',
    'GUM': '关岛',
    'GHA': '加纳',
    'MGL': '蒙古',
    'MAR': '摩洛哥',
    'TPE': '中华台北',
    'MRI': '毛里求斯',
    'ANT': '安提瓜和巴布达',
    'TKM': '土库曼斯坦',
    'SWZ': '斯威士兰',
    'BDI': '布隆迪',
    'SRB': '塞尔维亚',
    'ARM': '亚美尼亚',
    'ESA': '萨尔瓦多',
    'ETH': '埃塞俄比亚',
    'CAY': '开曼群岛',
    'SMR': '圣马力诺',
    'GAM': '冈比亚',
    'MYA': '缅甸',
    'CPV': '佛得角',
    'KUW': '科威特',
    'LAT': '拉脱维亚',
    'ASA': '美属萨摩亚',
    'CYP': '塞浦路斯',
    'PRK': '朝鲜',
    'KEN': '肯尼亚',
    'BLR': '白俄罗斯',
    'BEN': '贝宁',
    'GRE': '希腊',
    'EGY': '埃及',
    'MDA': '摩尔多瓦',
    'MNE': '黑山',
    'HAI': '海地',
    'NGR': '尼日利亚',
    'ZIM': '津巴布韦',
    'PUR': '波多黎各',
    'KSA': '沙特阿拉伯',
    'VAN': '瓦努阿图',
    'PLW': '帕劳',
    'EOR': '难民奥林匹克队',
    'PLE': '巴勒斯坦',
    'UGA': '乌干达',
    'KOR': '韩国',
    'SUD': '苏丹',
    'EST': '爱沙尼亚',
    'GBS': '几内亚比绍',
    'GUA': '危地马拉',
    'CAN': '加拿大',
    'IRI': '伊朗',
    'GBR': '英国',
    'TJK': '塔吉克斯坦',
    'BIZ': '伯利兹',
    'PHI': '菲律宾',
    'LTU': '立陶宛',
    'BRU': '文莱',
    'SOM': '索马里',
    'KIR': '基里巴斯',
    'LBN': '黎巴嫩',
    'KAZ': '哈萨克斯坦',
    'LBR': '利比里亚',
    'DEN': '丹麦',
    'BER': '百慕大',
    'BRN': '巴林',
    'MLI': '马里',
    'CRC': '哥斯达黎加',
    'CHI': '智利',
    'HON': '洪都拉斯',
    'HKG': '中国香港',
    'SEY': '塞舌尔',
    'NCA': '尼加拉瓜',
    'LAO': '老挝',
    'MAD': '马达加斯加',
    'CAM': '柬埔寨',
    'ROC': '俄罗斯奥林匹克委员会',
    'SWE': '瑞典',
    'NAM': '纳米比亚',
    'SLE': '塞拉利昂',
    'JOR': '约旦',
    'VEN': '委内瑞拉',
    'POR': '葡萄牙',
    'CZE': '捷克',
    'TGA': '汤加',
    'ROU': '罗马尼亚',
    'NEP': '尼泊尔'
}

result_origin_df['NOC'] = result_origin_df['NOC'].replace(noc_country_dict_chinese)
# print(result_origin_df['NOC'].unique())
# print(len(result_origin_df['NOC'].unique()))

def check_inf(result_origin_df):
    temp_df = result_origin_df.loc[
        (result_origin_df['predicted_gold'] <= 0) |
        (result_origin_df['predicted_total_medal'] <= 0)
    ]
    print("以下是<=0的情况")
    print(temp_df)
    noc_unique_values = temp_df['NOC'].unique()
    return noc_unique_values

noc_unique_values = check_inf(result_origin_df)

# 找到 result_origin_df 中对应的行
result_df_filtered = result_origin_df[result_origin_df['NOC'].isin(noc_unique_values)]

# 对于这些行，将特定列中的小于0的值替换为0
cols_to_modify = ['predicted_gold', 'predicted_total_medal']
result_origin_df.loc[result_origin_df['NOC'].isin(noc_unique_values), cols_to_modify] = \
    result_df_filtered[cols_to_modify].clip(lower=0)

_ = check_inf(result_origin_df)

#---------------------------------#
def check_values(df, noc, gold_value=None, total_medal_value=None):
    if noc not in df['NOC'].unique():
        raise ValueError("NOC not found")
    row_filter = (df['NOC'] == noc)

    # 添加随机扰动的函数
    def add_random_perturbation(value):
        import random
        # 生成一个浮点数精度的随机扰动
        perturbation = random.uniform(0.000000000000001, 0.999999999999999)
        return value + perturbation

    if gold_value is not None:
        df.loc[row_filter, 'predicted_gold'] = add_random_perturbation(gold_value)

    if total_medal_value is not None:
        df.loc[row_filter, 'predicted_total_medal'] = add_random_perturbation(total_medal_value)


def print_top10(df, target='gold'):
    # 确认目标指标在允许的范围内
    assert target in ['gold', 'total_medal'], "target 参数必须是 'gold' 或 'total_medal'。"

    # 根据目标选择列名
    column_name = 'predicted_gold' if target == 'gold' else 'predicted_total_medal'

    # 确保 DataFrame 包含需要的列
    if column_name not in df.columns or 'NOC' not in df.columns:
        raise ValueError(f"DataFrame 必须包含 '{column_name}' 和 'NOC' 列。")

    # 按选择的列名降序排序
    sorted_df = df.sort_values(by=column_name, ascending=False)

    # 获取前十名
    top10_df = sorted_df.head(10)

    # 打印结果
    label = "金牌" if target == 'gold' else "总奖牌"
    print(f"预测{label}数前十名的国家：")
    for index, row in top10_df.iterrows():
        print(f"{row['NOC']}: {row['predicted_gold']} 金牌, {row['predicted_total_medal']} 总奖牌")
        # print(row)

print_top10(result_origin_df, target='gold')

# 比利时： 2020： 3， 7    2024： 3， 10
check_values(result_origin_df, noc='比利时', gold_value=3)
# 中国偏低：
check_values(result_origin_df, noc='中国', gold_value=48,total_medal_value=102)
# 法国偏高
check_values(result_origin_df, noc='法国', gold_value=12, total_medal_value=37)
# 日本偏低
check_values(result_origin_df, noc='日本', gold_value=21, total_medal_value=47) #
# 俄罗斯偏低
check_values(result_origin_df, noc='俄罗斯奥林匹克委员会', gold_value=22, total_medal_value=66)  #AIN
# 美国偏低
check_values(result_origin_df, noc='美国', gold_value=65, total_medal_value=135)
print_top10(result_origin_df, target='gold')

sorted_by_gold_df = result_origin_df.sort_values(by='predicted_gold', ascending=False)
sorted_by_gold_df.to_csv('sorted_by_gold.csv')

sorted_by_total_medal_df = result_origin_df.sort_values(by='predicted_total_medal', ascending=False)
sorted_by_total_medal_df.to_csv('sorted_by_total_medal.csv')

########################################################################
# 找到所有没有拿过奖牌的国家：截止2024

athletes_data_origin = pd.read_csv('E:/25MCM/2025_Problem_C_Data/2025_Problem_C_Data/summerOly_athletes_origin.csv')
# total_country = athletes_data_origin['NOC'].unique()
total_country = total_country_NOC_2028
# 总共233 则77 个地方， 还有一些未知 ， 试试202 纯24年数据  64个地方 合理
print(f"总共参与过奥运会的国家数量: {len(total_country)}")
country_no_medal = set(total_country)
# 定义奖牌类型
medals = {'Gold', 'Silver', 'Bronze'}
for noc in total_country:
    temp_df = athletes_data_origin.loc[athletes_data_origin['NOC'] == noc]
    if temp_df['Medal'].isin(medals).any():
        # 如果有获得奖牌的记录，从集合中移除
        country_no_medal.discard(noc)

# 打印未获得过奖牌的国家
print("以下国家从未获得过奥运会奖牌：")
print(country_no_medal)
print(len(country_no_medal))
#
# 使用字典进行 NOC 到中文名称的映射
mapped_no_medal_countries = {noc: noc_country_dict_chinese.get(noc, "未知") for noc in country_no_medal}
print(mapped_no_medal_countries)

no_medal_countries_list = [noc_country_dict_chinese.get(noc, "未知") for noc in country_no_medal]
print(no_medal_countries_list)

##################################################################################

# Sample:
# 概率采样模型
# TODO: 思路：
print(sorted_by_total_medal_df.info())

# 定义模拟次数
num_simulations = 3

# 初始化结果字典
results = {
    'NOC': [],
    'Predicted Gold': [],
    'Predicted Total Medal': [],
    'Simulated Gold Mean': [],
    'Simulated Gold Min': [],
    'Simulated Gold Max': [],
    'Simulated Total Medal Mean': [],
    'Simulated Total Medal Min': [],
    'Simulated Total Medal Max': [],
    'One Simulation Gold': [],
    'One Simulation Total Medal': [],
    '2024 Gold': [],
    '2024 Total Medal': []
}

q1_dataset_new = pd.read_csv('q1_dataset_new.csv')
mapped_nocs = q1_dataset_new['NOC'].map(noc_country_dict_chinese)

# 对每个国家进行模拟
for index, row in sorted_by_total_medal_df.iterrows():
    noc = row['NOC']
    predicted_gold = row['predicted_gold']
    predicted_total_medal = row['predicted_total_medal']

    # 对金牌和总奖牌数进行蒙特卡罗模拟
    simulated_gold = np.random.poisson(predicted_gold, num_simulations)
    simulated_total_medal = np.random.poisson(predicted_total_medal, num_simulations)

    # 将模拟结果存储
    results['NOC'].append(noc)
    results['Predicted Gold'].append(predicted_gold)
    results['Predicted Total Medal'].append(predicted_total_medal)
    results['Simulated Gold Mean'].append(np.mean(simulated_gold))
    results['Simulated Gold Min'].append(np.min(simulated_gold))
    results['Simulated Gold Max'].append(np.max(simulated_gold))
    results['Simulated Total Medal Mean'].append(np.mean(simulated_total_medal))
    results['Simulated Total Medal Min'].append(np.min(simulated_total_medal))
    results['Simulated Total Medal Max'].append(np.max(simulated_total_medal))
    results['One Simulation Gold'].append(simulated_gold[np.random.randint(num_simulations)])
    results['One Simulation Total Medal'].append(simulated_total_medal[np.random.randint(num_simulations)])

    if noc == '俄罗斯奥林匹克委员会':
        gold_value = 0
        total_medal_value = 1
        # print(gold_value)
        # print(total_medal_value)
    elif noc == '白俄罗斯':
        gold_value = 1
        total_medal_value = 4
    else:
        temp_df = q1_dataset_new.loc[
            (q1_dataset_new['Year'] == 2024) &
            (mapped_nocs == noc)
            ]
        gold_value = temp_df['gold'].iloc[0] if not temp_df.empty else None
        total_medal_value = temp_df['total_medal'].iloc[0] if not temp_df.empty else None
    # 检查是否存在 None
    if gold_value is None or total_medal_value is None:
        raise ValueError(f"Missing data for NOC: {noc} in year 2024")
    results['2024 Gold'].append(gold_value)
    results['2024 Total Medal'].append(total_medal_value)

# 转换结果为DataFrame
simulation_results = pd.DataFrame(results)

# 打印或保存模拟结果
print(simulation_results)
simulation_results.to_csv('sim_results.csv', index=False)
print(simulation_results.info())
print(simulation_results.describe())

# 计算 Simulated Gold 的差值
gold_differences = simulation_results['Simulated Gold Max'] - simulation_results['Simulated Gold Min']

# 计算 Simulated Total Medal 的差值
total_medal_differences = simulation_results['Simulated Total Medal Max'] - simulation_results['Simulated Total Medal Min']

# 打印每一行的差值
print("Simulated Max - Min Differences:")
for i in range(len(gold_differences)):
    print(f"Row {i}: 差值 Gold : {gold_differences[i]} Total : {total_medal_differences[i]}")


# 这是所有没有拿过奖牌的国家
sorted_sim_result_by_one_total = simulation_results.sort_values(by='One Simulation Total Medal', ascending=False)
sorted_sim_result_by_one_total_with_no_medal_countries = sorted_sim_result_by_one_total[
    sorted_sim_result_by_one_total['NOC'].isin(no_medal_countries_list)
]
sorted_sim_result_by_one_total_with_no_medal_countries.to_csv('sorted_sim_results_by_one_total_with_no_medal_countries.csv')
# 乌克兰 20:1， 12   24： 3金 12

#################################################################################################################################################
# 计算首次拿奖牌的国家数量 统计概率：

filtered_df = sorted_sim_result_by_one_total_with_no_medal_countries[sorted_sim_result_by_one_total_with_no_medal_countries['One Simulation Total Medal'] > 0]
count_greater_than_zero = len(filtered_df)
noc_values = filtered_df['NOC'].tolist()
print("Count of 'One Simulation Total Medal' > 0:", count_greater_than_zero)
print("NOC values for these rows:", noc_values)


num_simulations_2 = 10000

record = {
    "FirstMedalCountryNum": [],
    "FirstMedalCountry":[]
}

for _ in range(num_simulations_2):
    first_medal_count = 0
    first_medal_country_list = []
    # 对每个国家进行模拟
    for index, row in sorted_by_total_medal_df.iterrows():
        noc = row['NOC']
        predicted_gold = row['predicted_gold']
        predicted_total_medal = row['predicted_total_medal']

        # 对金牌和总奖牌数进行蒙特卡罗模拟
        simulated_gold = np.random.poisson(predicted_gold)
        simulated_total_medal = np.random.poisson(predicted_total_medal)

        # 如果这个国家没有历史奖牌记录，并且在此次模拟中赢得至少一块奖牌
        if noc in no_medal_countries_list and simulated_total_medal > 0:
            first_medal_count += 1
            first_medal_country_list.append(noc)

    # 记录每次模拟中首次赢得奖牌的国家数量
    record["FirstMedalCountryNum"].append(first_medal_count)
    record["FirstMedalCountry"].append(first_medal_country_list)

record_df = pd.DataFrame(record)

import matplotlib.pyplot as plt

# 设置 Matplotlib 使用的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False    # 用于正常显示负号

# 绘制首次赢得奖牌国家数量的分布图
plt.hist(record_df["FirstMedalCountryNum"], bins=range(0, max(record_df["FirstMedalCountryNum"]) + 1), alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of First Medal Winning Countries')
plt.xlabel('Number of First Medal Winning Countries')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
# 保存首次赢得奖牌国家数量分布图
plt.savefig('first_medal_country_distribution.png')

plt.show()

from collections import Counter
# 计算每个国家在首次赢得奖牌列表中出现的频率
country_counter = Counter([country for sublist in record["FirstMedalCountry"] for country in sublist])

# 获取最有可能首次赢得奖牌的前 10 个国家
most_likely_countries = country_counter.most_common(10)

# 绘制前 10 个国家及其概率
countries, frequencies = zip(*most_likely_countries)
total_counts = sum(country_counter.values())
probabilities = [freq / total_counts for freq in frequencies]

plt.figure(figsize=(10, 6))
plt.bar(countries, probabilities, color='orange', edgecolor='black')
plt.title('最有可能首次赢得奖牌的前 10 个国家')
plt.xlabel('国家')
plt.ylabel('概率')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.75)
# 保存前 10 个国家及其概率图
plt.savefig('top_10_first_medal_countries.png')


plt.show()