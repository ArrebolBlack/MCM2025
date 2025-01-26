import pandas as pd
import numpy as np

path_prefix = "E:/25MCM/2025_Problem_C_Data/2025_Problem_C_Data/"

summerOly_athletes_path = path_prefix + "summerOly_athletes.csv"
summerOly_medal_counts_path = path_prefix + "summerOly_medal_counts.csv"
summerOly_hosts_path = path_prefix + "summerOly_hosts.csv"
summerOly_programs_path = path_prefix + "summerOly_programs.csv"

# 读取CSV文件
athlete_data = pd.read_csv(summerOly_athletes_path, encoding='ISO-8859-1')
medal_data = pd.read_csv(summerOly_medal_counts_path, encoding='ISO-8859-1')
host_data = pd.read_csv(summerOly_hosts_path, encoding='utf-8')
program_data = pd.read_csv(summerOly_programs_path, encoding='ISO-8859-1')

# 清理所有不可见或非打印字符
host_data['Host'] = host_data['Host'].str.encode('ascii', 'ignore').str.decode('ascii')
medal_data['NOC'] = medal_data['NOC'].str.encode('ascii', 'ignore').str.decode('ascii')
athlete_data = athlete_data[athlete_data['Year'] != 1906]
# 填补缺失值
program_data.fillna(0, inplace=True)
# 提取国家全称
host_data['Host Country'] = host_data['Host'].apply(lambda x: x.split(',')[-1])
# NOC特殊情况处理
# 替换字典
noc_replacements = {
    'AGR': 'ALG', 'AGL': 'ALG',
    'BAD': 'BAR',
    'DAY': 'BEN', 'DAH': 'BEN',
    'BSH': 'BIH',
    'HBR': 'BIZ',
    'VOL': 'BUR',
    'AFC': 'CAF',
    'CAB': 'CAM', 'KHM': 'CAM',
    'CHD': 'CHA',
    'CIL': 'CHI',
    'PRC': 'CHN',
    'IVC': 'CIV', 'CML': 'CIV',
    'COK': 'COD', 'ZAI': 'COD',
    'COS': 'CRC',
    'TCH': 'CZE',
    'DAN': 'DEN', 'DIN': 'DEN',
    'RAU': 'EGY', 'UAR': 'EGY',
    'SAL': 'ESA',
    'SPA': 'ESP',
    'ETI': 'ETH',
    'FIG': 'FIJ',
    'GRB': 'GBR', 'GBI': 'GBR',
    'ALL': 'GER', 'ALE': 'GER',
    'GUT': 'GUA',
    'GUI': 'GUY',
    'HOK': 'HKG',
    'UNG': 'HUN',
    'INS': 'INA',
    'IRN': 'IRI', 'IRA': 'IRI',
    'IRK': 'IRQ',
    'ICE': 'ISL',
    'GIA': 'JPN', 'JAP': 'JPN',
    'COR': 'KOR',
    'ARS': 'KSA', 'SAU': 'KSA',
    'LYA': 'LBA', 'LBY': 'LBA',
    'LEB': 'LBN', 'LIB': 'LBN',
    'LIC': 'LIE',
    'LIT': 'LTU',
    'MAG': 'MAD',
    'MRC': 'MAR',
    'MAL': 'MAS',
    'MLD': 'MDA',
    'MON': 'MGL',
    'MAT': 'MLT',
    'BIR': 'MYA', 'BUR': 'MYA',
    'NCG': 'NCA', 'NIC': 'NCA',
    'OLA': 'NED', 'NET': 'NED', 'PBA': 'NED', 'NLD': 'NED', 'HOL': 'NED',
    'NIG': 'NGR', 'NGA': 'NGR',
    'NZE': 'NZL',
    'FIL': 'PHI',
    'NGY': 'PNG', 'NGU': 'PNG',
    'NKO': 'PRK', 'CDN': 'PRK',
    'PRI': 'PUR', 'PRO': 'PUR',
    'ROM': 'ROU', 'RUM': 'ROU',
    'SAF': 'RSA',
    'OAR': 'RUS', 'ROC': 'RUS',
    'SGL': 'SEN',
    'SIN': 'SGP',
    'SLA': 'SLE',
    'SMA': 'SMR',
    'CEY': 'SRI', 'CEI': 'SRI',
    'SVI': 'SUI', 'SWI': 'SUI',
    'SVE': 'SWE', 'SUE': 'SWE',
    'RAU': 'SYR', 'SIR': 'SYR',
    'TON': 'TGA',
    'IOA': 'TLS',
    'RCF': 'TPE', 'TWN': 'TPE', 'ROC': 'TPE',
    'TRT': 'TTO', 'TRI': 'TTO',
    'URG': 'URU',
    'SUA': 'USA', 'EUA': 'USA',
    'VET': 'VIE', 'VNM': 'VIE',
    'NRH': 'ZAM',
    'RHO': 'ZIM'
}
# 替换 NOC 233 -> 220
athlete_data['NOC'] = athlete_data['NOC'].replace(noc_replacements)
# print(athlete_data['NOC'].unique())
# 替换 Sport -> 项目代码
program_code_mapping = {
    'Basketball': 'BKB',
    'Judo': 'JUD',
    'Football': 'FBL',
    'Tug-Of-War': None,  # 没有对应的现代奥运项目代码
    'Athletics': 'ATH',
    'Swimming': 'SWM',
    'Badminton': 'BDM',
    'Sailing': 'SAL',
    'Gymnastics': 'GAR',  # 指代一般体操
    'Art Competitions': None,  # 非现行奥运项目
    'Handball': 'HBL',
    'Weightlifting': 'WLF',
    'Wrestling': 'WRF',  # 默认自由式摔跤
    'Water Polo': 'WPO',
    'Hockey': 'HOC',
    'Rowing': 'ROW',
    'Fencing': 'FEN',
    'Equestrianism': 'EDR',  # 通用马术项目
    'Shooting': 'SHO',
    'Boxing': 'BOX',
    'Taekwondo': 'TKW',
    'Cycling': 'CRD',  # 默认自行车公路赛
    'Diving': 'DIV',
    'Canoeing': 'CSP',  # 默认静水皮划艇
    'Tennis': 'TEN',
    'Modern Pentathlon': 'MPN',
    'Golf': 'GLF',
    'Softball': 'SBL',
    'Archery': 'ARC',
    'Volleyball': 'VVO',
    'Synchronized Swimming': 'SWA',
    'Table Tennis': 'TTE',
    'Baseball': 'BSB',
    'Rhythmic Gymnastics': 'GRY',
    'Rugby Sevens': 'RU7',
    'Trampolining': 'GTR',
    'Beach Volleyball': 'VBV',
    'Triathlon': 'TRI',
    'Rugby': 'RU7',  # 默认为七人制橄榄球
    'Lacrosse': None,  # 非现行奥运项目
    'Polo': None,  # 非现行奥运项目
    'Cricket': None,  # 非现行奥运项目
    'Ice Hockey': None,  # 非夏季奥运项目
    'Racquets': None,  # 非现行奥运项目
    'Motorboating': None,  # 非现行奥运项目
    'Croquet': None,  # 非现行奥运项目
    'Figure Skating': None,  # 非夏季奥运项目
    'Jeu De Paume': None,  # 非现行奥运项目
    'Roque': None,  # 非现行奥运项目
    'Basque Pelota': 'PEL',
    'Alpinism': None,  # 非现行奥运项目
    'Aeronautics': None,  # 非现行奥运项目
    'Cycling Road': 'CRD',
    'Artistic Gymnastics': 'GAR',
    'Karate': 'KTE',
    'Baseball/Softball': 'BSB',  # 默认棒球
    'Trampoline Gymnastics': 'GTR',
    'Marathon Swimming': 'OWS',
    'Canoe Slalom': 'CSL',
    'Surfing': 'SRF',
    'Canoe Sprint': 'CSP',
    'Cycling BMX Racing': 'BMX',
    'Equestrian': 'EDR',
    'Artistic Swimming': 'SWA',
    'Cycling Track': 'CTR',
    'Skateboarding': 'SKB',
    'Cycling Mountain Bike': 'MTB',
    '3x3 Basketball': 'BK3',
    'Cycling BMX Freestyle': 'BMF',
    'Sport Climbing': 'CLB',
    'Marathon Swimming, Swimming': 'OWS',  # 假设主要指代马拉松游泳
    'Breaking': 'BKG',
    'Cycling Road, Cycling Track': 'CRD',  # 默认自行车公路赛
    'Cycling Road, Cycling Mountain Bike': 'CRD',  # 默认自行车公路赛
    'Cycling Road, Triathlon': 'TRI',  # 假设主要指代铁人三项
    '3x3 Basketball, Basketball': 'BKB'
}
athlete_data['Sport'] = athlete_data['Sport'].replace(program_code_mapping)
# print(athlete_data['Sport'].unique())
# # 国家全称到简写的映射
country_to_noc = {
    # 修正部分
    'United States': 'USA',
    'Greece': 'GRE',
    'Germany': 'GER',
    'France': 'FRA',
    'Great Britain': 'GBR',
    'United Kingdom': 'GBR',  # 添加 United Kingdom
    'Hungary': 'HUN',
    'Austria': 'AUT',
    'Australia': 'AUS',
    'Denmark': 'DEN',
    'Switzerland': 'SUI',
    'Mixed team': 'ZZX',  # 非正式队伍
    'Belgium': 'BEL',
    'Italy': 'ITA',
    'Cuba': 'CUB',
    'Canada': 'CAN',
    'Spain': 'ESP',
    'Luxembourg': 'LUX',
    'Norway': 'NOR',
    'Netherlands': 'NED',
    'India': 'IND',
    'Bohemia': 'BOH',  # 历史名称
    'Sweden': 'SWE',
    'Australasia': 'ANZ',  # 澳大利亚和新西兰联合队
    'Russian Empire': 'RU1',  # 俄国历史名称
    'Finland': 'FIN',
    'South Africa': 'RSA',
    'Estonia': 'EST',
    'Brazil': 'BRA',
    'Japan': 'JPN',
    'Czechoslovakia': 'TCH',  # 捷克斯洛伐克历史名称
    'New Zealand': 'NZL',
    'Yugoslavia': 'YUG',  # 南斯拉夫历史名称
    'Argentina': 'ARG',
    'Uruguay': 'URU',
    'Poland': 'POL',
    'Haiti': 'HAI',
    'Portugal': 'POR',
    'Romania': 'ROU',
    'Egypt': 'EGY',
    'Ireland': 'IRL',
    'Chile': 'CHI',
    'Philippines': 'PHI',
    'Mexico': 'MEX',
    'Latvia': 'LAT',
    'Turkey': 'TUR',
    'Jamaica': 'JAM',
    'Peru': 'PER',
    'Ceylon': 'CEY',  # 斯里兰卡的历史名称
    'Trinidad and Tobago': 'TTO',
    'Panama': 'PAN',
    'South Korea': 'KOR',
    'Iran': 'IRI',
    'Puerto Rico': 'PUR',
    'Soviet Union': 'URS',  # 苏联历史名称
    'Lebanon': 'LIB',
    'Bulgaria': 'BUL',
    'Venezuela': 'VEN',
    'United Team of Germany': 'EUA',  # 德国联合队历史名称
    'Iceland': 'ISL',
    'Pakistan': 'PAK',
    'Bahamas': 'BAH',
    'Ethiopia': 'ETH',
    'Formosa': 'TPE',  # 台湾的历史名称
    'Ghana': 'GHA',
    'Morocco': 'MAR',
    'Singapore': 'SGP',
    'British West Indies': 'BWI',  # 英属西印度群岛历史名称
    'Iraq': 'IRQ',
    'Tunisia': 'TUN',
    'Kenya': 'KEN',
    'Nigeria': 'NGR',
    'East Germany': 'GDR',  # 东德历史名称
    'West Germany': 'FRG',  # 西德历史名称
    'Mongolia': 'MGL',
    'Uganda': 'UGA',
    'Cameroon': 'CMR',
    'Taiwan': 'TPE',
    'North Korea': 'PRK',
    'Colombia': 'COL',
    'Niger': 'NIG',
    'Bermuda': 'BER',
    'Thailand': 'THA',
    'Zimbabwe': 'ZIM',
    'Tanzania': 'TAN',
    'Guyana': 'GUY',
    'China': 'CHN',
    'Ivory Coast': 'CIV',
    'Syria': 'SYR',
    'Algeria': 'ALG',
    'Chinese Taipei': 'TPE',
    'Dominican Republic': 'DOM',
    'Zambia': 'ZAM',
    'Suriname': 'SUR',
    'Costa Rica': 'CRC',
    'Indonesia': 'INA',
    'Netherlands Antilles': 'AHO',  # 荷属安的列斯历史名称
    'Senegal': 'SEN',
    'Virgin Islands': 'ISV',
    'Djibouti': 'DJI',
    'Unified Team': 'EUN',  # 独联体联合队历史名称
    'Lithuania': 'LTU',
    'Namibia': 'NAM',
    'Croatia': 'CRO',
    'Independent Olympic Participants': 'IOP',  # 独立参赛者
    'Israel': 'ISR',
    'Slovenia': 'SLO',
    'Malaysia': 'MAS',
    'Qatar': 'QAT',
    'Russia': 'RUS',
    'Ukraine': 'UKR',
    'Czech Republic': 'CZE',
    'Kazakhstan': 'KAZ',
    'Belarus': 'BLR',
    'FR Yugoslavia': 'YUG',
    'Slovakia': 'SVK',
    'Armenia': 'ARM',
    'Burundi': 'BDI',
    'Ecuador': 'ECU',
    'Hong Kong': 'HKG',
    'Moldova': 'MDA',
    'Uzbekistan': 'UZB',
    'Azerbaijan': 'AZE',
    'Tonga': 'TGA',
    'Georgia': 'GEO',
    'Mozambique': 'MOZ',
    'Saudi Arabia': 'KSA',
    'Sri Lanka': 'SRI',
    'Vietnam': 'VIE',
    'Barbados': 'BAR',
    'Kuwait': 'KUW',
    'Kyrgyzstan': 'KGZ',
    'Macedonia': 'MKD',
    'United Arab Emirates': 'UAE',
    'Serbia and Montenegro': 'SCG',  # 塞黑历史名称
    'Paraguay': 'PAR',
    'Eritrea': 'ERI',
    'Serbia': 'SRB',
    'Tajikistan': 'TJK',
    'Samoa': 'SAM',
    'Sudan': 'SUD',
    'Afghanistan': 'AFG',
    'Mauritius': 'MRI',
    'Togo': 'TOG',
    'Bahrain': 'BRN',
    'Grenada': 'GRN',
    'Botswana': 'BOT',
    'Cyprus': 'CYP',
    'Gabon': 'GAB',
    'Guatemala': 'GUA',
    'Montenegro': 'MNE',
    'Independent Olympic Athletes': 'IOA',
    'Fiji': 'FIJ',
    'Jordan': 'JOR',
    'Kosovo': 'KOS',
    'ROC': 'ROC',  # 俄罗斯奥委会
    'San Marino': 'SMR',
    'North Macedonia': 'MKD',
    'Turkmenistan': 'TKM',
    'Burkina Faso': 'BUR',
    'Saint Lucia': 'LCA',
    'Dominica': 'DMA',
    'Albania': 'ALB',
    'Cabo Verde': 'CPV',
    'Refugee Olympic Team': 'ROT',
    # 扩展部分
    'Chad': 'CHA',
    'Nicaragua': 'NCA',
    'United Arab Republic': 'UAR',
    'Libya': 'LBA',
    'Palestine': 'PLE',
    'Comoros': 'COM',
    'Brunei': 'BRU',
    'Maldives': 'MDV',
    'North Yemen': 'YAR',
    'Congo (Brazzaville)': 'CGO',
    'Monaco': 'MON',
    'Benin': 'BEN',
    'Somalia': 'SOM',
    'Mali': 'MLI',
    'Angola': 'ANG',
    'Bangladesh': 'BAN',
    'El Salvador': 'ESA',
    'Honduras': 'HON',
    'Seychelles': 'SEY',
    'Mauritania': 'MTN',
    'Saint Kitts and Nevis': 'SKN',
    'Saint Vincent and the Grenadines': 'VIN',
    'Liberia': 'LBR',
    'Nepal': 'NEP',
    'Palau': 'PLW',
    'American Samoa': 'ASA',
    'Rwanda': 'RWA',
    'Malta': 'MLT',
    'Guinea': 'GUI',
    'Belize': 'BIZ',
    'South Yemen': 'YMD',
    'Sierra Leone': 'SLE',
    'Papua New Guinea': 'PNG',
    'Yemen': 'YEM',
    'Oman': 'OMA',
    'Vanuatu': 'VAN',
    'British Virgin Islands': 'IVB',
    'Central African Republic': 'CAF',
    'Madagascar': 'MAD',
    'Malaya': 'MAL',
    'Bosnia and Herzegovina': 'BIH',
    'Guam': 'GUM',
    'Cayman Islands': 'CAY',
    'Guinea Bissau': 'GBS',
    'Timor Leste': 'TLS',
    'Congo (Kinshasa)': 'COD',
    'Laos': 'LAO',
    'GyoshuII': 'CAM',
    'Solomon Islands': 'SOL',
    'Equatorial Guinea': 'GEQ',
    'Bolivia': 'BOL',
    'Saar': 'SAA',
    'Antigua and Barbuda': 'ANT',
    'Andorra': 'AND',
    'Federated States of Micronesia': 'FSM',
    'Myanmar': 'MYA',
    'Malawi': 'MAW',
    'Rhodesia': 'RHO',
    'Sao Tome and Principe': 'STP',
    'Liechtenstein': 'LIE',
    'Gambia': 'GAM',
    'Cook Islands': 'COK',
    'Circus': 'WIF',
    'Swaziland': 'SWZ',
    'North Borneo': 'NBO',
    'Aruba': 'ARU',
    'Nauru': 'NRU',
    'South Vietnam': 'VNM',
    'Bhutan': 'BHU',
    'Marshall Islands': 'MHL',
    'Kiribati': 'KIR',
    'Unknown': 'UNK',
    'Tuvalu': 'TUV',
    'Newfoundland': 'NFL',
    'South Sudan': 'SSD',
    'Lesotho': 'LES',
    'Refugee Olympic Team': 'EOR',
    'Lebanon': 'LBN',
    'AIN': 'AIN',
    # 'Lebanon': 'LIB',
    'Refugee Olympic Athletes': 'ROT'
    # 省略其他扩展部分，需根据真实情况补充...
}


# 最后选出52个项目
program_code_list = [
 'SWA', 'DIV', 'OWS', 'SWM', 'WPO', 'ARC', 'ATH', 'BDM', 'BSB', 'SBL', 'BK3', 'BKB',
 'PEL', 'BOX', 'BKG', 'CSP', 'CSL', 'BMF', 'BMX', 'MTB', 'CRD', 'CTR', 'EDR', 'EVE',
 'EJP', 'FEN', 'HOC', 'FBL', 'GLF', 'GAR', 'GRY', 'GTR', 'HBL', 'JUD', 'KTE', 'MPN',
 'ROW', 'RU7', 'SAL', 'SHO', 'SKB', 'CLB', 'SRF', 'TTE', 'TKW', 'TEN', 'TRI', 'VBV',
 'VVO', 'WLF', 'WRF', 'WRG',
]
# 选出年份
year_list = [
    1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960,
    1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016,
    2020, 2024,
]
# 国家，按照NOC （可能一个NOC有多个Team）
country_list = [
 'CHN', 'DEN', 'NED', 'FIN', 'NOR', 'ROU', 'EST', 'FRA', 'MAR', 'ESP', 'EGY', 'IRI',
 'BUL', 'ITA', 'CHA', 'AZE', 'SUD', 'RUS', 'ARG', 'CUB', 'BLR', 'GRE', 'CMR', 'TUR',
 'CHI', 'MEX', 'USA', 'URS', 'NCA', 'HUN', 'NGR', 'ALG', 'KUW', 'BRN', 'PAK', 'IRQ',
 'LBN', 'QAT', 'MAS', 'GER', 'CAN', 'IRL', 'AUS', 'RSA', 'ERI', 'TAN', 'JOR', 'TUN',
 'LBA', 'BEL', 'DJI', 'PLE', 'COM', 'KAZ', 'BRU', 'IND', 'KSA', 'SYR', 'MDV', 'ETH',
 'UAE', 'YAR', 'INA', 'PHI', 'SGP', 'UZB', 'KGZ', 'TJK', 'EUN', 'JPN', 'CGO', 'SUI',
 'BRA', 'GDR', 'MGL', 'ISR', 'URU', 'SWE', 'SRI', 'ARM', 'CIV', 'KEN', 'BEN', 'GBR',
 'GHA', 'SOM', 'MLI', 'AFG', 'POL', 'CRC', 'PAN', 'GEO', 'SLO', 'GUY', 'NZL', 'POR',
 'PAR', 'ANG', 'VEN', 'COL', 'FRG', 'BAN', 'PER', 'ESA', 'PUR', 'UGA', 'HON', 'ECU',
 'TKM', 'MRI', 'SEY', 'CZE', 'LUX', 'MTN', 'SKN', 'TTO', 'DOM', 'VIN', 'JAM', 'LBR',
 'SUR', 'NEP', 'AUT', 'PLW', 'LTU', 'TOG', 'NAM', 'AHO', 'UKR', 'ISL', 'ASA', 'SAM',
 'RWA', 'CRO', 'DMA', 'HAI', 'MLT', 'CYP', 'BIZ', 'YMD', 'THA', 'BER', 'ANZ', 'SCG',
 'SLE', 'PNG', 'YEM', 'TLS', 'OMA', 'FIJ', 'VAN', 'MDA', 'BAH', 'GUA', 'YUG', 'LAT',
 'SRB', 'IVB', 'MOZ', 'ISV', 'CAF', 'MAD', 'BIH', 'GUM', 'CAY', 'SVK', 'BAR', 'GBS',
 'COD', 'GAB', 'SMR', 'LAO', 'BOT', 'ROT', 'KOR', 'CAM', 'PRK', 'SOL', 'SEN', 'CPV',
 'GEQ', 'BOL', 'SAA', 'ANT', 'AND', 'ZIM', 'GRN', 'HKG', 'LCA', 'FSM', 'MYA', 'MAW',
 'ZAM', 'TPE', 'STP', 'MKD', 'BOH', 'LIE', 'MNE', 'GAM', 'ALB', 'WIF', 'SWZ', 'NBO',
 'BDI', 'ARU', 'NRU', 'VIE', 'BHU', 'MHL', 'KIR', 'UNK', 'TUV', 'TGA', 'NFL', 'KOS',
 'SSD', 'LES', 'EOR', 'AIN',
]



# 构建数据集
data = []
year_list_2 = [1920] + year_list # 1916 停办
for year in year_list_2:
    host = host_data.loc[host_data['Year'] == year, 'Host Country'].values[0]
    host_NOC = country_to_noc[host]
    for country in country_list:
        if(host_NOC == country):
            isHost = 1
        else:
            isHost = 0
        # 筛选出符合条件的奖牌数据
        medal = medal_data.loc[
            (medal_data['Year'] == year) & (medal_data['NOC'].map(country_to_noc) == country)
            ]

        # 检查匹配结果
        if medal.empty:
            # 如果没有匹配的行，设置奖牌数量为0
            gold = 0
            silver = 0
            bronze = 0
            total_medal = 0
        elif len(medal) > 1:
            # 如果匹配到多行，抛出异常
            raise ValueError(f"Multiple rows found for year {year} and country {country}")
        else:
            # 如果只有一行匹配，提取奖牌数据
            gold = medal.iloc[0]['Gold']
            silver = medal.iloc[0]['Silver']
            bronze = medal.iloc[0]['Bronze']
            total_medal = medal.iloc[0]['Total']
        # # for program in program_code_list:
        data.append([year, country, isHost, gold, silver, bronze, total_medal])

columns = ['year', 'country', 'is_host', 'gold', 'silver', 'bronze', 'total_medal']
data_df = pd.DataFrame(data, columns=columns)
# print(data_df.info())
# print(data_df)

data_df.to_csv("YearCountry-host-medal.csv")

# year = 2024
# athlete_data.loc[]

# data = athlete_data
# # 查询每列的唯一值
# for column in data.columns:
#     print(f"列 {column} 的唯一值：")
#     print(data[column].unique())


