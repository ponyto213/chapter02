import pandas as pd
import numpy as np

df = pd.read_csv(r"D:\BaiduNetdiskDownload\机器学习和推荐系统"
                 r"\hands-on-data-analysis-master\hands-on-data-analysis-master"
                 r"\第二章项目集合\result_pt.csv")

print(df.head())

"""
Group_by 以。。。为分组的类别进行聚合
一般分组之后只能取一些聚合函数，例如sum, average, max, min ...
"""
print(df.columns)
gFare = df['Fare'].groupby(df['Sex'])
print("***" * 20)
print(gFare.mean())
print("***" * 20)

survivedSex = df['Survived'].groupby(df['Sex'])
print(survivedSex.sum())
# print(survivedSex)

print("***" * 20)
survived_Pclass = df['Survived'].groupby(df['Pclass'])
print(survived_Pclass.sum())
print(survived_Pclass.count())
print(survived_Pclass.mean())

"""
女性存活人数较多，女性的平均票价也更高 ————> 女性买的船仓位置更好
"""

"""
使用agg()一次性按照同一个分组进行统计分析
还支持修改列名

        mean_fare  count_pclass  Survived
Sex                                      
female  44.479818           314       233
male    25.523893           577       109
"""
print(df.groupby('Sex')
      .agg({'Fare': 'mean', 'Pclass': 'count', 'Survived': 'sum'})
      .rename(columns={'Fare': 'mean_fare', 'Pclass': 'count_pclass'}))
print("***" * 20)
group_F = df.groupby(['Pclass', 'Age'])['Fare']
print(group_F.mean().head())

# 横向拼接两个输出结果，按照Sex
result = pd.merge(gFare.mean(), survivedSex.sum(), on='Sex')
print("***" * 20)
print(result)
# result.to_csv(r"C:\Users\Ponyto\Desktop\unit_result.csv")

print("***" * 20)

survived_age = df.groupby('Age')['Survived'].sum()
# print(survived_age.sum().head(10))
# print(survived_age.count().head(10))

print(survived_age[survived_age.values == survived_age.max()])

print(df['Survived'].sum())

print(15 / 342)