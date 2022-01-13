import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

"""
从数据的各个角度进行观察数据。
主要包括数据清洗、数据的特征处理、数据重构以及数据可视化。
为数据分析最后的建模和模型评价做铺垫。
"""

df = pd.read_csv(r"D:\BaiduNetdiskDownload\机器学习和推荐系统"
                 r"\hands-on-data-analysis-master\hands-on-data-analysis-master"
                 r"\第二章项目集合\train.csv")

# 方便显示完整
# df.head().to_csv(r"C:\Users\Ponyto\Desktop\1.csv")
#
# print(df.head())

print(df.columns)

# Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
#       dtype='object')

"""
数据清洗：
学习缺失值、重复值、字符串和数据转换等操作，将数据清洗成可以分析或建模的样子
"""

""" task01 缺失值处理"""
# 查看数据中缺失值分布情况
print(df.info())  # 统计非Nan
print("*" * 30)
print(df.isnull().sum())  # 只统计数据中为Nan

print("*" * 30)
print(df[['Pclass', 'Embarked']].head())

# 建议处理缺失值的方法是：
# 先尝试删除有缺失项的数据，然后训练模型，先把baseline做出来；
# 然后会依次尝试：特殊值填充，（特殊）平均值填充和最近邻法。

# 处理的方法有：1. 不处理 2. 平均值填充 3. 特殊值填充 4. 热卡填充等
# https://cloud.tencent.com/developer/article/1667676

# 都填充0
df.head(30).to_csv(r"C:\Users\Ponyto\Desktop\1.csv")
print("*" * 30)
# df[df['Age'] == np.nan] = 0
# df[df['Age'].isnull()] = 0  # 不可取 把一个样本全部为0

# df1 = df.fillna(0)  # 数据中为Nan的全部设置为0
df['Age'].fillna(0, inplace=True)  # 指定列设置为0
# 检索空缺值用np.nan,None以及.isnull()哪个更好
# 数值列读取数据后，空缺值的数据类型为float64所以用None一般索引不到，比较的时候最好用np.nan
# df1.head(30).to_csv(r"C:\Users\Ponyto\Desktop\3.csv")
# df.head(30).to_csv(r"C:\Users\Ponyto\Desktop\2.csv")

""" task02 重复值处理"""
print("*" * 30)
df[df.duplicated()].to_csv(r"C:\Users\Ponyto\Desktop\4.csv")

print(df.head())

# df.to_csv(r"C:\Users\Ponyto\Desktop\train_clear_pt.csv")

print(df.info())

""" task03 特征观察与处理"""

"""
数值型特征：Survived ，Pclass， Age ，SibSp， Parch， Fare，
           其中Survived， Pclass为离散型数值特征，Age，SibSp， Parch， Fare为连续型数值特征
文本型特征：Name， Sex， Cabin，Embarked， Ticket，
           其中Sex， Cabin， Embarked， Ticket为类别型文本特征。
数值型特征一般可以直接用于模型的训练，但有时候为了模型的稳定性及鲁棒性会对连续变量进行离散化。
文本型特征往往需要转换成数值型特征才能用于建模分析
"""

# 对年龄进行分箱
df['AgeBand'] = pd.cut(df['Age'], 5, labels=[1, 2, 3, 4, 5])  # 平均分箱

# df.to_csv(r"C:\Users\Ponyto\Desktop\band_avg.csv")
df['AgeBand'] = pd.cut(df['Age'], [0, 5, 15, 30, 50, 80], labels=[1, 2, 3, 4, 5])
# df.to_csv(r"C:\Users\Ponyto\Desktop\band_def.csv")

# 将连续变量Age按10% 30% 50 70% 90%五个年龄段，并用分类变量12345表示
df['AgeBand'] = pd.qcut(df['Age'], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1], labels=[1, 2, 3, 4, 5], duplicates="drop")
# df.to_csv(r"C:\Users\Ponyto\Desktop\band_percent.csv")


"""
对文本变量进行转换
Name， Sex， Cabin，Embarked， Ticket，
其中Sex， Cabin， Embarked， Ticket为类别型文本特征。
"""
print("*" * 30)
print(df['Sex'].value_counts())
df['Sex_Number'] = df['Sex'].replace(['male', 'female'], [1, 0])
print("*" * 30)

# for feat in ["Age", "Embarked", 'Cabin', 'Ticket']:
#     #     x = pd.get_dummies(df["Age"] // 6)
#     #     x = pd.get_dummies(pd.cut(df['Age'],5))
#     x = pd.get_dummies(df[feat], prefix=feat)
#     df = pd.concat([df, x], axis=1)
#     # df[feat] = pd.get_dummies(df[feat], prefix=feat)
#
# for i in ['Cabin', 'Ticket', 'Embarked', 'Age']:
#     del df[i]
# df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)  # ?
# df.to_csv(r"C:\Users\Ponyto\Desktop\data_clear2.csv")

del df['Sex']  # 把Sex特征删除，直接使用Sex_Number
df.to_csv(r"C:\Users\Ponyto\Desktop\sex.csv")

print("*" * 30)

print(df['Cabin'].value_counts())
print("*" * 30)
print(df['Ticket'].value_counts())
print("*" * 30)
print(df['Embarked'].value_counts())
print("*" * 30)
df['Embarked_Number'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

for feat in ['Cabin', 'Ticket']:
    lbl = LabelEncoder()
    label_dict = dict(zip(df[feat].unique(), range(df[feat].nunique())))
    df[feat + "_labelEncode"] = df[feat].map(label_dict)
    df[feat + "_labelEncode"] = lbl.fit_transform(df[feat].astype(str))

for i in ['Cabin', 'Ticket', 'Embarked']:
    del df[i]
# print(df.head())
df.to_csv(r"C:\Users\Ponyto\Desktop\data_clear1.csv")

#  one-hot编码  https://www.cnblogs.com/lianyingteng/p/7755545.html

df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)  # ?
df.to_csv(r"C:\Users\Ponyto\Desktop\data_clear1.csv")


