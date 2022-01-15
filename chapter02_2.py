import pandas as pd
import numpy as np

"""数据重构"""

text_left_up = pd.read_csv(r"D:\BaiduNetdiskDownload\机器学习和推荐系统"
                           r"\hands-on-data-analysis-master\hands-on-data-analysis-master"
                           r"\第二章项目集合\data\train-left-up.csv")
# print(text.head())

text_left_down = pd.read_csv(r"D:\BaiduNetdiskDownload\机器学习和推荐系统"
                             r"\hands-on-data-analysis-master\hands-on-data-analysis-master"
                             r"\第二章项目集合\data\train-left-down.csv")

text_right_down = pd.read_csv(r"D:\BaiduNetdiskDownload\机器学习和推荐系统"
                              r"\hands-on-data-analysis-master\hands-on-data-analysis-master"
                              r"\第二章项目集合\data\train-right-down.csv")

text_right_up = pd.read_csv(r"D:\BaiduNetdiskDownload\机器学习和推荐系统"
                            r"\hands-on-data-analysis-master\hands-on-data-analysis-master"
                            r"\第二章项目集合\data\train-right-up.csv")

print(text_left_up.columns)
print(text_left_down.columns)
print(text_right_up.columns)
print(text_right_down.columns)

list_up = [text_left_up, text_right_up]
result_up = pd.concat(list_up, axis=1)
# result_up.to_csv(r"C:\Users\Ponyto\Desktop\result_up.csv")
print(result_up.describe())
print("***" * 20)
print(result_up.info())

list_down = [text_left_down, text_right_down]
result_down = pd.concat(list_down, axis=1)

list_res = [result_up, result_down]
result = pd.concat(list_res, axis=0)
# result.to_csv(r"C:\Users\Ponyto\Desktop\result.csv")

print("***" * 20)

"""
join:横向拼接
append:纵向拼接
"""
result_up = text_left_up.join(text_right_up)
result_down = text_left_down.join(text_right_down)
result = result_up.append(result_down)

result_up = pd.merge(text_left_up, text_right_up, left_index=True, right_index=True)
result_down = pd.merge(text_left_down, text_right_down, left_index=True, right_index=True)
result = result_up.append(result_down)

df = pd.read_csv(r"D:\BaiduNetdiskDownload\机器学习和推荐系统"
                 r"\hands-on-data-analysis-master\hands-on-data-analysis-master"
                 r"\第二章项目集合\result_pt.csv")

print(df.head())

unit_result = df.stack().head(20)
print("***" * 20)
print(unit_result.head())
unit_result.to_csv(r"C:\Users\Ponyto\Desktop\unit_result.csv")

