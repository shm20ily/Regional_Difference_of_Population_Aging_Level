import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from snapshot_selenium import snapshot as driver
from pyecharts.render import make_snapshot
from pyecharts.charts import Pie
from pyecharts.charts import Bar
from pyecharts.charts import HeatMap
from pyecharts import options as opts
from pyecharts.charts import Map, Timeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import cluster
from sklearn import metrics

# 加载字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
# 显示负号
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
df1 = pd.read_excel("2002年到2019年各省老龄化人数.xlsx", sheet_name='Sheet1')
# 将某一列设为索引
df2 = df1.set_index('地区')
# print(df2)
# # 数据预处理-----------------------------------------------------------------------------------------------------------
# # 查看重复值
# print(df2.duplicated().any())  # 无重复值，无需处理
# # 查看缺失值
# print(df2.isnull().any())  # 无缺失值，无需处理
# # 数据探索
# print('df2.shape:', '-' * 50, '\n', df2.shape)
# print('df2.info:', '-' * 50, '\n', df2.info())
# print('df2.dtypes:', '-' * 50, '\n', df2.dtypes)
# print('df2.head:', '-' * 50, '\n', df2.head())
# print('df2.describe:', '-' * 50, '\n', df2.describe())
# df2.mean()  # 平均值
# df2.std()  # 标准差
# df2.var()  # 方差
# df2.median()  # 中位数
# df2.mode()  # 众数
# print('缺失值个数:', '-' * 50, '\n', df2.isnull().sum())

# # 图1
# df2.plot(style='--.', alpha=0.8)
# plt.xticks(range(31), df2.index, rotation=90)
# plt.xlabel('省份')
# plt.ylabel('老龄化人数')
# plt.title('2002年到2019年各省老龄化人数走势图')
# plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)  # 设置legend地位置，将其放在图外
# plt.grid()
# plt.tight_layout()
# # plt.savefig('2002年到2019年各省老龄化人数走势图.png')
# plt.show()
# print('图1', '-' * 80)

# # 绘图2
# plt.plot(df2.index,  # x轴数据
#          df2['2005年'].values,  # y轴数据
#          linestyle='-',  # 折线类型
#          linewidth=2,  # 折线宽度
#          color='steelblue',  # 折线颜色
#          marker='o',  # 点的形状
#          markersize=6,  # 点的大小
#          markeredgecolor='black',  # 点的边框色
#          markerfacecolor='brown',  # 点的填充色
#          label='2005年')
# plt.plot(df2.index,  # x轴数据
#          df2['2015年'].values,  # y轴数据
#          linestyle='-',  # 折线类型
#          linewidth=2,  # 折线宽度
#          color='#ff9999',  # 折线颜色
#          marker='o',  # 点的形状
#          markersize=6,  # 点的大小
#          markeredgecolor='black',  # 点的边框色
#          markerfacecolor='#ff9999',  # 点的填充色
#          label='2015年')  # 添加标签
# # 添加标题和坐标轴标签
# plt.title('2005年与2015年各省老龄化人数趋势图')
# plt.xticks(rotation=90)
# plt.xlabel('省份')
# plt.ylabel('老龄化人数')
# plt.grid()
# plt.tight_layout()
# plt.legend()  # 设置legend地位置，将其放在图外
# # plt.savefig('2005年与2015年各省老龄化人数趋势图.png')
# plt.show()
# print('图2', '-' * 80)


# ['北京' '天津' '河北' '山西' '内蒙古' '辽宁' '吉林' '黑龙江' '上海' '江苏' '浙江' '安徽' '福建' '江西'
#  '山东' '河南' '湖北' '湖南' '广东' '广西' '海南' '重庆' '四川' '贵州' '云南' '西藏' '陕西' '甘肃'
#  '青海' '宁夏' '新疆']
df = pd.read_excel("2002年到2019年各省老龄化数据.xlsx", sheet_name='Sheet1')
pd.set_option('display.max_columns', None)

# # 图3
# plt.bar(range(31), df['老年人口抚养比例'][:31].values, align='center', color='steelblue', alpha=0.8)
# plt.xticks(range(31), df['地区'][:31].values, rotation=90)
# plt.xlabel('省份')
# plt.ylabel('老年人口抚养比例')
# plt.title('各省老年人口抚养比例')
# # 为每个条形图添加数值标签
# for x, y in enumerate(df['老年人口抚养比例'][:31].values):
#     plt.text(x, y + 0.3, '%s' % round(y, 1), ha='center')
# plt.tight_layout()
# plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)  # 设置legend地位置，将其放在图外
# # plt.savefig('各省老年人口抚养比例图表.png')
# plt.show()
# print('图3', '-' * 80)

# # 图4
# # 六十五岁以上占比与老年人口抚养比例的关系
# x = df['老年人口抚养比例'].tolist()
# y = df['六十五岁以上占比'].tolist()
# t = np.arctan2(y, x)
# plt.scatter(x, y, c=t)
# plt.xlabel('老年人口抚养比例')
# plt.ylabel('六十五岁以上占比')
# plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)  # 设置legend地位置，将其放在图外
# # plt.savefig('老年人口抚养比例与六十五岁以上占比的散点图.png')
# plt.show()
# print('图4', '-' * 80)

# # 图5
# with sns.axes_style("white"):
#     sns.jointplot(x='六十五岁以上占比', y='老少比', data=df, kind='hex')
# plt.savefig('六十五岁以上占比与老少比之间关系图.png')
# plt.show()
# print('图5', '-' * 80)
#
# # 图6
# sns.kdeplot(df.loc[:, '六十五岁以上占比'], color="Red", shade=True)
# plt.savefig('六十五岁以上占比变化趋势图.png')
# plt.show()
# print('图6', '-' * 80)

# # 图7
# sns.relplot(x='六十五岁以上占比', y='老少比', data=df, hue='地区')
# plt.savefig('六十五岁以上占比与老少比区域差异图.png')
# plt.show()
# print('图7', '-' * 80)

a1 = sum(df2.values.tolist(), [])
value = [[i, j] for i in range(31) for j in range(18)]
values = np.hstack([np.array(value), np.array(a1).reshape(-1, 1)])
data = [[d[1], d[0], d[2] or "-"] for d in values]

# # 图8
# hm = HeatMap(init_opts=opts.InitOpts(width='1350px', height='750px'))
# hm.add_xaxis(df2.columns.tolist())
# hm.add_yaxis(
#     "老龄化人数", df2.index.tolist(), data, label_opts=opts.LabelOpts(position="middle")
# )
# hm.set_global_opts(
#     title_opts=opts.TitleOpts(title="2002年到2019年各省老龄化人数热力图"),
#     visualmap_opts=opts.VisualMapOpts(min_=120, max_=200000, is_calculable=True, orient="horizontal",
#                                       pos_left="center"),
# )
# make_snapshot(driver, hm.render("2002年到2019年各省老龄化人数热力图.html"), '2002年到2019年各省老龄化人数热力图.png')
# print('图8', '-' * 80)

# 图9
tl = Timeline()
for i in range(2002, 2020):
    data = df1[['地区', str(i) + '年']].values.tolist()
    print(data)
    map0 = (
        Map()
        .add("省份", df1[['地区', str(i) + '年']].values.tolist(), "china")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="{}年全国老龄化人口分布情况".format(i)),
            visualmap_opts=opts.VisualMapOpts(
                is_piecewise=True,
                pieces=[
                    {"min": 0, "max": 1000, "label": "1~10000", "color": "cyan"},
                    {"min": 1001, "max": 10000, "label": "10001~20000", "color": "yellow"},
                    {"min": 10000, "max": 80000, "label": "20001~50000", "color": "orange"},
                    {"min": 80001, "max": 100000, "label": "50001~80000", "color": "coral"},
                    {"min": 100001, "max": 200000, "label": "80001~12000", "color": "red"},
                ]), ))
    tl.add(map0, "{}年".format(i))
make_snapshot(driver, tl.render("2002年到2019年全国老龄化人口分布情况.html"), '2002年到2019年全国老龄化人口分布情况.png')
print('图9', '-' * 80)

older = df.groupby('地区')['六十五岁以上占比'].sum().sort_values(ascending=False)
# print(older)

# # 图10
# pie = Pie(init_opts=opts.InitOpts(width='1350px', height='750px'))
# pie.add(
#     "",
#     [
#         list(z)
#         for z in zip(
#         older.index,
#         older.values,
#     )
#     ],
#     center=["40%", "50%"],
#     rosetype="radius",
#     radius="55%"
# )
# pie.set_global_opts(
#     title_opts=opts.TitleOpts(title="2002年到2019年各省人口六十五岁以上占比"),
#     legend_opts=opts.LegendOpts(type_="scroll", pos_left="80%", orient="vertical"),
# )
# pie.set_series_opts(
#     label_opts=opts.LabelOpts(formatter="{b}:{c}")
# )
# make_snapshot(driver, pie.render("2002年到2019年各省人口六十五岁以上占比.html"), '2002年到2019年各省人口六十五岁以上占比.png')
# print('图10', '-' * 80)


# # 数据预处理-----------------------------------------------------------------------------------------------------------
# # 查看重复值
# print('重复值:', df.duplicated().any())  # 无重复值，无需处理
# # 查看缺失值
# print('缺失值', df.isnull().any())  # 无缺失值，无需处理
# # 数据探索
# print('df.shape:', '-' * 50, '\n', df.shape)
# print('df.info:', '-' * 50, '\n', df.info())
# print('df.dtypes:', '-' * 50, '\n', df.dtypes)
# print('df.head:', '-' * 50, '\n', df.head())
# print('df.describe:', '-' * 50, '\n', df.describe())
# df.mean()  # 平均值
# df.std()  # 标准差
# df.var()  # 方差
# df.median()  # 中位数
# df.mode()  # 众数

# # 计算各个特征的相关系数
# corrDf = df.corr()
# print('特征的相关系数', '-' * 50, '\n', corrDf)
# colormap = plt.cm.viridis
# plt.figure(figsize=(10, 10))
# plt.title('Pearson Correaltion of Feature', y=1.05, size=15)
# sns.heatmap(corrDf, linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# plt.show()
# data = pd.read_excel('2002年-2021年各省老龄化趋势.xlsx', sheet_name='Sheet1')
# data.dropna(inplace=True)
# X = data.iloc[:, 3:].values
# SSE = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     kmeans.fit(X)
#     SSE.append(kmeans.inertia_)
# plt.plot(range(1, 11), SSE)
# plt.title('The Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('SSE')
# plt.savefig('./images/k.png')
# plt.show()

y_kmeans = KMeans(n_clusters=4).fit(X)

# # plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='magenta', label='Careful')
# # plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='yellow', label='Standard')
# # plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='Target')
# # plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Careless')
# # plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='burlywood', label='Sensible')
# plt.scatter(y_kmeans.cluster_centers_[:, 0], y_kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
# plt.title('Cluster of Clients')
# plt.xlabel('纬度')
# plt.ylabel('经度')
# plt.legend()
# plt.savefig('./images/kmeans.png')
# plt.show()
#
# sns.lmplot(x='人口数_人口抽样调查_人', y='老龄人口占比', data=data, fit_reg=True, hue='地区')
# plt.tight_layout()
# plt.savefig('./images/chayi.png')
# plt.show()
