import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体（防止中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=columns)

# 图1：柱状图——各品种平均花瓣长度
avg = df.groupby('species')['petal_length'].mean()
plt.figure(figsize=(8, 5))
plt.bar(avg.index, avg.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
plt.title('各品种鸢尾花平均花瓣长度')
plt.xlabel('品种')
plt.ylabel('花瓣长度 (cm)')
plt.savefig('iris_bar_chart.png')
plt.show()

# 图2：散点图——花瓣长度 vs 花瓣宽度
plt.figure(figsize=(8, 5))
for species in df['species'].unique():
    subset = df[df['species'] == species]
    plt.scatter(subset['petal_length'], subset['petal_width'], label=species)
plt.title('花瓣长度 vs 花瓣宽度')
plt.xlabel('花瓣长度 (cm)')
plt.ylabel('花瓣宽度 (cm)')
plt.legend()
plt.savefig('iris_scatter.png')
plt.show()

# 图3：折线图——前50行花萼长度变化
plt.figure(figsize=(10, 5))
plt.plot(df['sepal_length'].values[:50], label='前50行花萼长度')
plt.title('花萼长度变化趋势')
plt.xlabel('样本编号')
plt.ylabel('花萼长度 (cm)')
plt.legend()
plt.savefig('iris_line.png')
plt.show()

print("三张图表已保存：iris_bar_chart.png、iris_scatter.png、iris_line.png")# 三张图表分别展示了柱状图、散点图和折线图