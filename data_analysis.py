import pandas as pd

# 1. 读取 CSV 数据（在线鸢尾花数据集）
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
df = pd.read_csv(url, names=columns)

# 2. 查看数据基本信息
print("数据形状：", df.shape)
print("\n前5行：")
print(df.head())

# 3. 统计每种花的数量
print("\n各品种数量统计：")
print(df['species'].value_counts())

# 4. 计算每种花的平均花瓣长度
print("\n各品种平均花瓣长度：")
print(df.groupby('species')['petal_length'].mean())

# 5. 保存处理后的数据到本地 CSV
df.to_csv('iris_processed.csv', index=False)
print("\n数据已保存为 iris_processed.csv")