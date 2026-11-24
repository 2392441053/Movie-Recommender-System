import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# 1. 加载数据
print("正在加载 MovieLens 100K 数据集...")
# 告诉 Surprise，评分范围是 1 到 5
reader = Reader(line_format='user item rating timestamp', sep='\t', rating_scale=(1, 5))
data = Dataset.load_from_file('ml-100k/u.data', reader=reader)

# 2. 使用经典的 SVD 算法（矩阵分解）
algo = SVD()

# 3. 交叉验证（评估模型）
print("开始训练并进行 3 折交叉验证（大概需要几十秒）...")
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)