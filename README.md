# 🍿 基于内容与协同过滤的混合电影推荐系统

本项目基于 MovieLens 数据集，构建了交互式个性化电影推荐系统。针对推荐系统中常见的“冷启动”与“长尾效应”痛点，实现了基于模型协同过滤与基于文本内容特征融合的混合架构。

## 🌟 核心功能
- **新用户冷启动（Content-Based）**：支持多选历史喜好电影与指定目标类型，基于电影文本特征计算余弦相似度，实现千人千面的即时推荐。
- **老用户精准召回（SVD）**：基于矩阵分解算法挖掘用户-物品隐向量，并结合历史类型偏好进行特征重排。
- **长尾物品过滤**：通过计算物品热度，自动剔除交互频次过低的冷门电影，提升推荐质量。
- **前端可视化**：基于 Streamlit 搭建轻量级交互式 Web 界面。

## 🛠️ 技术栈
- **数据处理**：Pandas, Numpy
- **算法框架**：Scikit-learn (CountVectorizer, Cosine Similarity), Surprise (SVD)
- **前端交互**：Streamlit

## 🚀 快速开始
1. 安装依赖：`pip install pandas numpy scikit-learn surprise streamlit`
2. 确保 `movies.csv` 和 `ratings.csv` 在当前目录下。
3. 运行程序：`streamlit run app.py`
