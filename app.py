import pandas as pd
import numpy as np
import streamlit as st
from surprise import Dataset, Reader, SVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="混合电影推荐系统", page_icon="🍿")
st.title("🍿 混合架构电影推荐系统 (Hybrid RecSys)")

GENRE_DICT = {
    'unknown': '未知', 'Action': '动作', 'Adventure': '冒险', 'Animation': '动画',
    "Children's": '儿童', 'Comedy': '喜剧', 'Crime': '犯罪', 'Documentary': '纪录片',
    'Drama': '剧情', 'Fantasy': '奇幻', 'Film-Noir': '黑色电影', 'Horror': '恐怖',
    'Musical': '音乐', 'Mystery': '悬疑', 'Romance': '爱情', 'Sci-Fi': '科幻',
    'Thriller': '惊悚', 'War': '战争', 'Western': '西部'
}


@st.cache_resource
def load_data_and_models():
    # 1. 加载数据
    df_movies = pd.read_csv('movies.csv')
    df_ratings = pd.read_csv('ratings.csv')

    # 2. 训练 SVD 协同过滤模型 (给老用户用)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df_ratings[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    algo_svd = SVD()
    algo_svd.fit(trainset)

    # 3. 计算基于内容的相似度矩阵 (给新用户用)
    df_movies['genres_str'] = df_movies['genres'].str.replace('|', ' ')
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    genre_matrix = cv.fit_transform(df_movies['genres_str'])
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

    # 4. 计算热度，过滤冷门
    movie_popularity = df_ratings['movieId'].value_counts().to_dict()

    return df_movies, df_ratings, algo_svd, cosine_sim, movie_popularity


df_movies, df_ratings, algo_svd, cosine_sim, movie_popularity = load_data_and_models()

# ================= 侧边栏：双模式切换 =================
st.sidebar.header("推荐模式选择")
mode = st.sidebar.radio("请选择你的身份：", ["新用户 (冷启动 - 基于内容)", "老用户 (基于协同过滤)"])

top_n = st.sidebar.slider("推荐数量", min_value=5, max_value=20, value=10)

# ---------------- 模式一：新用户（基于内容） ----------------
if mode == "新用户 (冷启动 - 基于内容)":
    st.sidebar.subheader("构建你的口味画像")
    popular_movies = df_movies[
        df_movies['movieId'].isin([mid for mid, count in movie_popularity.items() if count > 100])]
    movie_list = popular_movies['title'].tolist()

    selected_movies = st.sidebar.multiselect(
        "选择你看过的电影 (建议选 2-3 部)：",
        movie_list,
        default=["Toy Story (1995)"] if "Toy Story (1995)" in movie_list else None
    )

    genre_options_cn = list(GENRE_DICT.values())
    selected_genre = st.sidebar.selectbox("这次你想看什么类型？", ["不限"] + genre_options_cn)

    if st.sidebar.button("生成推荐") and selected_movies:
        movie_indices = [df_movies[df_movies['title'] == title].index[0] for title in selected_movies]
        sim_scores_list = [cosine_sim[idx] for idx in movie_indices]
        avg_sim_scores = np.mean(sim_scores_list, axis=0)

        sim_scores = list(enumerate(avg_sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        recommended = []
        for movie_idx, score in sim_scores:
            info = df_movies.iloc[movie_idx]
            if info['title'] in selected_movies: continue
            if movie_popularity.get(info['movieId'], 0) < 50: continue

            genres_cn_list = [GENRE_DICT.get(g, g) for g in info['genres'].split('|')]
            if selected_genre != "不限" and selected_genre not in genres_cn_list: continue

            recommended.append((info, score, genres_cn_list))
            if len(recommended) >= top_n: break

        st.subheader("🎯 基于内容特征为你推荐：")
        if selected_genre != "不限":
            st.write(f"*(已按你的要求，仅筛选 **{selected_genre}** 类型)*")
        st.write("---")

        for rank, (info, score, genres_cn_list) in enumerate(recommended, 1):
            st.write(f"**Top {rank}: {info['title']}**")
            st.write(f"类型：{'/'.join(genres_cn_list)} | 内容匹配度：{score * 100:.1f}%")
            st.write("---")

# ---------------- 模式二：老用户（协同过滤） ----------------
else:
    st.sidebar.subheader("输入你的用户 ID")
    max_user = int(df_ratings['userId'].max())
    user_id = st.sidebar.number_input(f"请输入 ID (1-{max_user})", min_value=1, max_value=max_user, value=1)

    if st.sidebar.button("生成推荐"):
        watched_movies = set(df_ratings[df_ratings['userId'] == user_id]['movieId'].tolist())
        all_movies = set(df_movies['movieId'].tolist())
        unwatched_movies = list(all_movies - watched_movies)

        # 分析老用户最喜欢的类型
        user_ratings = df_ratings[(df_ratings['userId'] == user_id) & (df_ratings['rating'] >= 4)]
        liked_movies_info = df_movies[df_movies['movieId'].isin(user_ratings['movieId'])]
        all_genres = [genre for sublist in liked_movies_info['genres'].str.split('|') for genre in sublist if
                      isinstance(sublist, list)]
        favorite_genre = pd.Series(all_genres).value_counts().index[0] if len(all_genres) > 0 else "Drama"
        favorite_genre_cn = GENRE_DICT.get(favorite_genre, favorite_genre)

        predictions = []
        for movie_id in unwatched_movies:
            if movie_popularity.get(movie_id, 0) < 50: continue  # 过滤冷门

            pred = algo_svd.predict(user_id, movie_id)
            movie_info = df_movies[df_movies['movieId'] == movie_id].iloc[0]

            # 特征重排：如果符合用户喜好类型，加0.2分
            is_fav = favorite_genre in str(movie_info['genres'])
            final_score = pred.est + (0.2 if is_fav else 0)
            predictions.append((movie_id, pred.est, final_score, is_fav))

        predictions.sort(key=lambda x: x[2], reverse=True)

        st.subheader(f"🎬 基于协同过滤，为用户 {user_id} 推荐：")
        st.write(f"*(分析得出，该用户历史最偏爱的类型是：**{favorite_genre_cn}**)*")
        st.write("---")

        for rank, (mid, est, final, is_fav) in enumerate(predictions[:top_n], 1):
            info = df_movies[df_movies['movieId'] == mid].iloc[0]
            genres_cn = "/".join([GENRE_DICT.get(g, g) for g in info['genres'].split('|')])
            tag = " 🔥(符合偏好)" if is_fav else ""
            st.write(f"**Top {rank}: {info['title']}**{tag}")
            st.write(f"类型：{genres_cn} | 预测评分：🌟 {est:.2f} 分")
            st.write("---")