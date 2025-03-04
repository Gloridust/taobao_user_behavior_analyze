# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['AppleGothic']  # macOS中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ====================== 数据预处理 ======================
def load_data():
    # 读取数据并添加列名
    df = pd.read_csv('UserBehavior.csv', 
                    names=['user_id', 'item_id', 'category_id', 'behavior', 'timestamp'])
    
    # 时间戳转换（处理秒级时间戳）
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['weekday'] = df['datetime'].dt.weekday + 1  # Monday=1, Sunday=7
    
    # 数据清洗
    print("原始数据量:", len(df))
    
    # 去除时间范围外的数据（根据论文描述数据应为2017-11-25至2017-12-03）
    valid_start = pd.to_datetime('2017-11-25').date()
    valid_end = pd.to_datetime('2017-12-03').date()
    df = df[(df['date'] >= valid_start) & (df['date'] <= valid_end)]
    
    print("有效数据量:", len(df))
    return df

# ====================== 分析模块 ======================
def behavioral_analysis(df):
    """用户行为基础分析"""
    # 1. 行为分布
    behavior_dist = df['behavior'].value_counts().reindex(['pv', 'cart', 'fav', 'buy'])
    
    # 2. 转化漏斗分析（改进版）
    funnel = pd.DataFrame()
    funnel['total'] = behavior_dist
    funnel['step_rate'] = funnel['total'] / funnel['total'].shift(1)
    funnel['overall_rate'] = funnel['total'] / funnel.loc['pv', 'total']
    
    # 3. 时间模式分析
    time_pattern = df.pivot_table(index='hour', columns='behavior', 
                                values='user_id', aggfunc='count', fill_value=0)
    
    return {'behavior_dist': behavior_dist, 
           'funnel': funnel, 
           'time_pattern': time_pattern}

def user_clustering(df):
    """基于RFM模型的用户分群（改进版）"""
    # 计算RFM指标
    rfm = df.groupby('user_id').agg(
        last_activity=('datetime', 'max'),      # Recency
        frequency=('behavior', 'count'),        # Frequency
        monetary=('behavior', lambda x: (x == 'buy').sum())  # Monetary(用购买次数代替金额
    ).reset_index()
    
    # 计算Recency天数（相对于数据最新时间）
    now = df['datetime'].max()
    rfm['recency'] = (now - rfm['last_activity']).dt.days
    
    # 数据标准化
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
    
    # K-means聚类
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # 分析聚类特征
    cluster_profile = rfm.groupby('cluster').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean'
    })
    
    return {'rfm': rfm, 'cluster_profile': cluster_profile}

def pattern_mining(df):
    """用户行为模式挖掘"""
    # 关联规则挖掘（类目级别）
    purchased_categories = df[df['behavior'] == 'buy'].groupby('user_id')['category_id'].apply(list)
    
    te = TransactionEncoder()
    te_ary = te.fit_transform(purchased_categories)
    df_te = pd.DataFrame(te_ary, columns=te.columns_)
    
    # 频繁项集挖掘
    frequent_itemsets = apriori(df_te, min_support=0.005, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    # 时间序列模式
    time_series = df[df['behavior'] == 'buy'].groupby('date').size()
    
    return {'association_rules': rules, 'time_series': time_series}

# ====================== 可视化模块 ======================
def plot_behavioral_analysis(results):
    """可视化用户行为分析结果"""
    # 行为分布
    plt.figure(figsize=(10, 6))
    results['behavior_dist'].plot(kind='bar', color='skyblue')
    plt.title('用户行为类型分布')
    plt.xlabel('行为类型')
    plt.ylabel('发生次数')
    plt.xticks(rotation=0)
    plt.show()
    
    # 转化漏斗
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=range(4), y='overall_rate', data=results['funnel'], 
                marker='o', color='darkorange')
    plt.title('用户行为转化漏斗')
    plt.xticks([0,1,2,3], ['浏览', '加购', '收藏', '购买'])
    plt.ylabel('总体转化率')
    plt.show()
    
    # 时间模式热力图
    plt.figure(figsize=(12, 6))
    sns.heatmap(results['time_pattern'], cmap='YlGnBu')
    plt.title('用户行为时间分布热力图')
    plt.show()

def plot_clusters(cluster_profile):
    """可视化用户分群结果"""
    plt.figure(figsize=(10, 6))
    radar_data = cluster_profile.reset_index()
    radar_data = pd.melt(radar_data, id_vars='cluster', 
                        value_vars=['recency', 'frequency', 'monetary'])
    
    sns.lineplot(x='variable', y='value', hue='cluster', data=radar_data, 
                marker='o', palette='tab10')
    plt.title('用户分群特征雷达图')
    plt.xlabel('指标')
    plt.ylabel('标准化值')
    plt.show()

# ====================== 高级分析 ======================  
def predict_purchase(df):
    """用户购买行为预测"""
    # 构建特征矩阵
    user_features = df.groupby('user_id').agg(
        pv_count=('behavior', lambda x: (x == 'pv').sum()),
        cart_count=('behavior', lambda x: (x == 'cart').sum()),
        fav_count=('behavior', lambda x: (x == 'fav').sum()),
        last_activity=('datetime', 'max')
    ).reset_index()
    
    # 创建标签（未来24小时是否购买）
    latest_time = df['datetime'].max()
    cutoff_time = latest_time - pd.Timedelta(hours=24)
    
    # 获取每个用户最后购买时间
    last_purchase = df[df['behavior'] == 'buy'].groupby('user_id')['datetime'].max()
    user_features['label'] = (last_purchase >= cutoff_time).astype(int)
    
    # 数据预处理
    user_features.fillna(0, inplace=True)
    X = user_features[['pv_count', 'cart_count', 'fav_count']]
    y = user_features['label']
    
    # 训练预测模型
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    
    # 模型评估
    print(classification_report(y_test, rf.predict(X_test)))
    
    return rf

# ====================== 主程序 ======================
if __name__ == "__main__":
    # 数据加载与预处理
    df = load_data()
    
    # 基础分析
    behavior_results = behavioral_analysis(df)
    plot_behavioral_analysis(behavior_results)
    
    # 用户分群
    cluster_results = user_clustering(df)
    plot_clusters(cluster_results['cluster_profile'])
    
    # 模式挖掘
    mining_results = pattern_mining(df)
    print("Top关联规则：\n", mining_results['association_rules'].sort_values('lift', ascending=False).head())
    
    # 购买预测
    purchase_model = predict_purchase(df)
    
    # 保存分析结果（供论文使用）
    behavior_results['funnel'].to_csv('result/funnel_analysis.csv')
    cluster_results['rfm'].to_csv('result/user_clusters.csv')
    mining_results['association_rules'].to_csv('result/association_rules.csv')