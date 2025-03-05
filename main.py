#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
淘宝用户行为分析系统
----------------------------
本脚本分析淘宝用户行为数据，提取市场营销、产品推荐和平台优化的见解。

数据集:
- UserBehavior.csv: 用户行为数据（用户ID、商品ID、类目ID、行为类型、时间戳）
- sam_tianchi_2014002_rec_tmall_log.csv: 天猫用户商品交互数据

日期: 2025年3月5日
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import platform
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings

# 使用用户确认可行的字体设置方式
plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 抑制警告信息
warnings.filterwarnings('ignore')

# 设置可视化样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)


def create_directory(directory_path):
    """如果目录不存在，则创建目录"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"已创建目录: {directory_path}")


def load_and_understand_data():
    """加载数据集并进行初步理解"""
    print("===== 数据加载与理解 =====")
    
    # 创建结果目录
    create_directory("result")
    create_directory("result/figures")
    create_directory("result/tables")
    
    # 加载 UserBehavior 数据集
    print("加载 UserBehavior 数据集...")
    try:
        userbehavior_df = pd.read_csv('dataset/UserBehavior.csv', header=None, 
                                    names=['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp'])
        print(f"UserBehavior 数据集加载成功，包含 {userbehavior_df.shape[0]} 行和 {userbehavior_df.shape[1]} 列。")
    except Exception as e:
        print(f"加载 UserBehavior 数据集时出错: {e}")
        userbehavior_df = None
    
    # 加载天猫日志数据集
    print("加载天猫日志数据集...")
    try:
        tmall_df = pd.read_csv('dataset/sam_tianchi_2014002_rec_tmall_log.csv')
        print(f"天猫数据集加载成功，包含 {tmall_df.shape[0]} 行和 {tmall_df.shape[1]} 列。")
    except Exception as e:
        print(f"加载天猫数据集时出错: {e}")
        tmall_df = None
    
    return userbehavior_df, tmall_df


def explore_and_clean_userbehavior(df):
    """探索并清洗 UserBehavior 数据集"""
    if df is None:
        return None
    
    print("\n===== UserBehavior 数据集探索与清洗 =====")
    
    # 检查基本信息
    print("\n数据集基本信息:")
    df_info = df.info()
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n缺失值数量:")
    print(missing_values)
    
    # 检查重复值
    duplicates = df.duplicated().sum()
    print(f"\n重复行数量: {duplicates}")
    
    # 删除重复值（如果有）
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"已删除 {duplicates} 行重复数据。")
    
    # 将时间戳转换为日期时间格式
    print("\n转换时间戳为日期时间格式...")
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 检查行为类型
    behavior_counts = df['behavior_type'].value_counts()
    print("\n行为类型及数量:")
    print(behavior_counts)
    
    # 保存清洗后的数据
    print("\n保存清洗后的 UserBehavior 数据...")
    df.to_csv('result/tables/cleaned_userbehavior.csv', index=False)
    
    # 生成数据描述统计表
    desc_stats = df.describe().T
    desc_stats.to_csv('result/tables/userbehavior_descriptive_stats.csv')
    
    return df


def explore_and_clean_tmall(df):
    """探索并清洗天猫数据集"""
    if df is None:
        return None
    
    print("\n===== 天猫数据集探索与清洗 =====")
    
    # 检查基本信息
    print("\n数据集基本信息:")
    df_info = df.info()
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    print("\n缺失值数量:")
    print(missing_values)
    
    # 检查重复值
    duplicates = df.duplicated().sum()
    print(f"\n重复行数量: {duplicates}")
    
    # 删除重复值（如果有）
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"已删除 {duplicates} 行重复数据。")
    
    # 转换时间格式
    print("\n转换时间为日期时间格式...")
    df['vtime'] = pd.to_datetime(df['vtime'])
    df['date'] = df['vtime'].dt.date
    df['hour'] = df['vtime'].dt.hour
    df['day_of_week'] = df['vtime'].dt.dayofweek
    
    # 检查行为类型
    action_counts = df['action'].value_counts()
    print("\n行为类型及数量:")
    print(action_counts)
    
    # 保存清洗后的数据
    print("\n保存清洗后的天猫数据...")
    df.to_csv('result/tables/cleaned_tmall.csv', index=False)
    
    # 生成数据描述统计表
    desc_stats = df.describe().T
    desc_stats.to_csv('result/tables/tmall_descriptive_stats.csv')
    
    return df


def analyze_user_behavior(df):
    """分析 UserBehavior 数据集中的用户行为模式"""
    if df is None:
        return
    
    print("\n===== 用户行为分析 (UserBehavior 数据集) =====")
    
    # 1. 基本统计
    print("\n计算基本统计数据...")
    total_users = df['user_id'].nunique()
    total_items = df['item_id'].nunique()
    total_categories = df['category_id'].nunique()
    total_behaviors = df.shape[0]
    
    # 创建统计摘要表
    summary_df = pd.DataFrame({
        '指标': ['总用户数', '总商品数', '总类目数', '总行为数'],
        '数值': [total_users, total_items, total_categories, total_behaviors]
    })
    print(summary_df)
    summary_df.to_csv('result/tables/basic_statistics.csv', index=False)
    
    # 2. 行为分布
    behavior_distribution = df['behavior_type'].value_counts().reset_index()
    behavior_distribution.columns = ['行为类型', '数量']
    behavior_distribution['百分比'] = behavior_distribution['数量'] / behavior_distribution['数量'].sum() * 100
    behavior_distribution.to_csv('result/tables/behavior_distribution.csv', index=False)
    
    # 可视化行为分布
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(10, 6))
    sns.barplot(x='行为类型', y='数量', data=behavior_distribution)
    plt.title('用户行为分布')
    plt.ylabel('数量')
    plt.tight_layout()
    plt.savefig('result/figures/behavior_distribution.png')
    plt.close()
    
    # 3. 每日行为模式
    daily_behaviors = df.groupby(['date', 'behavior_type']).size().unstack().fillna(0)
    daily_behaviors.to_csv('result/tables/daily_behaviors.csv')
    
    # 可视化每日模式
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 7))
    daily_behaviors.plot(figsize=(14, 7))
    plt.title('每日用户行为模式')
    plt.xlabel('日期')
    plt.ylabel('行为数量')
    plt.legend(title='行为类型')
    plt.tight_layout()
    plt.savefig('result/figures/daily_behavior_patterns.png')
    plt.close()
    
    # 4. 每小时行为模式
    hourly_behaviors = df.groupby(['hour', 'behavior_type']).size().unstack().fillna(0)
    hourly_behaviors.to_csv('result/tables/hourly_behaviors.csv')
    
    # 可视化小时模式
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 7))
    hourly_behaviors.plot(figsize=(14, 7))
    plt.title('每小时用户行为模式')
    plt.xlabel('小时')
    plt.ylabel('行为数量')
    plt.legend(title='行为类型')
    plt.xticks(range(24))
    plt.tight_layout()
    plt.savefig('result/figures/hourly_behavior_patterns.png')
    plt.close()
    
    # 5. 星期几行为模式
    dow_behaviors = df.groupby(['day_of_week', 'behavior_type']).size().unstack().fillna(0)
    # 将数字日期映射为名称
    dow_map = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 
               4: '周五', 5: '周六', 6: '周日'}
    dow_behaviors.index = dow_behaviors.index.map(dow_map)
    dow_behaviors.to_csv('result/tables/day_of_week_behaviors.csv')
    
    # 可视化星期几模式
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 7))
    dow_behaviors.plot(kind='bar', figsize=(14, 7))
    plt.title('星期几用户行为模式')
    plt.xlabel('星期')
    plt.ylabel('行为数量')
    plt.legend(title='行为类型')
    plt.tight_layout()
    plt.savefig('result/figures/day_of_week_behavior_patterns.png')
    plt.close()
    
    # 6. 转化漏斗分析
    print("\n计算转化漏斗...")
    funnel_df = behavior_distribution.copy()
    funnel_df = funnel_df.sort_values('数量', ascending=False)
    
    # 计算不同阶段间的转化率
    conversion_rates = []
    behavior_order = ['pv', 'cart', 'fav', 'buy']  # 典型用户旅程
    
    behavior_counts = {behavior: count for behavior, count in zip(behavior_distribution['行为类型'], behavior_distribution['数量'])}
    
    for i in range(len(behavior_order) - 1):
        current = behavior_order[i]
        next_step = behavior_order[i+1]
        
        if current in behavior_counts and next_step in behavior_counts:
            rate = (behavior_counts[next_step] / behavior_counts[current]) * 100
            conversion_rates.append({
                '从': current,
                '到': next_step,
                '转化率 (%)': rate
            })
    
    conversion_df = pd.DataFrame(conversion_rates)
    conversion_df.to_csv('result/tables/conversion_rates.csv', index=False)
    
    # 7. 热门类目分析
    print("\n分析热门类目...")
    top_categories = df.groupby('category_id').size().reset_index(name='count')
    top_categories = top_categories.sort_values('count', ascending=False).head(20)
    top_categories.to_csv('result/tables/top_categories.csv', index=False)
    
    # 可视化热门类目
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 8))
    sns.barplot(x='category_id', y='count', data=top_categories)
    plt.title('前20个热门商品类目')
    plt.xlabel('类目ID')
    plt.ylabel('交互次数')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('result/figures/top_categories.png')
    plt.close()
    
    # 8. 用户购买频率分析
    print("\n分析用户购买行为...")
    user_purchases = df[df['behavior_type'] == 'buy'].groupby('user_id').size().reset_index(name='purchase_count')
    purchase_frequency = user_purchases['purchase_count'].value_counts().reset_index()
    purchase_frequency.columns = ['购买次数', '用户数']
    purchase_frequency = purchase_frequency.sort_values('购买次数')
    purchase_frequency.to_csv('result/tables/purchase_frequency.csv', index=False)
    
    # 可视化购买频率
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 7))
    sns.barplot(x='购买次数', y='用户数', data=purchase_frequency.head(20))
    plt.title('用户购买频率分布')
    plt.xlabel('购买次数')
    plt.ylabel('用户数')
    plt.tight_layout()
    plt.savefig('result/figures/purchase_frequency.png')
    plt.close()
    
    # 9. 用户参与度（每个用户的行为计数）
    print("\n分析用户参与度...")
    user_engagement = df.groupby('user_id')['behavior_type'].count().reset_index(name='behavior_count')
    engagement_stats = user_engagement['behavior_count'].describe().reset_index()
    engagement_stats.columns = ['统计量', '数值']
    engagement_stats.to_csv('result/tables/user_engagement_statistics.csv', index=False)
    
    # 按参与度水平对用户进行分类
    user_engagement['engagement_level'] = pd.qcut(user_engagement['behavior_count'], 
                                                q=5, 
                                                labels=['很低', '低', '中', '高', '很高'])
    engagement_distribution = user_engagement['engagement_level'].value_counts().reset_index()
    engagement_distribution.columns = ['参与度水平', '用户数']
    engagement_distribution = engagement_distribution.sort_values('参与度水平')
    engagement_distribution.to_csv('result/tables/user_engagement_distribution.csv', index=False)
    
    # 可视化参与度分布
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(12, 7))
    sns.barplot(x='参与度水平', y='用户数', data=engagement_distribution)
    plt.title('用户参与度水平分布')
    plt.ylabel('用户数')
    plt.tight_layout()
    plt.savefig('result/figures/user_engagement_distribution.png')
    plt.close()
    
    print("用户行为分析完成并保存到结果目录。")


def analyze_tmall_data(df):
    """分析天猫数据集中的模式"""
    if df is None:
        return
    
    print("\n===== 天猫用户行为分析 =====")
    
    # 1. 基本统计
    print("\n计算基本统计数据...")
    total_users = df['user_id'].nunique()
    total_items = df['item_id'].nunique()
    total_actions = df.shape[0]
    
    # 创建统计摘要表
    summary_df = pd.DataFrame({
        '指标': ['总用户数', '总商品数', '总行为数'],
        '数值': [total_users, total_items, total_actions]
    })
    print(summary_df)
    summary_df.to_csv('result/tables/tmall_basic_statistics.csv', index=False)
    
    # 2. 行为分布
    action_distribution = df['action'].value_counts().reset_index()
    action_distribution.columns = ['行为类型', '数量']
    action_distribution['百分比'] = action_distribution['数量'] / action_distribution['数量'].sum() * 100
    action_distribution.to_csv('result/tables/tmall_action_distribution.csv', index=False)
    
    # 可视化行为分布
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(10, 6))
    sns.barplot(x='行为类型', y='数量', data=action_distribution)
    plt.title('天猫用户行为分布')
    plt.ylabel('数量')
    plt.tight_layout()
    plt.savefig('result/figures/tmall_action_distribution.png')
    plt.close()
    
    # 3. 每日行为模式
    daily_actions = df.groupby(['date', 'action']).size().unstack().fillna(0)
    daily_actions.to_csv('result/tables/tmall_daily_actions.csv')
    
    # 可视化每日模式
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 7))
    daily_actions.plot(figsize=(14, 7))
    plt.title('每日天猫用户行为模式')
    plt.xlabel('日期')
    plt.ylabel('行为数量')
    plt.legend(title='行为类型')
    plt.tight_layout()
    plt.savefig('result/figures/tmall_daily_action_patterns.png')
    plt.close()
    
    # 4. 每小时行为模式
    hourly_actions = df.groupby(['hour', 'action']).size().unstack().fillna(0)
    hourly_actions.to_csv('result/tables/tmall_hourly_actions.csv')
    
    # 可视化小时模式
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 7))
    hourly_actions.plot(figsize=(14, 7))
    plt.title('每小时天猫用户行为模式')
    plt.xlabel('小时')
    plt.ylabel('行为数量')
    plt.legend(title='行为类型')
    plt.xticks(range(24))
    plt.tight_layout()
    plt.savefig('result/figures/tmall_hourly_action_patterns.png')
    plt.close()
    
    # 5. 星期几行为模式
    dow_actions = df.groupby(['day_of_week', 'action']).size().unstack().fillna(0)
    # 将数字日期映射为名称
    dow_map = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 
               4: '周五', 5: '周六', 6: '周日'}
    dow_actions.index = dow_actions.index.map(dow_map)
    dow_actions.to_csv('result/tables/tmall_day_of_week_actions.csv')
    
    # 可视化星期几模式
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 7))
    dow_actions.plot(kind='bar', figsize=(14, 7))
    plt.title('星期几天猫用户行为模式')
    plt.xlabel('星期')
    plt.ylabel('行为数量')
    plt.legend(title='行为类型')
    plt.tight_layout()
    plt.savefig('result/figures/tmall_day_of_week_action_patterns.png')
    plt.close()
    
    # 6. 用户商品交互频率
    print("\n分析用户-商品交互频率...")
    user_item_interactions = df.groupby(['user_id', 'item_id']).size().reset_index(name='interaction_count')
    interaction_stats = user_item_interactions['interaction_count'].describe().reset_index()
    interaction_stats.columns = ['统计量', '数值']
    interaction_stats.to_csv('result/tables/tmall_interaction_statistics.csv', index=False)
    
    # 可视化交互频率分布
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(12, 7))
    sns.histplot(user_item_interactions['interaction_count'], kde=True, bins=30)
    plt.title('用户-商品交互频率分布')
    plt.xlabel('交互次数')
    plt.ylabel('数量')
    plt.tight_layout()
    plt.savefig('result/figures/tmall_interaction_frequency.png')
    plt.close()
    
    print("天猫行为分析完成并保存到结果目录。")


def user_segmentation(df):
    """使用RFM分析执行用户分群"""
    if df is None or 'buy' not in df['behavior_type'].values:
        print("无法执行用户分群：缺少数据或没有购买行为。")
        return
    
    print("\n===== 用户分群分析 =====")
    
    # 提取有购买行为的用户
    purchase_df = df[df['behavior_type'] == 'buy']
    
    # 计算RFM指标
    # 最近一次购买 - 距离最后一次购买的天数
    max_date = df['timestamp'].max().date()
    user_recency = purchase_df.groupby('user_id')['timestamp'].max().reset_index()
    user_recency['recency'] = user_recency['timestamp'].apply(lambda x: (max_date - x.date()).days)
    
    # 频率 - 购买次数
    user_frequency = purchase_df.groupby('user_id').size().reset_index(name='frequency')
    
    # 合并RFM指标
    rfm_df = user_recency.merge(user_frequency, on='user_id')
    rfm_df = rfm_df[['user_id', 'recency', 'frequency']]
    
    # 计算各指标的四分位数
    # 使用duplicates='drop'解决重复边界值问题
    rfm_df['R_Score'] = pd.qcut(rfm_df['recency'], 4, labels=[4, 3, 2, 1], duplicates='drop')  # 反向（值越低越好）
    
    # 由于frequency可能有大量重复值，我们使用两种处理方式
    try:
        # 方法1：尝试使用duplicates='drop'
        rfm_df['F_Score'] = pd.qcut(rfm_df['frequency'], 4, labels=[1, 2, 3, 4], duplicates='drop')
    except ValueError:
        # 方法2：如果仍然失败，则使用rank函数先对数据进行排名，再分箱
        rfm_df['frequency_rank'] = rfm_df['frequency'].rank(method='first')
        rfm_df['F_Score'] = pd.qcut(rfm_df['frequency_rank'], 4, labels=[1, 2, 3, 4])
    
    # 计算RFM得分
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str)
    
    # 定义分群
    segments = {
        '44': '冠军客户',
        '43': '忠诚客户',
        '42': '潜力忠诚客户',
        '41': '新客户',
        '34': '有前景客户',
        '33': '需要关注客户',
        '32': '面临流失客户',
        '31': '即将沉睡客户',
        '24': '不能失去的客户',
        '23': '沉睡客户',
        '22': '流失客户',
        '21': '流失客户',
        '14': '不能失去的客户',
        '13': '沉睡客户',
        '12': '流失客户',
        '11': '流失客户'
    }
    
    rfm_df['Segment'] = rfm_df['RFM_Score'].map(segments)
    
    # 保存分群结果
    rfm_df.to_csv('result/tables/user_segmentation.csv', index=False)
    
    # 可视化分群分布
    segment_counts = rfm_df['Segment'].value_counts().reset_index()
    segment_counts.columns = ['客户群体', '用户数']
    
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 8))
    sns.barplot(x='客户群体', y='用户数', data=segment_counts)
    plt.title('用户分群分布')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('result/figures/user_segment_distribution.png')
    plt.close()
    
    # 分析不同分群的行为模式
    print("\n分析不同客户群体的行为模式...")
    
    # 如果数据量大则抽样以提高性能
    if rfm_df.shape[0] > 10000:
        rfm_sample = rfm_df.sample(n=10000, random_state=42)
    else:
        rfm_sample = rfm_df
    
    # 创建分群摘要
    segment_summary = rfm_sample.groupby('Segment').agg({
        'recency': ['mean', 'min', 'max'],
        'frequency': ['mean', 'min', 'max'],
        'user_id': 'count'
    })
    
    segment_summary.columns = ['平均最近购买天数', '最短最近购买天数', '最长最近购买天数', 
                              '平均购买频率', '最低购买频率', '最高购买频率', 
                              '用户数']
    segment_summary = segment_summary.reset_index()
    segment_summary.to_csv('result/tables/segment_summary.csv', index=False)
    
    print("用户分群分析完成并保存到结果目录。")


def analyze_purchase_patterns(df):
    """分析购买模式和商品相关性"""
    if df is None or 'buy' not in df['behavior_type'].values:
        print("无法执行购买模式分析：缺少数据或没有购买行为。")
        return
    
    print("\n===== 购买模式分析 =====")
    
    # 筛选购买行为
    purchase_df = df[df['behavior_type'] == 'buy']
    
    # 分析用户购买周期
    print("\n分析用户购买周期...")
    
    # 按用户分组，并排序购买时间
    user_purchases = purchase_df.groupby('user_id')['timestamp'].apply(list).reset_index()
    
    # 计算每位用户的购买间隔
    purchase_intervals = []
    for _, row in user_purchases.iterrows():
        timestamps = sorted(row['timestamp'])
        if len(timestamps) > 1:
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() / 86400 for i in range(len(timestamps)-1)]  # 转换为天
            user_id = row['user_id']
            avg_interval = np.mean(intervals)
            purchase_intervals.append({'user_id': user_id, 'avg_interval_days': avg_interval})
    
    if purchase_intervals:
        intervals_df = pd.DataFrame(purchase_intervals)
        interval_stats = intervals_df['avg_interval_days'].describe().reset_index()
        interval_stats.columns = ['统计量', '天数']
        interval_stats.to_csv('result/tables/purchase_interval_statistics.csv', index=False)
        
        # 可视化购买间隔分布
        plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
        plt.figure(figsize=(12, 7))
        sns.histplot(intervals_df['avg_interval_days'], kde=True, bins=30)
        plt.title('用户平均购买间隔分布')
        plt.xlabel('平均购买间隔（天）')
        plt.ylabel('用户数')
        plt.tight_layout()
        plt.savefig('result/figures/purchase_interval_distribution.png')
        plt.close()
    
    # 分析多品类购买行为
    print("\n分析多品类购买行为...")
    
    user_categories = purchase_df.groupby('user_id')['category_id'].nunique().reset_index()
    user_categories.columns = ['user_id', 'category_count']
    
    category_stats = user_categories['category_count'].describe().reset_index()
    category_stats.columns = ['统计量', '品类数']
    category_stats.to_csv('result/tables/user_category_statistics.csv', index=False)
    
    # 可视化用户购买品类数分布
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(12, 7))
    sns.countplot(data=user_categories, x='category_count')
    plt.title('用户购买品类数分布')
    plt.xlabel('购买品类数')
    plt.ylabel('用户数')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('result/figures/user_category_distribution.png')
    plt.close()
    
    # 分析品类共现
    print("\n分析品类共现...")
    
    # 抽样以提高性能
    if purchase_df.shape[0] > 100000:
        purchase_sample = purchase_df.sample(n=100000, random_state=42)
    else:
        purchase_sample = purchase_df
    
    # 计算每个用户购买的品类组合
    user_category_sets = purchase_sample.groupby('user_id')['category_id'].apply(set).reset_index()
    
    # 只分析购买了多个品类的用户
    multi_category_users = user_category_sets[user_category_sets['category_id'].apply(len) > 1]
    
    # 计算品类对的共现次数
    if not multi_category_users.empty:
        category_pairs = []
        for _, row in multi_category_users.iterrows():
            categories = list(row['category_id'])
            for i in range(len(categories)):
                for j in range(i+1, len(categories)):
                    category_pairs.append(tuple(sorted([categories[i], categories[j]])))
        
        # 计算共现频率
        pair_counts = Counter(category_pairs)
        
        # 转换为DataFrame
        if pair_counts:
            cooccurrence_df = pd.DataFrame([
                {'category1': pair[0], 'category2': pair[1], 'count': count}
                for pair, count in pair_counts.most_common(20)
            ])
            
            cooccurrence_df.to_csv('result/tables/category_cooccurrence.csv', index=False)
    
    print("购买模式分析完成并保存到结果目录。")


def analyze_user_journey(df):
    """分析用户旅程和行为序列模式"""
    if df is None:
        return
    
    print("\n===== 用户旅程分析 =====")
    
    # 抽样用户进行分析
    sample_size = min(1000, df['user_id'].nunique())
    sample_users = np.random.choice(df['user_id'].unique(), sample_size, replace=False)
    
    # 筛选样本用户数据
    sample_df = df[df['user_id'].isin(sample_users)]
    
    # 按用户和时间戳排序
    sample_df = sample_df.sort_values(['user_id', 'timestamp'])
    
    # 跟踪行为序列
    print("分析用户行为序列...")
    
    # 为每个用户创建行为序列
    user_sequences = {}
    for user_id, group in sample_df.groupby('user_id'):
        behaviors = group['behavior_type'].tolist()
        user_sequences[user_id] = behaviors
    
    # 计算常见的2步序列
    two_step_sequences = []
    for user_id, behaviors in user_sequences.items():
        for i in range(len(behaviors) - 1):
            two_step_sequences.append((behaviors[i], behaviors[i+1]))
    
    # 计算序列频率
    sequence_counts = Counter(two_step_sequences)
    
    # 转换为DataFrame
    sequence_df = pd.DataFrame([
        {'first_behavior': first, 'next_behavior': second, 'count': count}
        for (first, second), count in sequence_counts.most_common()
    ])
    
    # 计算转换概率
    behavior_totals = {}
    for (first, _), count in sequence_counts.items():
        if first not in behavior_totals:
            behavior_totals[first] = 0
        behavior_totals[first] += count
    
    # 添加概率到dataframe
    sequence_df['probability'] = sequence_df.apply(
        lambda row: row['count'] / behavior_totals[row['first_behavior']], axis=1
    )
    
    # 保存序列分析
    sequence_df.to_csv('result/tables/behavior_sequences.csv', index=False)
    
    # 可视化前20个转换
    top_sequences = sequence_df.head(20)
    
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 8))
    sns.barplot(x='first_behavior', y='probability', hue='next_behavior', data=top_sequences)
    plt.title('行为转换概率')
    plt.xlabel('初始行为')
    plt.ylabel('转换概率')
    plt.legend(title='下一个行为')
    plt.tight_layout()
    plt.savefig('result/figures/behavior_transitions.png')
    plt.close()
    
    # 分析行为之间的时间
    print("分析行为之间的时间间隔...")
    
    # 计算连续行为之间的时间差
    time_diffs = []
    for user_id, group in sample_df.groupby('user_id'):
        timestamps = group['timestamp'].tolist()
        behaviors = group['behavior_type'].tolist()
        
        for i in range(len(timestamps) - 1):
            time_diff = (timestamps[i+1] - timestamps[i]).total_seconds() / 60  # 以分钟为单位
            time_diffs.append({
                'user_id': user_id,
                'first_behavior': behaviors[i],
                'next_behavior': behaviors[i+1],
                'time_diff_minutes': time_diff
            })
    
    time_diff_df = pd.DataFrame(time_diffs)
    
    # 计算行为之间的平均时间
    avg_time_diff = time_diff_df.groupby(['first_behavior', 'next_behavior'])['time_diff_minutes'].mean().reset_index()
    avg_time_diff.columns = ['初始行为', '下一个行为', '平均时间(分钟)']
    avg_time_diff = avg_time_diff.sort_values('平均时间(分钟)')
    
    # 保存时间差分析
    avg_time_diff.to_csv('result/tables/avg_time_between_behaviors.csv', index=False)
    
    # 可视化行为之间的时间
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 8))
    sns.barplot(x='初始行为', y='平均时间(分钟)', hue='下一个行为', data=avg_time_diff.head(20))
    plt.title('用户行为之间的平均时间')
    plt.xlabel('初始行为')
    plt.ylabel('平均时间(分钟)')
    plt.yscale('log')  # 对数尺度以便更好地可视化
    plt.tight_layout()
    plt.savefig('result/figures/time_between_behaviors.png')
    plt.close()
    
    print("用户旅程分析完成并保存。")


def category_analysis(df):
    """分析不同商品类目的用户行为模式"""
    if df is None or 'category_id' not in df.columns:
        return
    
    print("\n===== 商品类目分析 =====")
    
    # 按类目的行为分布
    category_behavior = df.groupby(['category_id', 'behavior_type']).size().reset_index(name='count')
    
    # 获取按总交互量排序的前20个类目
    top_categories = df.groupby('category_id').size().sort_values(ascending=False).head(20).index
    
    # 筛选前20个类目
    top_category_behavior = category_behavior[category_behavior['category_id'].isin(top_categories)]
    
    # 透视表格以便分析
    category_pivot = top_category_behavior.pivot(index='category_id', columns='behavior_type', values='count').fillna(0)
    
    # 计算转化率
    if 'pv' in category_pivot.columns and 'buy' in category_pivot.columns:
        category_pivot['pv_to_buy_rate'] = category_pivot['buy'] / category_pivot['pv'] * 100
    
    if 'cart' in category_pivot.columns and 'buy' in category_pivot.columns:
        category_pivot['cart_to_buy_rate'] = category_pivot['buy'] / category_pivot['cart'] * 100
    
    # 保存类目分析
    category_pivot.to_csv('result/tables/category_behavior_analysis.csv')
    
    # 可视化类目转化率
    if 'pv_to_buy_rate' in category_pivot.columns:
        category_conversion = category_pivot.reset_index()[['category_id', 'pv_to_buy_rate']].sort_values('pv_to_buy_rate', ascending=False)
        
        plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
        plt.figure(figsize=(14, 8))
        sns.barplot(x='category_id', y='pv_to_buy_rate', data=category_conversion)
        plt.title('按类目的PV到购买转化率')
        plt.xlabel('类目ID')
        plt.ylabel('转化率 (%)')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('result/figures/category_conversion_rates.png')
        plt.close()
    
    # 分析类目内部的行为分布
    behavior_props = top_category_behavior.groupby('category_id')['count'].sum().reset_index()
    behavior_props = behavior_props.rename(columns={'count': 'total'})
    
    top_category_behavior = top_category_behavior.merge(behavior_props, on='category_id')
    top_category_behavior['percentage'] = top_category_behavior['count'] / top_category_behavior['total'] * 100
    
    # 保存按类目的行为分布
    top_category_behavior.to_csv('result/tables/behavior_distribution_by_category.csv', index=False)
    
    # 可视化前5个类目的行为分布
    top5_categories = df.groupby('category_id').size().sort_values(ascending=False).head(5).index
    
    top5_category_behavior = top_category_behavior[top_category_behavior['category_id'].isin(top5_categories)]
    
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(14, 8))
    sns.barplot(x='category_id', y='percentage', hue='behavior_type', data=top5_category_behavior)
    plt.title('前5个类目的行为分布')
    plt.xlabel('类目ID')
    plt.ylabel('行为百分比 (%)')
    plt.tight_layout()
    plt.savefig('result/figures/top5_category_behavior_distribution.png')
    plt.close()
    
    print("商品类目分析完成并保存。")


def user_clustering(df):
    """使用聚类方法对用户进行分群"""
    if df is None:
        return
    
    print("\n===== 用户聚类分析 =====")
    
    # 为每个用户计算行为特征
    print("计算用户行为特征...")
    
    # 用户行为计数
    user_behavior_counts = df.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0).reset_index()
    
    # 如果缺少某些行为类型，添加零列
    for behavior in ['pv', 'buy', 'cart', 'fav']:
        if behavior not in user_behavior_counts.columns:
            user_behavior_counts[behavior] = 0
    
    # 计算用户活跃天数
    user_active_days = df.groupby('user_id')['date'].nunique().reset_index()
    user_active_days.columns = ['user_id', 'active_days']
    
    # 计算用户浏览的商品类目数
    user_categories = df.groupby('user_id')['category_id'].nunique().reset_index()
    user_categories.columns = ['user_id', 'unique_categories']
    
    # 合并所有特征
    user_features = user_behavior_counts.merge(user_active_days, on='user_id')
    user_features = user_features.merge(user_categories, on='user_id')
    
    # 标准化聚类特征
    features = ['pv', 'buy', 'cart', 'fav', 'active_days', 'unique_categories']
    features_available = [f for f in features if f in user_features.columns]
    
    X = user_features[features_available].values
    X = StandardScaler().fit_transform(X)
    
    # 确定最佳聚类数 (使用肘部法则)
    inertia = []
    k_range = range(2, 11)
    
    # 使用小样本进行K值选择（如果数据量大）
    if X.shape[0] > 10000:
        X_sample = X[np.random.choice(X.shape[0], 10000, replace=False)]
    else:
        X_sample = X
    
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        model.fit(X_sample)
        inertia.append(model.inertia_)
    
    # 可视化肘部法则
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-')
    plt.title('K值选择的肘部法则')
    plt.xlabel('簇数量')
    plt.ylabel('惯性值')
    plt.grid(True)
    plt.savefig('result/figures/kmeans_elbow.png')
    plt.close()
    
    # 选择合适的K值
    # 简单起见，这里选择K=4
    optimal_k = 4
    
    # 执行KMeans聚类
    print(f"执行K-means聚类 (k={optimal_k})...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X)
    
    # 添加聚类标签到用户特征
    user_features['cluster'] = clusters
    
    # 保存用户聚类结果
    user_features.to_csv('result/tables/user_clustering.csv', index=False)
    
    # 可视化聚类结果
    cluster_profile = user_features.groupby('cluster')[features_available].mean()
    
    # 为了雷达图，将特征标准化到0-1范围
    cluster_profile_normalized = cluster_profile.copy()
    for feature in features_available:
        min_val = cluster_profile[feature].min()
        max_val = cluster_profile[feature].max()
        if max_val > min_val:
            cluster_profile_normalized[feature] = (cluster_profile[feature] - min_val) / (max_val - min_val)
    
    # 可视化每个聚类的特征
    plt.figure(figsize=(15, 10))
    for i in range(optimal_k):
        plt.subplot(2, 2, i+1)
        values = cluster_profile_normalized.iloc[i].values
        angles = np.linspace(0, 2*np.pi, len(features_available), endpoint=False).tolist()
        # 闭合雷达图
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
        plt.polar(angles, values)
        plt.fill(angles, values, alpha=0.25)
        plt.title(f'聚类 {i} 特征')
        plt.xticks(angles[:-1], features_available)
    
    plt.tight_layout()
    plt.savefig('result/figures/cluster_radar_charts.png')
    plt.close()
    
    # 可视化聚类大小
    cluster_sizes = user_features['cluster'].value_counts().sort_index()
    
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(10, 6))
    cluster_sizes.plot(kind='bar')
    plt.title('各聚类的用户数量')
    plt.xlabel('聚类')
    plt.ylabel('用户数')
    plt.tight_layout()
    plt.savefig('result/figures/cluster_sizes.png')
    plt.close()
    
    # 分析每个聚类的购买转化率
    if 'pv' in user_features.columns and 'buy' in user_features.columns:
        user_features['buy_rate'] = user_features['buy'] / user_features['pv']
        cluster_conversion = user_features.groupby('cluster')['buy_rate'].mean().reset_index()
        
        plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
        plt.figure(figsize=(10, 6))
        sns.barplot(x='cluster', y='buy_rate', data=cluster_conversion)
        plt.title('各聚类的平均购买转化率')
        plt.xlabel('聚类')
        plt.ylabel('平均购买转化率')
        plt.tight_layout()
        plt.savefig('result/figures/cluster_conversion_rates.png')
        plt.close()
    
    # 用文字描述每个聚类的特征
    cluster_description = []
    
    for i in range(optimal_k):
        profile = cluster_profile.iloc[i]
        size = cluster_sizes[i]
        percent = (size / cluster_sizes.sum()) * 100
        
        # 确定该聚类的显著特征
        features_rank = profile.sort_values(ascending=False)
        top_features = features_rank.index[:3].tolist()
        
        description = f"聚类 {i}: {size} 用户 ({percent:.2f}%)\n"
        description += "显著特征:\n"
        
        for feature in top_features:
            description += f"- {feature}: {profile[feature]:.2f}\n"
        
        cluster_description.append(description)
    
    # 保存聚类描述
    with open('result/tables/cluster_descriptions.txt', 'w') as f:
        for desc in cluster_description:
            f.write(desc + "\n\n")
    
    print("用户聚类分析完成并保存。")


def build_prediction_model(df):
    """构建购买行为预测模型"""
    if df is None:
        return
    
    print("\n===== 购买行为预测模型 =====")
    
    # 准备特征和目标变量
    print("准备模型数据...")
    
    # 为每个用户-商品对准备特征
    # 示例: 我们将使用用户对特定商品的点击、收藏、加购次数以及总体活跃度作为特征
    
    # 为了简化计算，我们将抽样数据
    if df.shape[0] > 100000:
        df_sample = df.sample(n=100000, random_state=42)
    else:
        df_sample = df
    
    # 特征工程
    print("执行特征工程...")
    user_item_data = []
    
    # 筛选有"购买"行为的用户-商品对
    purchase_pairs = df_sample[df_sample['behavior_type'] == 'buy'][['user_id', 'item_id']].drop_duplicates()
    purchase_pairs['purchased'] = 1
    
    # 随机选择一些没有购买行为的用户-商品对作为负样本
    all_pairs = df_sample[['user_id', 'item_id']].drop_duplicates()
    all_pairs = all_pairs.merge(purchase_pairs, on=['user_id', 'item_id'], how='left')
    all_pairs['purchased'] = all_pairs['purchased'].fillna(0)
    
    # 均衡正负样本
    positive_samples = all_pairs[all_pairs['purchased'] == 1]
    negative_samples = all_pairs[all_pairs['purchased'] == 0].sample(n=len(positive_samples), random_state=42)
    balanced_pairs = pd.concat([positive_samples, negative_samples])
    
    # 计算每个用户-商品对的行为计数
    behavior_counts = df_sample.groupby(['user_id', 'item_id', 'behavior_type']).size().unstack(fill_value=0).reset_index()
    
    # 确保所有行为类型都有列
    for behavior in ['pv', 'cart', 'fav']:
        if behavior not in behavior_counts.columns:
            behavior_counts[behavior] = 0
    
    # 合并特征和目标变量
    model_data = balanced_pairs.merge(behavior_counts, on=['user_id', 'item_id'], how='left')
    
    # 填充缺失值
    for behavior in ['pv', 'cart', 'fav']:
        if behavior in model_data.columns:
            model_data[behavior] = model_data[behavior].fillna(0)
    
    # 添加用户行为总量作为特征
    user_behavior_totals = df_sample.groupby('user_id').size().reset_index(name='user_activity')
    model_data = model_data.merge(user_behavior_totals, on='user_id', how='left')
    model_data['user_activity'] = model_data['user_activity'].fillna(0)
    
    # 检查准备好的数据
    print(f"已准备好的模型数据形状: {model_data.shape}")
    
    # 选择特征和目标变量
    feature_cols = ['pv', 'cart', 'fav', 'user_activity']
    feature_cols = [col for col in feature_cols if col in model_data.columns]
    X = model_data[feature_cols]
    y = model_data['purchased']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练随机森林模型
    print("训练购买行为预测模型...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    
    # 生成分类报告
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('result/tables/purchase_prediction_report.csv')
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    feature_importance.to_csv('result/tables/feature_importance.csv', index=False)
    
    # 可视化特征重要性
    plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('特征重要性')
    plt.tight_layout()
    plt.savefig('result/figures/feature_importance.png')
    plt.close()
    
    print("购买预测模型分析完成并保存。")


def generate_combined_visualizations(df, tmall_df):
    """生成一些组合可视化，整合分析结果"""
    print("\n===== 生成综合可视化 =====")
    
    # 1. 创建一个综合的转化漏斗
    if df is not None:
        behavior_counts = df['behavior_type'].value_counts().reset_index()
        behavior_counts.columns = ['行为', '频次']
        behavior_order = ['pv', 'fav', 'cart', 'buy']
        
        # 对行为进行排序
        behavior_counts['排序'] = behavior_counts['行为'].map({b: i for i, b in enumerate(behavior_order)})
        behavior_counts = behavior_counts.sort_values('排序')
        
        # 计算转化率
        max_count = behavior_counts['频次'].max()
        behavior_counts['百分比'] = behavior_counts['频次'] / max_count * 100
        
        # 可视化转化漏斗
        plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
        plt.figure(figsize=(12, 8))
        sns.barplot(x='行为', y='百分比', data=behavior_counts, order=behavior_order)
        plt.title('用户行为转化漏斗')
        plt.ylabel('相对百分比 (%)')
        plt.xlabel('行为类型')
        
        # 添加数值标签
        for i, row in behavior_counts.iterrows():
            plt.text(row['排序'], row['百分比'] / 2, 
                    f"{row['频次']:,}\n({row['百分比']:.1f}%)", 
                    ha='center', va='center', color='white', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('result/figures/conversion_funnel.png')
        plt.close()
    
    # 2. 整合用户活跃度时间分布
    if df is not None:
        # 按小时和星期几统计行为
        hourly_counts = df.groupby('hour').size()
        dow_counts = df.groupby('day_of_week').size()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 按小时分布
        hourly_counts.plot(kind='bar', ax=ax1)
        ax1.set_title('用户行为的小时分布')
        ax1.set_xlabel('小时')
        ax1.set_ylabel('行为数量')
        ax1.set_xticks(range(24))
        
        # 按星期几分布
        dow_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        dow_counts = dow_counts.reset_index()
        dow_counts['day_name'] = dow_counts['day_of_week'].map({i: name for i, name in enumerate(dow_names)})
        
        sns.barplot(x='day_name', y=0, data=dow_counts, order=dow_names, ax=ax2)
        ax2.set_title('用户行为的星期分布')
        ax2.set_xlabel('星期')
        ax2.set_ylabel('行为数量')
        
        plt.tight_layout()
        plt.savefig('result/figures/user_activity_time_distribution.png')
        plt.close()
    
    # 3. 综合每日行为分布热图
    if df is not None:
        # 按日期和小时统计行为
        df['date_str'] = df['date'].astype(str)
        hourly_activity = df.groupby(['date_str', 'hour']).size().reset_index(name='count')
        
        # 创建透视表
        pivot_data = hourly_activity.pivot(index='hour', columns='date_str', values='count').fillna(0)
        
        # 绘制热图
        plt.rcParams['font.family'] = ['Hei', 'Arial', 'Helvetica', 'Times New Roman']
        plt.figure(figsize=(15, 8))
        sns.heatmap(pivot_data, cmap='YlGnBu')
        plt.title('每日每小时用户活跃度热图')
        plt.xlabel('日期')
        plt.ylabel('小时')
        plt.tight_layout()
        plt.savefig('result/figures/hourly_daily_activity_heatmap.png')
        plt.close()
    
    print("综合可视化完成并保存。")


def generate_executive_summary(df, tmall_df):
    """生成分析结果的执行摘要，总结关键发现"""
    print("\n===== 生成执行摘要 =====")
    
    summary = "# 淘宝用户行为分析 - 执行摘要\n\n"
    summary += "## 1. 数据概览\n\n"
    
    if df is not None:
        summary += f"- UserBehavior数据集: {df.shape[0]:,} 行, {df['user_id'].nunique():,} 用户\n"
        summary += f"- 时间范围: {df['date'].min()} 至 {df['date'].max()}\n"
        summary += f"- 商品数量: {df['item_id'].nunique():,}\n"
        summary += f"- 类目数量: {df['category_id'].nunique():,}\n\n"
    
    if tmall_df is not None:
        summary += f"- 天猫数据集: {tmall_df.shape[0]:,} 行, {tmall_df['user_id'].nunique():,} 用户\n"
        summary += f"- 时间范围: {tmall_df['date'].min()} 至 {tmall_df['date'].max()}\n"
        summary += f"- 商品数量: {tmall_df['item_id'].nunique():,}\n\n"
    
    summary += "## 2. 关键发现\n\n"
    
    if df is not None:
        # 行为分布
        behavior_counts = df['behavior_type'].value_counts()
        summary += "### 行为分布\n\n"
        for behavior, count in behavior_counts.items():
            summary += f"- {behavior}: {count:,} ({count/len(df)*100:.2f}%)\n"
        
        # 转化率
        if 'pv' in behavior_counts and 'buy' in behavior_counts:
            pv_to_buy = behavior_counts['buy'] / behavior_counts['pv'] * 100
            summary += f"\n浏览到购买转化率: {pv_to_buy:.2f}%\n"
        
        if 'cart' in behavior_counts and 'buy' in behavior_counts:
            cart_to_buy = behavior_counts['buy'] / behavior_counts['cart'] * 100
            summary += f"加购到购买转化率: {cart_to_buy:.2f}%\n\n"
        
        # 高峰期
        hourly_counts = df.groupby('hour').size()
        peak_hour = hourly_counts.idxmax()
        summary += f"活跃高峰时段: {peak_hour}:00\n"
        
        dow_counts = df.groupby('day_of_week').size()
        peak_day = dow_counts.idxmax()
        day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
        summary += f"最活跃的星期: {day_names[peak_day]}\n\n"
    
    summary += "## 3. 用户行为洞察\n\n"
    
    # 从之前保存的分析结果中提取洞察
    try:
        # 尝试读取用户分群文件
        segment_df = pd.read_csv('result/tables/user_segmentation.csv')
        segment_counts = segment_df['Segment'].value_counts()
        
        summary += "### 用户分群\n\n"
        for segment, count in segment_counts.items():
            summary += f"- {segment}: {count:,} 用户 ({count/len(segment_df)*100:.2f}%)\n"
        summary += "\n"
    except:
        pass
    
    try:
        # 尝试读取购买间隔统计
        interval_stats = pd.read_csv('result/tables/purchase_interval_statistics.csv')
        avg_interval = interval_stats[interval_stats['统计量'] == 'mean']['天数'].values[0]
        
        summary += f"平均购买间隔: {avg_interval:.2f} 天\n\n"
    except:
        pass
    
    summary += "## 4. 商品与类目洞察\n\n"
    
    if df is not None:
        # 热门类目
        top_categories = df.groupby('category_id').size().sort_values(ascending=False).head(5)
        
        summary += "### 热门商品类目 (前5)\n\n"
        for category, count in top_categories.items():
            summary += f"- 类目 {category}: {count:,} 交互\n"
        summary += "\n"
    
    summary += "## 5. 优化建议\n\n"
    
    summary += "基于分析结果，我们建议以下优化措施：\n\n"
    
    summary += "1. **精准营销策略**：针对识别出的不同用户群体，制定差异化的营销策略。\n"
    summary += "2. **购物流程优化**：优化从浏览到购买的转化流程，特别是加购到购买阶段的转化率。\n"
    summary += "3. **时间性营销**：在用户活跃高峰期加强推广力度。\n"
    summary += "4. **类目推广策略**：针对高转化率类目加强推广，对低转化率类目进行优化。\n"
    summary += "5. **个性化推荐**：基于用户行为模式，提供更精准的个性化商品推荐。\n"
    
    # 保存摘要
    with open('result/executive_summary.md', 'w') as f:
        f.write(summary)
    
    print("执行摘要已生成并保存到 result/executive_summary.md")


def main():
    """主函数，协调整个分析流程"""
    print("开始淘宝用户行为分析...\n")
    
    # 加载数据
    userbehavior_df, tmall_df = load_and_understand_data()
    
    # 数据清洗和预处理
    if userbehavior_df is not None:
        cleaned_userbehavior = explore_and_clean_userbehavior(userbehavior_df)
    else:
        cleaned_userbehavior = None
        print("警告: 未能加载UserBehavior数据集，跳过相关分析。")
    
    if tmall_df is not None:
        cleaned_tmall = explore_and_clean_tmall(tmall_df)
    else:
        cleaned_tmall = None
        print("警告: 未能加载天猫数据集，跳过相关分析。")
    
    # 用户行为分析
    if cleaned_userbehavior is not None:
        analyze_user_behavior(cleaned_userbehavior)
        user_segmentation(cleaned_userbehavior)
        analyze_purchase_patterns(cleaned_userbehavior)
        analyze_user_journey(cleaned_userbehavior)
        category_analysis(cleaned_userbehavior)
        user_clustering(cleaned_userbehavior)
        build_prediction_model(cleaned_userbehavior)
    
    # 天猫数据分析
    if cleaned_tmall is not None:
        analyze_tmall_data(cleaned_tmall)
    
    # 生成综合可视化
    generate_combined_visualizations(cleaned_userbehavior, cleaned_tmall)
    
    # 生成执行摘要
    generate_executive_summary(cleaned_userbehavior, cleaned_tmall)
    
    print("\n淘宝用户行为分析完成！所有结果已保存到 result/ 目录。")


if __name__ == "__main__":
    # 设置开始时间
    start_time = time.time()
    
    # 执行主程序
    main()
    
    # 计算并显示运行时间
    execution_time = time.time() - start_time
    print(f"\n程序执行时间: {execution_time:.2f} 秒")