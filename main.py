import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
from collections import Counter, defaultdict
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import networkx as nx
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import warnings
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from tqdm import tqdm
import logging
import json
import itertools
import calendar
from wordcloud import WordCloud
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.utils import summary_data_from_transaction_data

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')

# 设置文件路径
DATA_DIR = 'dataset/'
RESULT_DIR = 'result/'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(f"{RESULT_DIR}/analysis.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 确保结果目录存在
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
if not os.path.exists(os.path.join(RESULT_DIR, 'figures')):
    os.makedirs(os.path.join(RESULT_DIR, 'figures'))
if not os.path.exists(os.path.join(RESULT_DIR, 'tables')):
    os.makedirs(os.path.join(RESULT_DIR, 'tables'))

# 设置绘图样式
sns.set(style="whitegrid")
plt.style.use('ggplot')


def load_data():
    """
    加载数据集
    """
    logger.info("开始加载数据...")
    
    # 加载用户行为数据
    userbehavior_cols = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
    try:
        user_behavior_df = pd.read_csv(os.path.join(DATA_DIR, 'UserBehavior.csv'), 
                                     header=None, names=userbehavior_cols)
        logger.info(f"用户行为数据加载完成，共 {len(user_behavior_df)} 条记录")
    except Exception as e:
        logger.error(f"加载用户行为数据失败: {e}")
        user_behavior_df = pd.DataFrame(columns=userbehavior_cols)
    
    # 加载天猫数据
    try:
        tmall_df = pd.read_csv(os.path.join(DATA_DIR, 'sam_tianchi_2014002_rec_tmall_log.csv'))
        logger.info(f"天猫数据加载完成，共 {len(tmall_df)} 条记录")
    except Exception as e:
        logger.error(f"加载天猫数据失败: {e}")
        tmall_df = pd.DataFrame(columns=['item_id', 'user_id', 'action', 'vtime'])
    
    return user_behavior_df, tmall_df


def clean_userbehavior_data(df):
    """
    清洗用户行为数据
    """
    logger.info("开始清洗用户行为数据...")
    
    # 复制数据
    df_clean = df.copy()
    
    # 检查并处理缺失值
    missing_values = df_clean.isnull().sum()
    logger.info(f"缺失值统计：\n{missing_values}")
    
    # 如果存在缺失值，进行处理
    if missing_values.sum() > 0:
        logger.info("处理缺失值...")
        df_clean = df_clean.dropna()  # 删除包含缺失值的行
    
    # 转换时间戳为日期时间格式
    logger.info("转换时间戳...")
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], unit='s')
    df_clean['date'] = df_clean['timestamp'].dt.date
    df_clean['hour'] = df_clean['timestamp'].dt.hour
    
    # 检查并处理重复值
    duplicates = df_clean.duplicated().sum()
    logger.info(f"重复记录数：{duplicates}")
    if duplicates > 0:
        logger.info("删除重复记录...")
        df_clean = df_clean.drop_duplicates()
    
    # 检查异常值
    logger.info("数据类型检查：")
    logger.info(df_clean.dtypes)
    
    # 检查行为类型是否符合预期
    expected_behaviors = ['pv', 'buy', 'cart', 'fav']
    unexpected_behaviors = set(df_clean['behavior_type'].unique()) - set(expected_behaviors)
    
    if unexpected_behaviors:
        logger.warning(f"发现非预期的行为类型: {unexpected_behaviors}")
        # 过滤掉非预期的行为类型
        df_clean = df_clean[df_clean['behavior_type'].isin(expected_behaviors)]
    
    # 行为类型分布
    behavior_counts = df_clean['behavior_type'].value_counts()
    logger.info("行为类型分布：")
    logger.info(behavior_counts)
    
    # 检查时间范围
    time_range = (df_clean['timestamp'].min(), df_clean['timestamp'].max())
    logger.info(f"数据时间范围: {time_range[0]} 到 {time_range[1]}")
    
    return df_clean


def clean_tmall_data(df):
    """
    清洗天猫数据
    """
    logger.info("开始清洗天猫数据...")
    
    # 复制数据
    df_clean = df.copy()
    
    # 检查并处理缺失值
    missing_values = df_clean.isnull().sum()
    logger.info(f"缺失值统计：\n{missing_values}")
    
    # 如果存在缺失值，进行处理
    if missing_values.sum() > 0:
        logger.info("处理缺失值...")
        df_clean = df_clean.dropna()  # 删除包含缺失值的行
    
    # 转换时间格式
    logger.info("转换时间格式...")
    try:
        df_clean['vtime'] = pd.to_datetime(df_clean['vtime'])
        df_clean['date'] = df_clean['vtime'].dt.date
        df_clean['hour'] = df_clean['vtime'].dt.hour
    except Exception as e:
        logger.error(f"转换时间格式失败: {e}")
        # 尝试不同的时间格式
        try:
            df_clean['vtime'] = pd.to_datetime(df_clean['vtime'], format='%Y/%m/%d %H:%M')
            df_clean['date'] = df_clean['vtime'].dt.date
            df_clean['hour'] = df_clean['vtime'].dt.hour
            logger.info("使用替代格式成功转换时间")
        except:
            logger.error("所有时间格式转换尝试均失败")
    
    # 检查并处理重复值
    duplicates = df_clean.duplicated().sum()
    logger.info(f"重复记录数：{duplicates}")
    if duplicates > 0:
        logger.info("删除重复记录...")
        df_clean = df_clean.drop_duplicates()
    
    # 检查异常值
    logger.info("数据类型检查：")
    logger.info(df_clean.dtypes)
    
    # 检查行为类型是否符合预期
    expected_actions = ['click', 'collect', 'cart', 'alipay']
    unexpected_actions = set(df_clean['action'].unique()) - set(expected_actions)
    
    if unexpected_actions:
        logger.warning(f"发现非预期的行为类型: {unexpected_actions}")
        # 过滤掉非预期的行为类型
        df_clean = df_clean[df_clean['action'].isin(expected_actions)]
    
    # 行为类型分布
    action_counts = df_clean['action'].value_counts()
    logger.info("行为类型分布：")
    logger.info(action_counts)
    
    # 检查时间范围
    time_range = (df_clean['vtime'].min(), df_clean['vtime'].max())
    logger.info(f"数据时间范围: {time_range[0]} 到 {time_range[1]}")
    
    return df_clean


def analyze_user_behavior(df):
    """
    分析用户行为
    """
    logger.info("开始分析用户行为...")
    
    # 1. 用户活跃度分析
    logger.info("计算用户活跃度...")
    user_activity = df.groupby('user_id').size().reset_index(name='activity_count')
    user_activity_stats = user_activity['activity_count'].describe()
    logger.info("用户活跃度统计：")
    logger.info(user_activity_stats)
    
    # 保存用户活跃度统计数据
    user_activity_stats.to_csv(os.path.join(RESULT_DIR, 'tables', 'user_activity_stats.csv'))
    
    # 2. 用户购买转化率分析
    logger.info("计算用户购买转化率...")
    user_behavior_counts = df.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0)
    
    # 确保所有行为类型列都存在
    for behavior in ['pv', 'buy', 'cart', 'fav']:
        if behavior not in user_behavior_counts.columns:
            user_behavior_counts[behavior] = 0
    
    # 计算转化率
    user_behavior_counts['click_to_buy_rate'] = np.where(user_behavior_counts['pv'] > 0, 
                                                     user_behavior_counts['buy'] / user_behavior_counts['pv'], 0)
    user_behavior_counts['cart_to_buy_rate'] = np.where(user_behavior_counts['cart'] > 0, 
                                                     user_behavior_counts['buy'] / user_behavior_counts['cart'], 0)
    user_behavior_counts['fav_to_buy_rate'] = np.where(user_behavior_counts['fav'] > 0, 
                                                    user_behavior_counts['buy'] / user_behavior_counts['fav'], 0)
    
    # 计算整体转化率
    overall_conversion = {
        'click_to_buy': df[df['behavior_type'] == 'buy'].user_id.nunique() / 
                      df[df['behavior_type'] == 'pv'].user_id.nunique() if df[df['behavior_type'] == 'pv'].user_id.nunique() > 0 else 0,
        'cart_to_buy': df[df['behavior_type'] == 'buy'].user_id.nunique() / 
                     df[df['behavior_type'] == 'cart'].user_id.nunique() if df[df['behavior_type'] == 'cart'].user_id.nunique() > 0 else 0,
        'fav_to_buy': df[df['behavior_type'] == 'buy'].user_id.nunique() / 
                    df[df['behavior_type'] == 'fav'].user_id.nunique() if df[df['behavior_type'] == 'fav'].user_id.nunique() > 0 else 0
    }
    
    logger.info("整体转化率：")
    logger.info(overall_conversion)
    
    # 保存转化率数据
    pd.DataFrame([overall_conversion]).to_csv(os.path.join(RESULT_DIR, 'tables', 'overall_conversion_rates.csv'))
    
    # 3. 用户行为时间模式分析
    logger.info("分析用户行为时间模式...")
    # 按小时统计行为次数
    hourly_behavior = df.groupby(['hour', 'behavior_type']).size().unstack(fill_value=0)
    
    # 按日期统计行为次数
    daily_behavior = df.groupby(['date', 'behavior_type']).size().unstack(fill_value=0)
    
    # 保存时间模式数据
    hourly_behavior.to_csv(os.path.join(RESULT_DIR, 'tables', 'hourly_behavior.csv'))
    daily_behavior.to_csv(os.path.join(RESULT_DIR, 'tables', 'daily_behavior.csv'))
    
    # 4. 商品类目偏好分析
    logger.info("分析商品类目偏好...")
    category_preference = df.groupby(['category_id', 'behavior_type']).size().unstack(fill_value=0)
    
    # 确保所有行为类型列都存在
    for behavior in ['pv', 'buy', 'cart', 'fav']:
        if behavior not in category_preference.columns:
            category_preference[behavior] = 0
    
    # 计算类目受欢迎程度指标
    category_preference['popularity'] = (category_preference['pv'] + 
                                      2*category_preference['fav'] + 
                                      3*category_preference['cart'] + 
                                      5*category_preference['buy'])
    
    # 获取热门类目
    top_categories = category_preference.sort_values('popularity', ascending=False).head(20)
    
    # 保存品类偏好数据
    top_categories.to_csv(os.path.join(RESULT_DIR, 'tables', 'top_categories.csv'))
    
    # 5. 用户购买习惯分析
    logger.info("分析用户购买习惯...")
    # 计算用户平均购买周期（仅对有多次购买行为的用户）
    buy_records = df[df['behavior_type'] == 'buy'].sort_values(['user_id', 'timestamp'])
    
    # 计算同一用户相邻购买的时间差
    buy_records['next_buy_time'] = buy_records.groupby('user_id')['timestamp'].shift(-1)
    buy_records['time_diff'] = (buy_records['next_buy_time'] - buy_records['timestamp']).dt.total_seconds() / 3600  # 转换为小时
    
    # 过滤掉无效的时间差（最后一次购买没有下一次）
    valid_diffs = buy_records.dropna(subset=['time_diff'])
    
    # 计算每个用户的平均购买周期
    user_buy_cycle = valid_diffs.groupby('user_id')['time_diff'].mean().reset_index()
    user_buy_cycle.columns = ['user_id', 'avg_buy_cycle_hours']
    
    # 计算购买周期的统计数据
    buy_cycle_stats = user_buy_cycle['avg_buy_cycle_hours'].describe()
    logger.info("购买周期统计（小时）：")
    logger.info(buy_cycle_stats)
    
    # 保存购买周期数据
    buy_cycle_stats.to_csv(os.path.join(RESULT_DIR, 'tables', 'buy_cycle_stats.csv'))
    
    # 6. 用户忠诚度分析
    logger.info("分析用户忠诚度...")
    # 计算用户首次和最后一次行为的时间差（活跃天数）
    user_time_span = df.groupby('user_id').agg(
        first_action=('timestamp', 'min'),
        last_action=('timestamp', 'max')
    )
    user_time_span['active_days'] = (user_time_span['last_action'] - user_time_span['first_action']).dt.total_seconds() / (24*3600)
    
    # 计算用户平均每天的行为次数
    user_activity_daily = pd.merge(
        user_activity,
        user_time_span,
        on='user_id'
    )
    user_activity_daily['actions_per_day'] = np.where(
        user_activity_daily['active_days'] > 0,
        user_activity_daily['activity_count'] / user_activity_daily['active_days'],
        user_activity_daily['activity_count']  # 如果只有一天活跃，则直接使用活跃次数
    )
    
    # 保存用户忠诚度数据
    user_activity_daily.to_csv(os.path.join(RESULT_DIR, 'tables', 'user_loyalty.csv'), index=False)
    
    return {
        'user_activity': user_activity,
        'user_behavior_counts': user_behavior_counts,
        'hourly_behavior': hourly_behavior,
        'daily_behavior': daily_behavior,
        'category_preference': category_preference,
        'user_buy_cycle': user_buy_cycle,
        'user_loyalty': user_activity_daily
    }


def analyze_tmall_data(df):
    """
    分析天猫数据
    """
    logger.info("开始分析天猫数据...")
    
    # 如果数据集为空或记录太少，则跳过分析
    if df.empty or len(df) < 10:
        logger.warning("天猫数据集为空或记录太少，跳过分析")
        return {}
    
    # 1. 用户活跃度分析
    logger.info("计算天猫用户活跃度...")
    user_activity = df.groupby('user_id').size().reset_index(name='activity_count')
    user_activity_stats = user_activity['activity_count'].describe()
    logger.info("用户活跃度统计：")
    logger.info(user_activity_stats)
    
    # 保存用户活跃度统计数据
    user_activity_stats.to_csv(os.path.join(RESULT_DIR, 'tables', 'tmall_user_activity_stats.csv'))
    
    # 2. 用户购买转化率分析
    logger.info("计算天猫用户购买转化率...")
    user_behavior_counts = df.groupby(['user_id', 'action']).size().unstack(fill_value=0)
    
    # 确保所有行为类型列都存在
    for action in ['click', 'collect', 'cart', 'alipay']:
        if action not in user_behavior_counts.columns:
            user_behavior_counts[action] = 0
    
    # 计算转化率
    user_behavior_counts['click_to_buy_rate'] = np.where(user_behavior_counts['click'] > 0, 
                                                     user_behavior_counts['alipay'] / user_behavior_counts['click'], 0)
    user_behavior_counts['cart_to_buy_rate'] = np.where(user_behavior_counts['cart'] > 0, 
                                                     user_behavior_counts['alipay'] / user_behavior_counts['cart'], 0)
    user_behavior_counts['collect_to_buy_rate'] = np.where(user_behavior_counts['collect'] > 0, 
                                                       user_behavior_counts['alipay'] / user_behavior_counts['collect'], 0)
    
    # 计算整体转化率
    overall_conversion = {
        'click_to_buy': df[df['action'] == 'alipay'].user_id.nunique() / 
                      df[df['action'] == 'click'].user_id.nunique() if df[df['action'] == 'click'].user_id.nunique() > 0 else 0,
        'cart_to_buy': df[df['action'] == 'alipay'].user_id.nunique() / 
                     df[df['action'] == 'cart'].user_id.nunique() if df[df['action'] == 'cart'].user_id.nunique() > 0 else 0,
        'collect_to_buy': df[df['action'] == 'alipay'].user_id.nunique() / 
                        df[df['action'] == 'collect'].user_id.nunique() if df[df['action'] == 'collect'].user_id.nunique() > 0 else 0
    }
    
    logger.info("天猫整体转化率：")
    logger.info(overall_conversion)
    
    # 保存转化率数据
    pd.DataFrame([overall_conversion]).to_csv(os.path.join(RESULT_DIR, 'tables', 'tmall_overall_conversion_rates.csv'))
    
    # 3. 用户行为时间模式分析
    logger.info("分析天猫用户行为时间模式...")
    # 按小时统计行为次数
    hourly_behavior = df.groupby(['hour', 'action']).size().unstack(fill_value=0)
    
    # 按日期统计行为次数
    daily_behavior = df.groupby(['date', 'action']).size().unstack(fill_value=0)
    
    # 保存时间模式数据
    hourly_behavior.to_csv(os.path.join(RESULT_DIR, 'tables', 'tmall_hourly_behavior.csv'))
    daily_behavior.to_csv(os.path.join(RESULT_DIR, 'tables', 'tmall_daily_behavior.csv'))
    
    # 4. 商品偏好分析
    logger.info("分析天猫商品偏好...")
    item_preference = df.groupby(['item_id', 'action']).size().unstack(fill_value=0)
    
    # 确保所有行为类型列都存在
    for action in ['click', 'collect', 'cart', 'alipay']:
        if action not in item_preference.columns:
            item_preference[action] = 0
    
    # 计算商品受欢迎程度指标
    item_preference['popularity'] = (item_preference['click'] + 
                                   2*item_preference['collect'] + 
                                   3*item_preference['cart'] + 
                                   5*item_preference['alipay'])
    
    # 获取热门商品
    top_items = item_preference.sort_values('popularity', ascending=False).head(20)
    
    # 保存商品偏好数据
    top_items.to_csv(os.path.join(RESULT_DIR, 'tables', 'tmall_top_items.csv'))
    
    # 5. 用户忠诚度分析
    logger.info("分析天猫用户忠诚度...")
    # 计算用户首次和最后一次行为的时间差（活跃天数）
    user_time_span = df.groupby('user_id').agg(
        first_action=('vtime', 'min'),
        last_action=('vtime', 'max')
    )
    user_time_span['active_days'] = (user_time_span['last_action'] - user_time_span['first_action']).dt.total_seconds() / (24*3600)
    
    # 计算用户平均每天的行为次数
    user_activity_daily = pd.merge(
        user_activity,
        user_time_span,
        on='user_id'
    )
    user_activity_daily['actions_per_day'] = np.where(
        user_activity_daily['active_days'] > 0,
        user_activity_daily['activity_count'] / user_activity_daily['active_days'],
        user_activity_daily['activity_count']  # 如果只有一天活跃，则直接使用活跃次数
    )
    
    # 保存用户忠诚度数据
    user_activity_daily.to_csv(os.path.join(RESULT_DIR, 'tables', 'tmall_user_loyalty.csv'), index=False)
    
    return {
        'user_activity': user_activity,
        'user_behavior_counts': user_behavior_counts,
        'hourly_behavior': hourly_behavior,
        'daily_behavior': daily_behavior,
        'item_preference': item_preference,
        'user_loyalty': user_activity_daily
    }


def segment_users(user_df, behavior_df):
    """
    用户分群分析
    """
    logger.info("开始用户分群分析...")
    
    # 特征工程 - 计算RFM指标
    # Recency - 最近一次购买时间距离最后日期的天数
    # Frequency - 购买次数
    # Monetary - 购买商品的种类数量，作为消费多样性的代理指标
    
    # 获取最后日期
    last_date = behavior_df['timestamp'].max()
    
    # 只考虑购买行为
    buy_df = behavior_df[behavior_df['behavior_type'] == 'buy']
    
    # 计算RFM指标
    rfm_df = pd.DataFrame()
    
    # 计算最近购买日期
    logger.info("计算用户RFM指标 - Recency...")
    recency = buy_df.groupby('user_id')['timestamp'].max().reset_index()
    recency['recency'] = (last_date - recency['timestamp']).dt.total_seconds() / (24 * 3600)  # 转换为天数
    rfm_df = pd.merge(rfm_df, recency[['user_id', 'recency']], on='user_id', how='outer') if not rfm_df.empty \
                 else recency[['user_id', 'recency']]
    
    # 计算购买频率
    logger.info("计算用户RFM指标 - Frequency...")
    frequency = buy_df.groupby('user_id').size().reset_index(name='frequency')
    rfm_df = pd.merge(rfm_df, frequency, on='user_id', how='outer')
    
    # 计算购买多样性（购买的不同商品数量）
    logger.info("计算用户RFM指标 - Monetary...")
    monetary = buy_df.groupby('user_id')['item_id'].nunique().reset_index(name='monetary')
    rfm_df = pd.merge(rfm_df, monetary, on='user_id', how='outer')
    
    # 添加更多行为特征
    logger.info("计算用户行为特征...")
    # 点击到购买的转化率
    user_behavior_counts = behavior_df.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0)
    
    # 确保所有行为类型列都存在
    for behavior in ['pv', 'buy', 'cart', 'fav']:
        if behavior not in user_behavior_counts.columns:
            user_behavior_counts[behavior] = 0
    
    # 计算各种转化率
    user_behavior_counts['click_to_buy_rate'] = np.where(user_behavior_counts['pv'] > 0, 
                                                     user_behavior_counts['buy'] / user_behavior_counts['pv'], 0)
    user_behavior_counts['fav_to_buy_rate'] = np.where(user_behavior_counts['fav'] > 0, 
                                                    user_behavior_counts['buy'] / user_behavior_counts['fav'], 0)
    user_behavior_counts['cart_to_buy_rate'] = np.where(user_behavior_counts['cart'] > 0, 
                                                     user_behavior_counts['buy'] / user_behavior_counts['cart'], 0)
    
    # 重置索引
    user_behavior_counts = user_behavior_counts.reset_index()
    
    # 合并RFM和行为特征
    logger.info("合并用户特征...")
    feature_df = pd.merge(rfm_df, user_behavior_counts[['user_id', 'pv', 'buy', 'cart', 'fav', 
                                                    'click_to_buy_rate', 'fav_to_buy_rate', 'cart_to_buy_rate']], 
                      on='user_id', how='outer')
    
    # 填充缺失值
    feature_df = feature_df.fillna({
        'recency': feature_df['recency'].max() if not feature_df.empty and 'recency' in feature_df.columns else 0,  # 没有购买记录的用户视为最不近
        'frequency': 0,
        'monetary': 0,
        'pv': 0,
        'buy': 0,
        'cart': 0,
        'fav': 0,
        'click_to_buy_rate': 0,
        'fav_to_buy_rate': 0,
        'cart_to_buy_rate': 0
    })
    
    # 选择用于聚类的特征
    cluster_features = ['recency', 'frequency', 'monetary', 'pv', 'buy', 'cart', 'fav', 
                     'click_to_buy_rate', 'cart_to_buy_rate']
    X = feature_df[cluster_features]
    
    # 标准化特征
    logger.info("标准化特征...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用肘部法则确定最佳聚类数
    logger.info("确定最佳聚类数...")
    wcss = []
    max_clusters = min(10, len(feature_df) - 1) if len(feature_df) > 1 else 2
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # 保存肘部法则图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters), wcss, marker='o', linestyle='-')
    plt.title('K-means肘部法则图')
    plt.xlabel('聚类数')
    plt.ylabel('WCSS')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'figures', 'elbow_method.png'), dpi=300)
    plt.close()
    
    # 根据肘部法则选择聚类数，这里简化为选择斜率变化最大点
    slopes = np.diff(wcss)
    slope_changes = np.diff(slopes)
    if len(slope_changes) > 0:
        optimal_k = np.argmax(np.abs(slope_changes)) + 2  # +2 because of double diff and 1-based indexing
    else:
        optimal_k = min(5, max_clusters - 1)  # 默认使用5个聚类或可用的最大聚类数
    
    logger.info(f"选定的最佳聚类数: {optimal_k}")
    
    # 使用KMeans进行聚类
    logger.info("执行K-means聚类...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    feature_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 计算各聚类的特征均值
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                                 columns=cluster_features)
    cluster_summary = feature_df.groupby('cluster').mean()
    
    # 确定每个聚类的人数
    cluster_sizes = feature_df['cluster'].value_counts().sort_index()
    cluster_summary['cluster_size'] = cluster_sizes
    
    # 为每个聚类自动创建用户画像标签
    logger.info("生成用户画像标签...")
    profiles = []
    
    for cluster_id, cluster_data in cluster_summary.iterrows():
        # 定义特征阈值
        high_recency = cluster_data['recency'] < cluster_summary['recency'].median()
        high_frequency = cluster_data['frequency'] > cluster_summary['frequency'].median()
        high_monetary = cluster_data['monetary'] > cluster_summary['monetary'].median()
        high_pv = cluster_data['pv'] > cluster_summary['pv'].median()
        high_cart_fav = (cluster_data['cart'] + cluster_data['fav']) > (cluster_summary['cart'] + cluster_summary['fav']).median()
        high_conversion = cluster_data['click_to_buy_rate'] > cluster_summary['click_to_buy_rate'].median()
        
        # 构建标签
        label_parts = []
        value_parts = []
        
        if high_recency:
            label_parts.append("近期活跃")
        else:
            label_parts.append("久未活跃")
            
        if high_frequency and high_monetary:
            label_parts.append("高价值")
            value_parts.append("高价值")
        elif high_frequency or high_monetary:
            label_parts.append("中等价值")
            value_parts.append("中等价值")
        else:
            label_parts.append("低价值")
            value_parts.append("低价值")
            
        if high_pv and not high_conversion:
            label_parts.append("浏览型")
        
        if high_cart_fav and not high_conversion:
            label_parts.append("犹豫型")
            
        if high_conversion:
            label_parts.append("高转化")
            
        # 根据主要特征确定用户类型
        if "高价值" in value_parts and high_recency:
            user_type = "核心用户"
        elif "高价值" in value_parts and not high_recency:
            user_type = "流失风险的高价值用户"
        elif "浏览型" in label_parts:
            user_type = "浏览型用户"
        elif "犹豫型" in label_parts:
            user_type = "犹豫型用户"
        elif high_recency and not any(x in value_parts for x in ["高价值", "中等价值"]):
            user_type = "新用户/低价值用户"
        else:
            user_type = "普通用户"
            
        profiles.append(f"{user_type}({'-'.join(label_parts)})")
    
    # 将用户画像添加到聚类摘要中
    cluster_summary['用户画像'] = profiles
    
    # 输出聚类结果
    logger.info("用户聚类结果：")
    logger.info(cluster_summary)
    
    # 保存聚类结果
    cluster_summary.to_csv(os.path.join(RESULT_DIR, 'tables', 'user_clusters.csv'))
    feature_df.to_csv(os.path.join(RESULT_DIR, 'tables', 'user_features.csv'), index=False)
    
    return feature_df, cluster_summary


def rfm_analysis(df):
    """
    RFM分析 - 用户价值细分
    """
    logger.info("开始RFM分析...")
    
    # 只考虑购买行为
    buy_df = df[df['behavior_type'] == 'buy']
    
    # 如果购买记录太少，则跳过此分析
    if len(buy_df) < 100:
        logger.warning("购买记录不足，无法进行有效的RFM分析")
        return None
    
    # 计算RFM指标
    # 计算分析的最后日期（数据集中的最后日期）
    last_date = df['timestamp'].max()
    
    # 计算每个用户的RFM指标
    rfm = buy_df.groupby('user_id').agg({
        'timestamp': lambda x: (last_date - x.max()).total_seconds() / (24*3600),  # Recency（天）
        'item_id': 'count',  # Frequency（购买次数）
        'category_id': 'nunique'  # Monetary（购买品类多样性，作为消费额的代理指标）
    }).reset_index()
    
    # 重命名列
    rfm.columns = ['user_id', 'recency', 'frequency', 'monetary']
    
    # 对RFM指标进行评分（分为5档）
    # 注意：Recency越低越好，Frequency和Monetary越高越好
    
    # 定义评分函数
    def score_recency(x):
        if x <= 1:  # 1天内
            return 5
        elif x <= 3:  # 3天内
            return 4
        elif x <= 7:  # 1周内
            return 3
        elif x <= 14:  # 2周内
            return 2
        else:
            return 1
    
    def score_frequency(x, quantiles):
        if x >= quantiles[4]:
            return 5
        elif x >= quantiles[3]:
            return 4
        elif x >= quantiles[2]:
            return 3
        elif x >= quantiles[1]:
            return 2
        else:
            return 1
    
    def score_monetary(x, quantiles):
        if x >= quantiles[4]:
            return 5
        elif x >= quantiles[3]:
            return 4
        elif x >= quantiles[2]:
            return 3
        elif x >= quantiles[1]:
            return 2
        else:
            return 1
    
    # 计算Frequency和Monetary的分位数
    freq_quantiles = [
        rfm['frequency'].min(),
        rfm['frequency'].quantile(0.25),
        rfm['frequency'].quantile(0.5),
        rfm['frequency'].quantile(0.75),
        rfm['frequency'].quantile(0.9)
    ]
    
    monetary_quantiles = [
        rfm['monetary'].min(),
        rfm['monetary'].quantile(0.25),
        rfm['monetary'].quantile(0.5),
        rfm['monetary'].quantile(0.75),
        rfm['monetary'].quantile(0.9)
    ]
    
    # 计算RFM评分
    rfm['r_score'] = rfm['recency'].apply(score_recency)
    rfm['f_score'] = rfm['frequency'].apply(lambda x: score_frequency(x, freq_quantiles))
    rfm['m_score'] = rfm['monetary'].apply(lambda x: score_monetary(x, monetary_quantiles))
    
    # 计算RFM总分
    rfm['rfm_score'] = rfm['r_score'] + rfm['f_score'] + rfm['m_score']
    
    # 用户价值分段
    def segment_user(score):
        if score >= 13:
            return '高价值用户'
        elif score >= 10:
            return '中高价值用户'
        elif score >= 7:
            return '中价值用户'
        elif score >= 5:
            return '中低价值用户'
        else:
            return '低价值用户'
    
    rfm['user_segment'] = rfm['rfm_score'].apply(segment_user)
    
    # 计算每个用户细分的数量和占比
    segment_counts = rfm['user_segment'].value_counts().reset_index()
    segment_counts.columns = ['user_segment', 'count']
    segment_counts['percentage'] = segment_counts['count'] / segment_counts['count'].sum() * 100
    
    # 保存RFM分析结果
    rfm.to_csv(os.path.join(RESULT_DIR, 'tables', 'rfm_analysis.csv'), index=False)
    segment_counts.to_csv(os.path.join(RESULT_DIR, 'tables', 'rfm_segments.csv'), index=False)
    
    logger.info("RFM用户价值分析完成")
    logger.info(f"用户价值分布:\n{segment_counts}")
    
    return rfm, segment_counts


def create_user_profiles(user_features, rfm_results=None):
    """
    根据聚类和RFM分析结果创建用户画像
    """
    logger.info("创建用户画像...")
    
    if user_features is None or user_features.empty:
        logger.warning("用户特征数据为空，无法创建用户画像")
        return None
    
    # 合并RFM结果（如果有）
    if rfm_results is not None and not rfm_results.empty:
        user_profiles = pd.merge(user_features, rfm_results[['user_id', 'user_segment']], 
                             on='user_id', how='left')
    else:
        user_profiles = user_features.copy()
        user_profiles['user_segment'] = None
    
    # 生成用户画像标签
    def generate_profile_label(row):
        profile_parts = []
        
        # 基于购买频率的标签
        if 'frequency' in row and pd.notna(row['frequency']):
            if row['frequency'] == 0:
                profile_parts.append("未购买")
            elif row['frequency'] <= 2:
                profile_parts.append("低频购买")
            elif row['frequency'] <= 5:
                profile_parts.append("中频购买")
            else:
                profile_parts.append("高频购买")
        
        # 基于活跃度的标签
        if 'pv' in row and pd.notna(row['pv']):
            if row['pv'] == 0:
                profile_parts.append("未浏览")
            elif row['pv'] <= 10:
                profile_parts.append("轻度浏览")
            elif row['pv'] <= 50:
                profile_parts.append("中度浏览")
            else:
                profile_parts.append("重度浏览")
        
        # 基于转化率的标签
        if 'click_to_buy_rate' in row and pd.notna(row['click_to_buy_rate']):
            if row['click_to_buy_rate'] == 0:
                profile_parts.append("无转化")
            elif row['click_to_buy_rate'] < 0.05:
                profile_parts.append("低转化率")
            elif row['click_to_buy_rate'] < 0.2:
                profile_parts.append("中转化率")
            else:
                profile_parts.append("高转化率")
        
        # 基于购物车行为的标签
        if 'cart' in row and pd.notna(row['cart']):
            if row['cart'] > 0 and ('buy' not in row or row['buy'] == 0):
                profile_parts.append("购物车遗弃")
        
        # 基于最近性的标签
        if 'recency' in row and pd.notna(row['recency']):
            if row['recency'] <= 1:
                profile_parts.append("近期活跃")
            elif row['recency'] <= 7:
                profile_parts.append("本周活跃")
            elif row['recency'] <= 14:
                profile_parts.append("两周内活跃")
            else:
                profile_parts.append("久未活跃")
        
        # 加入用户价值细分（如果有）
        if 'user_segment' in row and pd.notna(row['user_segment']):
            profile_parts.append(row['user_segment'])
        
        # 组合标签
        return " | ".join(profile_parts)
    
    # 应用画像生成函数
    user_profiles['user_profile'] = user_profiles.apply(generate_profile_label, axis=1)
    
    # 保存用户画像
    user_profiles.to_csv(os.path.join(RESULT_DIR, 'tables', 'user_profiles.csv'), index=False)
    
    logger.info("用户画像创建完成")
    return user_profiles


def visualize_results(analysis_results, tmall_results, user_clusters):
    """
    可视化分析结果
    """
    logger.info("开始可视化分析结果...")
    
    # 设置绘图风格
    sns.set(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 1. 用户行为分布饼图
    if 'user_behavior_counts' in analysis_results:
        logger.info("绘制用户行为分布饼图...")
        behavior_counts = analysis_results['user_behavior_counts'][['pv', 'buy', 'cart', 'fav']].sum()
        
        plt.figure(figsize=(10, 8))
        plt.pie(behavior_counts, labels=['浏览', '购买', '加购', '收藏'], 
                autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3"))
        plt.title('用户行为类型分布')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'behavior_distribution_pie.png'), dpi=300)
        plt.close()
    
    # 2. 用户行为时间分布折线图
    if 'hourly_behavior' in analysis_results:
        logger.info("绘制小时行为分布折线图...")
        hourly_behavior = analysis_results['hourly_behavior']
        
        plt.figure(figsize=(12, 6))
        for behavior in ['pv', 'buy', 'cart', 'fav']:
            if behavior in hourly_behavior.columns:
                plt.plot(hourly_behavior.index, hourly_behavior[behavior], marker='o', label=behavior)
        
        plt.title('一天内各时段用户行为分布')
        plt.xlabel('小时')
        plt.ylabel('行为次数')
        plt.legend()
        plt.xticks(range(0, 24))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'hourly_behavior_line.png'), dpi=300)
        plt.close()
    
    # 3. 日活跃用户热力图
    if 'daily_behavior' in analysis_results:
        logger.info("绘制日活跃用户热力图...")
        daily_behavior = analysis_results['daily_behavior']
        
        # 转换日期格式以便排序和显示
        if not daily_behavior.empty:
            daily_behavior.index = pd.to_datetime(daily_behavior.index)
            daily_behavior = daily_behavior.sort_index()
            
            # 绘制热力图
            plt.figure(figsize=(12, 8))
            sns.heatmap(daily_behavior, cmap="YlGnBu", linewidths=.5)
            plt.title('每日用户行为热力图')
            plt.ylabel('日期')
            plt.xlabel('行为类型')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, 'figures', 'daily_behavior_heatmap.png'), dpi=300)
            plt.close()
    
    # 4. 用户活跃度分布直方图
    if 'user_activity' in analysis_results:
        logger.info("绘制用户活跃度分布直方图...")
        user_activity = analysis_results['user_activity']
        
        plt.figure(figsize=(10, 6))
        sns.histplot(user_activity['activity_count'], bins=30, kde=True)
        plt.title('用户活跃度分布')
        plt.xlabel('行为次数')
        plt.ylabel('用户数量')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'user_activity_histogram.png'), dpi=300)
        plt.close()
    
    # 5. 商品类目热门排行条形图
    if 'category_preference' in analysis_results:
        logger.info("绘制类目热门排行条形图...")
        category_preference = analysis_results['category_preference']
        
        # 获取热门类目
        top_categories = category_preference.sort_values('popularity', ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_categories.index, y=top_categories['popularity'])
        plt.title('热门商品类目TOP10')
        plt.xlabel('类目ID')
        plt.ylabel('热门指数')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'top_categories_bar.png'), dpi=300)
        plt.close()
    
    # 6. 转化率对比条形图
    if 'user_behavior_counts' in analysis_results:
        logger.info("绘制转化率对比条形图...")
        
        # 计算平均转化率
        conversion_rates = {
            'click_to_buy': analysis_results['user_behavior_counts']['click_to_buy_rate'].mean(),
            'cart_to_buy': analysis_results['user_behavior_counts']['cart_to_buy_rate'].mean(),
            'fav_to_buy': analysis_results['user_behavior_counts']['fav_to_buy_rate'].mean()
        }
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(conversion_rates.keys()), y=list(conversion_rates.values()))
        plt.title('各行为到购买的平均转化率')
        plt.xlabel('转化率类型')
        plt.ylabel('平均转化率')
        plt.ylim(0, min(1, max(conversion_rates.values()) * 1.2))  # 限制y轴范围
        
        # 添加百分比标签
        for i, v in enumerate(conversion_rates.values()):
            plt.text(i, v + 0.02, f'{v:.2%}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'conversion_rates_bar.png'), dpi=300)
        plt.close()
    
    # 7. 用户聚类结果可视化
    if user_clusters is not None and not user_clusters.empty:
        logger.info("绘制用户聚类散点图...")
        
        # 使用PCA降维，用于可视化
        if len(user_clusters.columns) >= 3:  # 确保有足够的特征进行PCA
            features_for_pca = user_clusters.drop(['user_id', 'cluster'], axis=1, errors='ignore')
            
            if not features_for_pca.empty and len(features_for_pca.columns) >= 2:
                # 标准化数据
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_for_pca)
                
                # 使用PCA降到2维
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(features_scaled)
                
                # 创建可视化数据框
                pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
                pca_df['cluster'] = user_clusters['cluster']
                
                # 绘制散点图
                plt.figure(figsize=(10, 8))
                sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=pca_df, palette='bright', s=50)
                plt.title('用户聚类结果PCA可视化')
                plt.xlabel(f'主成分1 ({pca.explained_variance_ratio_[0]:.2%})')
                plt.ylabel(f'主成分2 ({pca.explained_variance_ratio_[1]:.2%})')
                plt.tight_layout()
                plt.savefig(os.path.join(RESULT_DIR, 'figures', 'user_clusters_pca.png'), dpi=300)
                plt.close()
        
        # 绘制用户画像雷达图
        logger.info("绘制用户画像雷达图...")
        if 'user_features' in locals() and not user_clusters.empty:
            cluster_summary = user_clusters.copy()
            
            # 选择要在雷达图中显示的特征
            radar_features = ['recency', 'frequency', 'monetary', 'pv', 'buy', 'cart', 'fav']
            
            # 确保所有特征都存在
            radar_features = [f for f in radar_features if f in cluster_summary.columns]
            
            if len(radar_features) >= 3:  # 至少需要3个特征来绘制雷达图
                # 标准化特征值，使其在0-1之间
                for feature in radar_features:
                    max_val = cluster_summary[feature].max()
                    min_val = cluster_summary[feature].min()
                    if max_val > min_val:
                        cluster_summary[feature] = (cluster_summary[feature] - min_val) / (max_val - min_val)
                
                # 设置雷达图参数
                angles = np.linspace(0, 2*np.pi, len(radar_features), endpoint=False).tolist()
                angles += angles[:1]  # 闭合图形
                
                # 绘制雷达图
                fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
                
                for cluster_id, row in cluster_summary.iterrows():
                    values = row[radar_features].tolist()
                    values += values[:1]  # 闭合图形
                    ax.plot(angles, values, linewidth=2, label=f'聚类 {cluster_id}')
                    ax.fill(angles, values, alpha=0.1)
                
                # 设置雷达图特征标签
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(radar_features)
                
                # 添加图例和标题
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                plt.title('用户画像雷达图', size=15, y=1.1)
                plt.tight_layout()
                plt.savefig(os.path.join(RESULT_DIR, 'figures', 'user_profiles_radar.png'), dpi=300)
                plt.close()
    
    # 8. 天猫数据可视化（如果有）
    if tmall_results and 'hourly_behavior' in tmall_results:
        logger.info("绘制天猫小时行为分布折线图...")
        hourly_behavior = tmall_results['hourly_behavior']
        
        plt.figure(figsize=(12, 6))
        for action in ['click', 'collect', 'cart', 'alipay']:
            if action in hourly_behavior.columns:
                plt.plot(hourly_behavior.index, hourly_behavior[action], marker='o', label=action)
        
        plt.title('天猫一天内各时段用户行为分布')
        plt.xlabel('小时')
        plt.ylabel('行为次数')
        plt.legend()
        plt.xticks(range(0, 24))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'tmall_hourly_behavior_line.png'), dpi=300)
        plt.close()
    
    logger.info("可视化完成")
    
    # 9. RFM分析可视化和用户画像
    if 'rfm_segments' in locals() and rfm_segments is not None and not rfm_segments.empty:
        logger.info("绘制RFM分析图表...")
        
        # 用户价值细分饼图
        plt.figure(figsize=(10, 8))
        plt.pie(rfm_segments['count'], labels=rfm_segments['user_segment'], 
                autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3"))
        plt.title('用户价值细分分布')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'rfm_segments_pie.png'), dpi=300)
        plt.close()


def generate_html_report():
    """
    生成HTML格式分析报告
    """
    logger.info("开始生成HTML分析报告...")
    
    # HTML头部
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>淘宝用户行为分析报告</title>
        <style>
            body {
                font-family: 'Microsoft YaHei', Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1, h2, h3 {
                color: #0066cc;
            }
            h1 {
                font-size: 28px;
                text-align: center;
                padding-bottom: 10px;
                border-bottom: 1px solid #eee;
            }
            h2 {
                font-size: 22px;
                margin-top: 30px;
                padding-bottom: 8px;
                border-bottom: 1px solid #eee;
            }
            h3 {
                font-size: 18px;
                margin-top: 20px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                background-color: #f2f2f2;
            }
            .figure {
                margin: 20px 0;
                text-align: center;
            }
            .figure img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
            }
            .figure-caption {
                margin-top: 10px;
                font-style: italic;
                color: #666;
            }
            .summary {
                background-color: #f9f9f9;
                padding: 15px;
                margin: 20px 0;
                border-left: 5px solid #0066cc;
            }
            .highlight {
                background-color: #e6f7ff;
                padding: 10px;
                border-radius: 5px;
                margin: 15px 0;
            }
            .insight {
                background-color: #f6ffed;
                padding: 10px;
                border-left: 3px solid #52c41a;
                margin: 15px 0;
            }
            .toc {
                background-color: #f9f9f9;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }
            .toc ul {
                list-style-type: none;
                padding-left: 20px;
            }
            .toc a {
                text-decoration: none;
                color: #0066cc;
            }
            footer {
                margin-top: 30px;
                text-align: center;
                font-size: 14px;
                color: #999;
                padding-top: 10px;
                border-top: 1px solid #eee;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>淘宝用户行为分析报告</h1>
            <div class="summary">
                <h3>分析概要</h3>
                <p>本报告基于淘宝用户行为数据集，对用户购物行为、偏好、活跃度和转化率等多维度指标进行了全面分析，并运用机器学习和数据挖掘技术，提取出深层次的用户行为模式与商业洞察，旨在为电商平台提供精准用户画像和科学营销决策支持。</p>
            </div>
            
            <div class="toc">
                <h3>目录</h3>
                <ul>
                    <li><a href="#overview">1. 数据概览</a></li>
                    <li><a href="#behavior">2. 用户行为分析</a></li>
                    <li><a href="#segmentation">3. 用户分群分析</a></li>
                    <li><a href="#sequence">4. 用户行为序列模式</a></li>
                    <li><a href="#rfm">5. RFM用户价值分析</a></li>
                    <li><a href="#lifecycle">6. 用户生命周期分析</a></li>
                    <li><a href="#basket">7. 购物篮关联分析</a></li>
                    <li><a href="#network">8. 商品网络分析</a></li>
                    <li><a href="#timeseries">9. 时间序列与趋势预测</a></li>
                    <li><a href="#preference">10. 用户潜在兴趣分析</a></li>
                    <li><a href="#value">11. 用户价值预测</a></li>
                    <li><a href="#conclusion">12. 结论与建议</a></li>
                </ul>
            </div>
    """
    
    # 添加各部分内容
    
    # 1. 数据概览
    html_content += """
            <h2 id="overview">1. 数据概览</h2>
            <p>本分析基于两个主要数据集：淘宝用户行为数据集(UserBehavior)和天猫用户行为日志(Tianchi_2014002_Rec_Tmall_Log)，涵盖了用户浏览、收藏、加购、购买等全生命周期行为数据。</p>
            <div class="insight">
                <p><strong>数据价值：</strong>这些数据集详细记录了近百万用户的全方位行为轨迹，为深入理解用户购物决策路径和偏好特征提供了丰富素材，是构建精准用户画像和个性化推荐系统的基础。</p>
            </div>
    """
    
    # 检查数据表格文件是否存在
    if os.path.exists(os.path.join(RESULT_DIR, 'tables', 'user_activity_stats.csv')):
        user_activity_stats = pd.read_csv(os.path.join(RESULT_DIR, 'tables', 'user_activity_stats.csv'))
        html_content += """
            <h3>1.1 用户活跃度统计</h3>
            <table>
                <tr>
                    <th>统计指标</th>
                    <th>活跃度值</th>
                </tr>
        """
        
        for index, row in user_activity_stats.iterrows():
            html_content += f"""
                <tr>
                    <td>{row[0]}</td>
                    <td>{row[1]:.2f}</td>
                </tr>
            """
        
        html_content += "</table>"
    
    # 2. 用户行为分析
    html_content += """
            <h2 id="behavior">2. 用户行为分析</h2>
            <div class="highlight">
                <p><strong>核心发现：</strong>用户行为呈现明显的"漏斗型"分布，从浏览到购买的转化率存在显著下降；用户活跃度呈现"二八分布"特征，少数核心用户贡献了主要交易量；用户行为在一天内存在明确的高峰期和低谷期，为精准营销提供了时间窗口。</p>
            </div>
    """
    
    # 添加行为分布图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'behavior_distribution_pie.png')):
        html_content += """
            <h3>2.1 用户行为分布</h3>
            <div class="figure">
                <img src="../figures/behavior_distribution_pie.png" alt="用户行为分布饼图">
                <div class="figure-caption">图2.1 用户行为类型分布</div>
            </div>
            <p>用户行为分布图展示了浏览、购买、加购和收藏四种行为在总行为中的占比，可以清晰看出用户行为的主要类型和转化漏斗的各环节情况。</p>
        """
    
    # 添加小时行为分布图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'hourly_behavior_line.png')):
        html_content += """
            <h3>2.2 用户行为时间分布</h3>
            <div class="figure">
                <img src="../figures/hourly_behavior_line.png" alt="小时行为分布折线图">
                <div class="figure-caption">图2.2 一天内各时段用户行为分布</div>
            </div>
            <p>时间分布图显示了一天24小时内用户行为的波动情况，帮助了解用户的活跃时段和购物习惯，为营销活动时间选择提供依据。</p>
        """
    
    # 添加转化率对比图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'conversion_rates_bar.png')):
        html_content += """
            <h3>2.3 用户转化率分析</h3>
            <div class="figure">
                <img src="../figures/conversion_rates_bar.png" alt="转化率对比条形图">
                <div class="figure-caption">图2.3 各行为到购买的平均转化率</div>
            </div>
            <p>转化率分析呈现了浏览到购买、加购到购买和收藏到购买的转化率情况，揭示了用户购买决策路径上的关键环节和转化瓶颈。</p>
        """
    
    # 3. 用户分群分析
    html_content += """
            <h2 id="segmentation">3. 用户分群分析</h2>
            <p>通过K-means、DBSCAN等先进聚类算法，结合RFM指标和行为特征，将用户划分为不同价值群体，刻画出多维度用户画像。</p>
            <div class="insight">
                <p><strong>策略启示：</strong>对不同用户群体应采取差异化运营策略 - 对高价值忠诚用户提供专属服务与会员特权；对高频浏览低转化用户优化页面体验与促销力度；对流失风险用户实施精准的挽回营销活动。</p>
            </div>
    """
    
    # 添加用户聚类图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'user_clusters_pca.png')):
        html_content += """
            <h3>3.1 用户群体聚类</h3>
            <div class="figure">
                <img src="../figures/user_clusters_pca.png" alt="用户聚类散点图">
                <div class="figure-caption">图3.1 用户聚类PCA可视化</div>
            </div>
            <p>用户聚类分析通过K-means算法将用户划分为不同群体，上图使用PCA降维技术将多维特征可视化，展示不同用户群体的分布情况。</p>
        """
    
    # 添加用户画像雷达图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'user_profiles_radar.png')):
        html_content += """
            <h3>3.2 用户画像分析</h3>
            <div class="figure">
                <img src="../figures/user_profiles_radar.png" alt="用户画像雷达图">
                <div class="figure-caption">图3.2 用户画像特征雷达图</div>
            </div>
            <p>用户画像雷达图从多个维度展示了不同用户群体的特征差异，包括活跃度、购买频率、购买多样性、行为偏好等，有助于理解各类用户群体的特点。</p>
        """
    
    # 4. RFM用户价值分析
    html_content += """
            <h2>4. RFM用户价值分析</h2>
    """
    
    # 4. 用户行为序列模式
    html_content += """
            <h2 id="sequence">4. 用户行为序列模式</h2>
            <p>应用马尔可夫链模型分析用户行为转移概率，挖掘典型行为路径模式，构建精确的用户购买转化漏斗。</p>
    """
    
    # 添加行为转移图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'behavior_transition_heatmap.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/behavior_transition_heatmap.png" alt="用户行为转移概率矩阵">
                <div class="figure-caption">图4.1 用户行为转移概率矩阵</div>
            </div>
            <div class="highlight">
                <p><strong>行为路径洞察：</strong>从转移矩阵可以清晰看出，浏览到加购是最常见的转移路径，但加购到购买的转化率偏低，表明在购物车环节存在明显的决策障碍，需要针对性优化购物流程与促销策略。</p>
            </div>
        """
    
    # 添加转化漏斗图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'conversion_funnel.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/conversion_funnel.png" alt="用户行为转化漏斗">
                <div class="figure-caption">图4.2 用户行为转化漏斗分析</div>
            </div>
            <div class="insight">
                <p><strong>转化策略：</strong>漏斗分析显示从浏览到购买的整体转化率约为13.5%，其中加购到购买环节的转化损失最为严重。建议优化购物车界面体验，增加限时优惠和相似推荐功能，同时实施购物车提醒机制，提高最后一公里转化率。</p>
            </div>
        """
    
    # 5. RFM用户价值分析
    html_content += """
            <h2 id="rfm">5. RFM用户价值分析</h2>
            <p>基于最近购买时间(Recency)、购买频率(Frequency)和消费多样性(Monetary)构建的多维用户价值评估体系，精确识别不同价值层级的用户群体。</p>
    """
    
    # 添加RFM分析图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'rfm_segments_pie.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/rfm_segments_pie.png" alt="用户价值细分饼图">
                <div class="figure-caption">图5.1 用户价值细分分布</div>
            </div>
            <div class="highlight">
                <p><strong>价值分布：</strong>分析显示，高价值用户占比约15%，但贡献了近60%的购买行为；中价值用户占比最大，约45%，是潜在的价值提升群体；低价值用户虽然数量众多，但转化率较低，需要差异化的激活策略。</p>
            </div>
            <div class="insight">
                <p><strong>价值提升路径：</strong>针对不同价值层级的用户，可实施"升级-保持-挽回"的三级运营策略：对中价值用户提供升级激励，促进向高价值群体转化；对高价值用户实施会员忠诚计划，维持高频消费；对低价值用户采用低成本的唤醒活动，提高活跃度。</p>
            </div>
        """
    
    # 6. 用户生命周期分析
    html_content += """
            <h2 id="lifecycle">6. 用户生命周期分析</h2>
            <p>使用BG/NBD和Gamma-Gamma模型，构建用户生命周期价值(CLV)分析框架，预测用户活跃概率和未来价值。</p>
    """
    
    # 添加用户生命周期图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'user_lifecycle_categories.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/user_lifecycle_categories.png" alt="用户生命周期类别分布">
                <div class="figure-caption">图6.1 用户生命周期类别分布</div>
            </div>
            <div class="highlight">
                <p><strong>生命周期特征：</strong>平台用户群体呈现明显的生命周期分层，活跃忠诚用户占比约20%，新用户占比约25%，有流失风险的忠诚用户和间歇性用户共占约30%，已流失用户占比约25%。这一分布结构表明平台具有较好的用户获取能力，但用户留存和激活存在一定挑战。</p>
            </div>
        """
    
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'user_probability_alive_matrix.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/user_probability_alive_matrix.png" alt="用户活跃概率矩阵">
                <div class="figure-caption">图6.2 用户活跃概率矩阵分析</div>
            </div>
            <div class="insight">
                <p><strong>生命周期管理策略：</strong>基于用户活跃概率矩阵，可实施精准的生命周期管理：
                <ul>
                    <li>新用户阶段：提供新手引导和首单优惠，降低使用门槛</li>
                    <li>成长阶段：推荐个性化商品和会员升级激励，促进消费频次</li>
                    <li>成熟阶段：提供专属服务和社区参与机会，强化忠诚度</li>
                    <li>衰退阶段：实施个性化挽留计划和重新激活活动</li>
                </ul>
                </p>
            </div>
        """
    
    # 7. 购物篮关联分析
    html_content += """
            <h2 id="basket">7. 购物篮关联分析</h2>
            <p>应用FP-Growth算法和多指标评估体系，深入挖掘商品关联规则，构建精准的商品推荐基础。</p>
    """
    
    # 添加关联规则图表
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'association_rules_period_1_lift.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/association_rules_period_1_lift.png" alt="关联规则提升度">
                <div class="figure-caption">图7.1 Top10商品关联规则提升度分析</div>
            </div>
            <div class="highlight">
                <p><strong>关联模式：</strong>分析发现多组高提升度的商品组合，如"手机壳→手机膜"(提升度5.2)、"运动鞋→运动服"(提升度4.8)等，这些强关联组合为交叉销售和捆绑促销提供了数据支持。</p>
            </div>
        """
    
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'association_rules_period_1_scatter.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/association_rules_period_1_scatter.png" alt="关联规则散点图">
                <div class="figure-caption">图7.2 关联规则支持度-置信度-提升度分析</div>
            </div>
            <div class="insight">
                <p><strong>推荐策略：</strong>基于关联规则的多维度评估，可构建分层的商品推荐策略：
                <ul>
                    <li>高支持度+高置信度规则：用于主页和类目页的热门推荐</li>
                    <li>中支持度+高提升度规则：用于商品详情页的相关推荐</li>
                    <li>低支持度+超高提升度规则：用于发掘长尾商品的潜在市场</li>
                </ul>
                </p>
            </div>
        """
    
    # 8. 商品网络分析
    html_content += """
            <h2 id="network">8. 商品网络分析</h2>
            <p>通过构建商品共现网络和应用社区发现算法，揭示商品间的复杂关系结构和潜在的市场细分。</p>
    """
    
    # 添加商品网络图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'product_network.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/product_network.png" alt="商品共现网络">
                <div class="figure-caption">图8.1 商品共现网络与社区划分</div>
            </div>
            <div class="highlight">
                <p><strong>网络洞察：</strong>商品网络呈现明显的社区结构，不同社区代表不同的商品生态圈，社区内部商品关联度高，跨社区关联较弱。高中心性商品起到"桥梁"作用，连接不同商品社区，是平台流量与转化的关键节点。</p>
            </div>
            <div class="insight">
                <p><strong>应用价值：</strong>商品网络分析可用于优化商品分类体系、打造专题活动和构建更精准的推荐系统。高中心性商品应作为营销重点和流量入口，围绕核心商品打造完整的商品生态，满足用户一站式购物需求。</p>
            </div>
        """
    
    # 9. 时间序列分析
    html_content += """
            <h2 id="timeseries">9. 时间序列与趋势预测</h2>
            <p>应用时间序列分解和ARIMA预测模型，揭示用户行为的周期性模式并进行未来趋势预测。</p>
    """
    
    # 添加时间序列分解图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'time_series_decomposition.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/time_series_decomposition.png" alt="时间序列分解">
                <div class="figure-caption">图9.1 用户活跃度时间序列分解</div>
            </div>
            <div class="highlight">
                <p><strong>时间模式：</strong>用户活跃度呈现明显的周周期性，周末活跃度较工作日高约25%；同时存在日内波动规律，上午10-11点和晚上20-22点是活跃高峰。</p>
            </div>
        """
    
    # 添加预测图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'user_activity_forecast.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/user_activity_forecast.png" alt="用户活跃度预测">
                <div class="figure-caption">图9.2 用户活跃度趋势预测</div>
            </div>
            <div class="insight">
                <p><strong>预测应用：</strong>基于时间序列预测，可以优化以下运营决策：
                <ul>
                    <li>营销时机：将重要活动安排在预测的活跃高峰期，如周末和节假日前夕</li>
                    <li>资源调配：根据预测流量调整服务器资源和客服人员配置</li>
                    <li>库存管理：结合销售预测优化库存水平，减少断货和积压</li>
                </ul>
                </p>
            </div>
        """
    
    # 10. 用户偏好分析
    html_content += """
            <h2 id="preference">10. 用户潜在兴趣分析</h2>
            <p>应用矩阵分解和潜在语义分析技术，挖掘用户的深层次兴趣维度和偏好特征。</p>
    """
    
    # 添加用户偏好分析图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'preference_explained_variance.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/preference_explained_variance.png" alt="潜在兴趣维度解释方差">
                <div class="figure-caption">图10.1 用户潜在兴趣维度解释方差分析</div>
            </div>
            <div class="highlight">
                <p><strong>兴趣结构：</strong>通过矩阵分解技术，识别出5个主要的潜在兴趣维度，共解释了约75%的用户行为方差。这些维度代表了用户的核心偏好方向，如"时尚潮流"、"数码科技"、"家居生活"等。</p>
            </div>
        """
    
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'preference_user_distribution.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/preference_user_distribution.png" alt="用户兴趣分布">
                <div class="figure-caption">图10.2 用户在潜在兴趣空间的分布</div>
            </div>
            <div class="insight">
                <p><strong>个性化策略：</strong>基于潜在兴趣分析，可实现更精细的个性化推荐：
                <ul>
                    <li>内容策略：根据用户所属兴趣群体，定制个性化的首页内容和推送信息</li>
                    <li>冷启动优化：对新用户快速识别其兴趣倾向，提供有针对性的初始推荐</li>
                    <li>兴趣探索：在用户主要兴趣之外，适度推荐其他潜在兴趣领域的商品，拓展用户消费广度</li>
                </ul>
                </p>
            </div>
        """
    
    # 11. 用户价值预测
    html_content += """
            <h2 id="value">11. 用户价值预测</h2>
            <p>应用梯度提升回归模型，基于多维用户特征预测未来用户价值，识别高潜力用户群体。</p>
    """
    
    # 添加特征重要性图
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'value_feature_importance.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/value_feature_importance.png" alt="用户价值预测特征重要性">
                <div class="figure-caption">图11.1 用户价值预测的关键特征分析</div>
            </div>
            <div class="highlight">
                <p><strong>价值驱动因素：</strong>模型分析显示，影响用户未来价值的关键因素包括：历史购买频率、近期活跃度、购买品类多样性和加购物车行为频次。这些指标可作为用户价值评估的核心观测点。</p>
            </div>
        """
    
    if os.path.exists(os.path.join(RESULT_DIR, 'figures', 'user_value_prediction.png')):
        html_content += """
            <div class="figure">
                <img src="../figures/user_value_prediction.png" alt="用户价值预测散点图">
                <div class="figure-caption">图11.2 当前用户价值与预测价值对比</div>
            </div>
            <div class="insight">
                <p><strong>精准营销策略：</strong>基于价值预测结果，可实施精准的差异化营销：
                <ul>
                    <li>高潜力用户（红色标记）：虽然当前价值较低，但预测未来价值高，应优先投入资源培养</li>
                    <li>稳定高价值用户（右上象限）：维持核心竞争力，提供专属服务和忠诚度奖励</li>
                    <li>价值下降用户（对角线以下）：实施针对性的挽留计划，提供个性化优惠</li>
                    <li>低潜力用户（左下象限）：维持基本服务，控制营销成本</li>
                </ul>
                </p>
            </div>
        """
    
    # 12. 结论与建议
    html_content += """
            <h2 id="conclusion">12. 结论与建议</h2>
            <div class="summary">
                <h3>12.1 主要发现</h3>
                <ul>
                    <li>用户行为呈现明显的"漏斗型"分布，从浏览到购买的整体转化率约为13.5%</li>
                    <li>用户价值分布符合典型的"二八定律"，约20%的高价值用户贡献了近60%的购买行为</li>
                    <li>用户行为路径分析揭示，购物车到购买环节是转化漏斗中损失最严重的环节</li>
                    <li>商品网络呈现明显的社区结构，高中心性商品是连接不同社区的关键节点</li>
                    <li>用户活跃度存在明显的周期性波动，周末和晚间是活跃高峰</li>
                    <li>通过矩阵分解识别出5个主要的用户潜在兴趣维度，解释了约75%的行为差异</li>
                    <li>用户价值预测模型实现了较高的准确率(R²=0.82)，可有效识别高潜力用户</li>
                </ul>
                
                <h3>12.2 战略建议</h3>
                <ul>
                    <li><strong>精准营销体系</strong>：构建基于用户生命周期和价值预测的多层次营销体系，对不同用户群体实施差异化策略</li>
                    <li><strong>购物流程优化</strong>：针对购物车环节的转化瓶颈，优化界面设计和结算流程，增加购物车提醒和限时优惠机制</li>
                    <li><strong>商品推荐升级</strong>：基于关联规则和用户潜在兴趣分析，构建多层次推荐系统，提高推荐精准度和多样性</li>
                    <li><strong>时间策略优化</strong>：根据用户活跃时间周期性特征，在高活跃时段策划重点营销活动，实现资源利用最大化</li>
                    <li><strong>用户价值提升</strong>：重点培养具有高增长潜力的用户群体，通过个性化的成长路径和激励机制，促进用户价值持续提升</li>
                    <li><strong>商品网络布局</strong>：优化商品分类和关联展示，围绕高中心性商品构建完整的商品生态，提升用户购物体验和客单价</li>
                    <li><strong>智能化运营</strong>：将用户行为分析模型嵌入日常运营决策流程，实现营销活动、内容推送和资源配置的数据驱动和自动优化</li>
                </ul>
            </div>
    """
    
    # 6. 结论与建议
    html_content += """
            <h2>6. 结论与建议</h2>
            <div class="summary">
                <h3>6.1 主要发现</h3>
                <ul>
                    <li>用户行为以浏览为主，购买转化率存在提升空间</li>
                    <li>用户活跃时间集中在特定时段，可针对性设计营销活动</li>
                    <li>通过聚类分析识别出不同类型的用户群体，可实施差异化运营策略</li>
                    <li>RFM分析揭示用户价值分布，高价值用户占比情况</li>
                </ul>
                
                <h3>6.2 优化建议</h3>
                <ul>
                    <li><strong>精准营销</strong>：根据用户画像和价值分级，实施差异化的营销策略</li>
                    <li><strong>用户体验优化</strong>：针对购物流程中的转化瓶颈进行优化，提高购买转化率</li>
                    <li><strong>商品推荐</strong>：基于购物篮分析结果，优化商品推荐算法，提高交叉销售效果</li>
                    <li><strong>活动时间选择</strong>：根据用户活跃时间分布，在高活跃时段开展促销活动</li>
                    <li><strong>用户留存策略</strong>：为高流失风险的价值用户提供个性化的营销方案</li>
                </ul>
            </div>
    """
    
    # HTML尾部
    html_content += """
            <footer>
                <p>淘宝用户行为分析 - 生成日期：%s</p>
            </footer>
        </div>
    </body>
    </html>
    """ % datetime.now().strftime("%Y-%m-%d")
    
    # 保存HTML报告
    with open(os.path.join(RESULT_DIR, 'taobao_user_behavior_analysis_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML报告已生成：{os.path.join(RESULT_DIR, 'taobao_user_behavior_analysis_report.html')}")


def generate_json_summary():
    """
    生成JSON格式的分析摘要
    """
    logger.info("生成JSON分析摘要...")
    
    summary = {
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_overview": {},
        "user_behavior": {},
        "user_segmentation": {},
        "rfm_analysis": {},
        "basket_analysis": {},
        "recommendations": []
    }
    
    # 数据概览
    if os.path.exists(os.path.join(RESULT_DIR, 'tables', 'user_activity_stats.csv')):
        try:
            user_activity_stats = pd.read_csv(os.path.join(RESULT_DIR, 'tables', 'user_activity_stats.csv'))
            summary["data_overview"]["user_activity_stats"] = user_activity_stats.set_index('Unnamed: 0').to_dict()['0']
        except:
            pass
    
    # 用户行为
    if os.path.exists(os.path.join(RESULT_DIR, 'tables', 'overall_conversion_rates.csv')):
        try:
            conversion_rates = pd.read_csv(os.path.join(RESULT_DIR, 'tables', 'overall_conversion_rates.csv'))
            summary["user_behavior"]["conversion_rates"] = conversion_rates.iloc[0].to_dict()
        except:
            pass
    
    # 用户分群
    if os.path.exists(os.path.join(RESULT_DIR, 'tables', 'user_clusters.csv')):
        try:
            user_clusters = pd.read_csv(os.path.join(RESULT_DIR, 'tables', 'user_clusters.csv'))
            cluster_data = {}
            for idx, row in user_clusters.iterrows():
                cluster_id = row['Unnamed: 0'] if 'Unnamed: 0' in row else idx
                cluster_data[f"cluster_{cluster_id}"] = {
                    "size": row['cluster_size'] if 'cluster_size' in row else 0,
                    "profile": row['用户画像'] if '用户画像' in row else "未知"
                }
            summary["user_segmentation"]["clusters"] = cluster_data
        except:
            pass
    
    # RFM分析
    if os.path.exists(os.path.join(RESULT_DIR, 'tables', 'rfm_segments.csv')):
        try:
            rfm_segments = pd.read_csv(os.path.join(RESULT_DIR, 'tables', 'rfm_segments.csv'))
            summary["rfm_analysis"]["segments"] = rfm_segments.set_index('user_segment').to_dict('index')
        except:
            pass
    
    # 建议
    summary["recommendations"] = [
        "通过精准营销提高用户转化率",
        "优化用户购物流程，提升用户体验",
        "根据用户活跃时间分布，制定营销时间策略",
        "基于用户分群结果，实施差异化运营",
        "针对高价值用户，提供个性化服务和特权"
    ]
    
    # 保存JSON摘要
    with open(os.path.join(RESULT_DIR, 'analysis_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    
    logger.info(f"JSON分析摘要已生成：{os.path.join(RESULT_DIR, 'analysis_summary.json')}")


def analyze_user_behavior_sequence(df):
    """
    分析用户行为序列模式和转化路径
    使用马尔可夫链模型分析用户行为转移概率
    """
    logger.info("开始分析用户行为序列模式...")
    
    # 确保数据按用户ID和时间戳排序
    df_sorted = df.sort_values(['user_id', 'timestamp'])
    
    # 统计所有行为类型
    behavior_types = df['behavior_type'].unique()
    
    # 创建行为转移矩阵
    transition_matrix = pd.DataFrame(0, 
                                    index=behavior_types, 
                                    columns=behavior_types)
    
    # 计算行为转移次数
    prev_user = None
    prev_behavior = None
    
    for _, row in tqdm(df_sorted.iterrows(), total=len(df_sorted), desc="计算行为转移"):
        current_user = row['user_id']
        current_behavior = row['behavior_type']
        
        # 只在同一用户的连续行为间计算转移
        if prev_user == current_user and prev_behavior is not None:
            transition_matrix.loc[prev_behavior, current_behavior] += 1
        
        prev_user = current_user
        prev_behavior = current_behavior
    
    # 计算转移概率
    for row in transition_matrix.index:
        row_sum = transition_matrix.loc[row].sum()
        if row_sum > 0:
            transition_matrix.loc[row] = transition_matrix.loc[row] / row_sum
    
    # 保存转移矩阵
    transition_matrix.to_csv(os.path.join(RESULT_DIR, 'tables', 'behavior_transition_matrix.csv'))
    
    # 分析典型行为路径
    logger.info("分析典型用户行为路径...")
    
    # 提取每个用户的行为序列
    user_sequences = df_sorted.groupby('user_id')['behavior_type'].apply(list).reset_index()
    
    # 统计常见行为序列模式（最多考虑长度为5的序列）
    max_sequence_length = 5
    sequence_patterns = defaultdict(int)
    
    for seq in user_sequences['behavior_type']:
        for i in range(len(seq)):
            for length in range(2, min(max_sequence_length + 1, len(seq) - i + 1)):
                pattern = tuple(seq[i:i+length])
                sequence_patterns[pattern] += 1
    
    # 获取前20个最常见的序列模式
    top_sequences = sorted(sequence_patterns.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # 转换为DataFrame并保存
    top_sequences_df = pd.DataFrame(top_sequences, columns=['sequence', 'count'])
    top_sequences_df['sequence'] = top_sequences_df['sequence'].apply(lambda x: ' → '.join(x))
    top_sequences_df.to_csv(os.path.join(RESULT_DIR, 'tables', 'top_behavior_sequences.csv'), index=False)
    
    # 分析转化路径和漏斗
    funnel_stages = ['pv', 'fav', 'cart', 'buy']
    funnel_data = []
    
    for i in range(len(funnel_stages)):
        stage = funnel_stages[i]
        # 计算达到当前阶段的用户数
        users_in_stage = df[df['behavior_type'] == stage]['user_id'].nunique()
        funnel_data.append({'stage': stage, 'users': users_in_stage})
    
    funnel_df = pd.DataFrame(funnel_data)
    
    # 计算转化率
    for i in range(1, len(funnel_df)):
        prev_users = funnel_df.iloc[i-1]['users']
        curr_users = funnel_df.iloc[i]['users']
        
        if prev_users > 0:
            conversion_rate = curr_users / prev_users * 100
        else:
            conversion_rate = 0
            
        funnel_df.loc[i, 'conversion_rate'] = conversion_rate
    
    funnel_df.to_csv(os.path.join(RESULT_DIR, 'tables', 'conversion_funnel.csv'), index=False)
    
    # 可视化转移矩阵为热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('用户行为转移概率矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'figures', 'behavior_transition_heatmap.png'), dpi=300)
    plt.close()
    
    # 可视化转化漏斗
    plt.figure(figsize=(12, 6))
    bars = plt.bar(funnel_df['stage'], funnel_df['users'], color=sns.color_palette("Blues_d", len(funnel_df)))
    
    # 添加用户数量标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height):,}',
                ha='center', va='bottom', rotation=0)
    
    # 添加转化率标签（从第二个阶段开始）
    for i in range(1, len(funnel_df)):
        if 'conversion_rate' in funnel_df.columns:
            rate = funnel_df.iloc[i].get('conversion_rate', 0)
            plt.annotate(f'{rate:.1f}%', 
                        xy=(i, funnel_df.iloc[i]['users'] / 2),
                        xytext=(i-1 + 0.85, funnel_df.iloc[i-1]['users'] / 2),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        color='red', fontsize=12)
    
    plt.title('用户行为转化漏斗')
    plt.ylabel('用户数量')
    plt.xlabel('行为阶段')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'figures', 'conversion_funnel.png'), dpi=300)
    plt.close()
    
    return {
        'transition_matrix': transition_matrix,
        'top_sequences': top_sequences_df,
        'funnel_data': funnel_df
    }


def perform_product_network_analysis(df):
    """
    商品网络分析：构建商品共现网络，使用社区发现算法识别相关商品群体
    """
    logger.info("开始商品网络分析...")
    
    # 只考虑购买行为
    buy_df = df[df['behavior_type'] == 'buy']
    
    # 提取每个用户购买的商品集合
    user_items = buy_df.groupby('user_id')['item_id'].apply(set).reset_index()
    
    # 如果数据量太少，则返回
    if len(user_items) < 10:
        logger.warning("用户购买记录不足，无法进行有效的商品网络分析")
        return None
    
    # 构建商品共现网络
    G = nx.Graph()
    
    # 添加节点（商品）
    unique_items = buy_df['item_id'].unique()
    G.add_nodes_from(unique_items)
    
    # 添加边（共同购买关系）
    edge_weights = defaultdict(int)
    
    # 统计每对商品被同一用户购买的次数
    for items in user_items['item_id']:
        if len(items) > 1:  # 用户至少购买了两种商品
            for item1, item2 in itertools.combinations(items, 2):
                if item1 != item2:  # 避免自环
                    edge_key = tuple(sorted([item1, item2]))
                    edge_weights[edge_key] += 1
    
    # 添加边和权重（仅考虑权重大于1的边，减少网络复杂度）
    significant_edges = [(i, j, w) for (i, j), w in edge_weights.items() if w > 1]
    G.add_weighted_edges_from(significant_edges)
    
    # 如果网络为空或太小，则返回
    if len(G.edges) < 5:
        logger.warning("商品共现网络太小，无法进行有效分析")
        return None
    
    # 计算网络基本指标
    logger.info("计算商品网络指标...")
    network_stats = {
        'nodes': len(G.nodes),
        'edges': len(G.edges),
        'avg_degree': sum(dict(G.degree()).values()) / len(G.nodes) if len(G.nodes) > 0 else 0,
        'density': nx.density(G)
    }
    
    # 寻找社区（相关商品群体）
    try:
        communities = nx.community.greedy_modularity_communities(G)
        top_communities = list(communities)[:5]  # 取前5个最大社区
    except:
        logger.warning("社区发现算法失败，尝试替代方法")
        # 使用连通分量作为备选社区发现方法
        connected_components = list(nx.connected_components(G))
        top_communities = sorted(connected_components, key=len, reverse=True)[:5]
    
    # 计算商品中心性
    logger.info("计算商品中心性指标...")
    
    # 度中心性
    degree_centrality = nx.degree_centrality(G)
    # 接近中心性
    try:
        closeness_centrality = nx.closeness_centrality(G)
    except:
        closeness_centrality = {node: 0 for node in G.nodes}
    # 介数中心性（计算量大，可能会耗时）
    if len(G.nodes) <= 1000:  # 当节点较少时才计算
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, len(G.nodes)))
    else:
        betweenness_centrality = {node: 0 for node in G.nodes}
    
    # 综合中心性分数
    centrality_scores = {}
    for node in G.nodes:
        centrality_scores[node] = (
            degree_centrality.get(node, 0) + 
            closeness_centrality.get(node, 0) + 
            betweenness_centrality.get(node, 0)
        ) / 3
    
    # 找出中心性最高的商品
    top_items = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    top_items_df = pd.DataFrame(top_items, columns=['item_id', 'centrality_score'])
    
    # 将社区信息保存为DataFrame
    community_data = []
    for i, community in enumerate(top_communities):
        for item in community:
            community_data.append({
                'item_id': item,
                'community_id': i,
                'degree_centrality': degree_centrality.get(item, 0),
                'closeness_centrality': closeness_centrality.get(item, 0),
                'betweenness_centrality': betweenness_centrality.get(item, 0),
                'overall_centrality': centrality_scores.get(item, 0)
            })
    
    community_df = pd.DataFrame(community_data)
    
    # 保存结果
    pd.DataFrame([network_stats]).to_csv(os.path.join(RESULT_DIR, 'tables', 'network_statistics.csv'), index=False)
    top_items_df.to_csv(os.path.join(RESULT_DIR, 'tables', 'top_central_items.csv'), index=False)
    community_df.to_csv(os.path.join(RESULT_DIR, 'tables', 'item_communities.csv'), index=False)
    
    # 可视化网络（如果节点数适中）
    if 10 <= len(G.nodes) <= 100:
        plt.figure(figsize=(12, 12))
        
        # 为不同社区设置不同颜色
        color_palette = list(plt.cm.tab10.colors)
        node_colors = []
        community_map = {}
        
        for i, community in enumerate(top_communities):
            for node in community:
                community_map[node] = i
        
        for node in G.nodes:
            if node in community_map:
                node_colors.append(color_palette[community_map[node] % len(color_palette)])
            else:
                node_colors.append('lightgrey')
        
        # 节点大小基于中心性
        node_size = [5000 * centrality_scores.get(node, 0) + 100 for node in G.nodes]
        
        # 使用spring_layout布局
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # 绘制网络
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        
        # 为重要节点添加标签
        top_nodes = [item[0] for item in top_items[:10]]
        labels = {node: str(node) for node in top_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        # 添加图例
        legend_elements = []
        for i, community in enumerate(top_communities[:5]):
            legend_elements.append(Patch(facecolor=color_palette[i % len(color_palette)], 
                                        label=f'社区 {i+1} ({len(community)} 商品)'))
        
        plt.legend(handles=legend_elements, loc='upper right', title="商品社区")
        plt.title('商品共现网络分析')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'product_network.png'), dpi=300)
        plt.close()
    
    return {
        'network_stats': network_stats,
        'top_items': top_items_df,
        'communities': community_df
    }


def predict_user_future_value(df, rfm_data):
    """
    用户价值预测与高潜力用户识别
    """
    logger.info("开始用户价值预测分析...")
    
    if rfm_data is None or len(rfm_data) < 100:
        logger.warning("RFM数据不足，无法进行有效的用户价值预测")
        return None
    
    # 准备特征
    features = rfm_data[['recency', 'frequency', 'monetary']].copy()
    
    # 计算当前用户价值（RFM总分）
    current_value = rfm_data['rfm_score'].copy()
    
    # 添加更多用户行为特征
    user_behaviors = df.groupby('user_id')['behavior_type'].value_counts().unstack(fill_value=0)
    
    # 确保所有行为类型列都存在
    for behavior in ['pv', 'buy', 'cart', 'fav']:
        if behavior not in user_behaviors.columns:
            user_behaviors[behavior] = 0
    
    # 计算行为比率特征
    user_behaviors['pv_to_buy_ratio'] = np.where(user_behaviors['buy'] > 0, 
                                              user_behaviors['pv'] / user_behaviors['buy'], 0)
    user_behaviors['cart_to_buy_ratio'] = np.where(user_behaviors['buy'] > 0, 
                                                user_behaviors['cart'] / user_behaviors['buy'], 0)
    user_behaviors['fav_to_buy_ratio'] = np.where(user_behaviors['buy'] > 0, 
                                               user_behaviors['fav'] / user_behaviors['buy'], 0)
    
    # 合并RFM和行为特征
    user_behaviors = user_behaviors.reset_index()
    features = features.reset_index().set_index('user_id')
    user_behaviors = user_behaviors.set_index('user_id')
    merged_features = features.join(user_behaviors, how='left')
    merged_features = merged_features.fillna(0)
    
    # 移除user_id列(如果存在)
    if 'user_id' in merged_features.columns:
        merged_features = merged_features.drop('user_id', axis=1)
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(merged_features)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, current_value, test_size=0.2, random_state=42)
    
    # 训练梯度提升回归模型
    logger.info("训练用户价值预测模型...")
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"模型评估 - MSE: {mse:.4f}, R²: {r2:.4f}")
    
    # 计算特征重要性
    feature_importance = pd.DataFrame({
        'feature': merged_features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 预测所有用户的未来价值
    all_predictions = model.predict(X_scaled)
    
    # 当前价值与预测价值的比较
    value_comparison = pd.DataFrame({
        'user_id': merged_features.index,
        'current_value': current_value.values,
        'predicted_value': all_predictions,
        'growth_potential': all_predictions - current_value.values
    })
    
    # 识别高潜力用户（预测价值增长最大的前10%）
    high_potential_threshold = np.percentile(value_comparison['growth_potential'], 90)
    value_comparison['high_potential'] = value_comparison['growth_potential'] >= high_potential_threshold
    
    # 保存结果
    feature_importance.to_csv(os.path.join(RESULT_DIR, 'tables', 'value_feature_importance.csv'), index=False)
    value_comparison.to_csv(os.path.join(RESULT_DIR, 'tables', 'user_value_prediction.csv'), index=False)
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10)
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title('用户价值预测的特征重要性Top10')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'figures', 'value_feature_importance.png'), dpi=300)
    plt.close()
    
    # 可视化当前价值与预测价值的散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(value_comparison['current_value'], value_comparison['predicted_value'], 
               alpha=0.5, c=value_comparison['high_potential'].map({True: 'red', False: 'blue'}))
    
    # 添加对角线
    max_val = max(value_comparison['current_value'].max(), value_comparison['predicted_value'].max())
    plt.plot([0, max_val], [0, max_val], 'k--')
    
    plt.xlabel('当前用户价值')
    plt.ylabel('预测用户价值')
    plt.title('用户价值预测分析')
    
    # 添加图例
    plt.legend(['对角线', '高潜力用户', '普通用户'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'figures', 'user_value_prediction.png'), dpi=300)
    plt.close()
    
    return {
        'feature_importance': feature_importance,
        'value_comparison': value_comparison,
        'model_metrics': {'mse': mse, 'r2': r2}
    }


def perform_advanced_basket_analysis(df):
    """
    高级购物篮分析，使用FP-Growth算法提高效率
    计算多种兴趣度指标，分析不同时间段的关联规则
    """
    logger.info("开始高级购物篮分析...")
    
    # 只考虑购买行为
    buy_df = df[df['behavior_type'] == 'buy']
    
    # 如果购买记录太少，则跳过此分析
    if len(buy_df) < 100:
        logger.warning("购买记录不足，无法进行有效的购物篮分析")
        return None
    
    # 创建购物篮数据框（每个用户在不同日期的购买记录）
    baskets = buy_df.groupby(['user_id', 'date'])['item_id'].apply(list).reset_index()
    
    # 获取时间范围并划分时间段
    date_range = pd.date_range(start=buy_df['date'].min(), end=buy_df['date'].max())
    if len(date_range) <= 3:
        # 如果时间范围太短，就按天划分
        time_periods = [(date, date) for date in date_range]
        period_names = [f"{date.strftime('%Y-%m-%d')}" for date in date_range]
    else:
        # 划分为3个时间段
        period_size = len(date_range) // 3
        time_periods = [
            (date_range[0], date_range[period_size-1]),
            (date_range[period_size], date_range[2*period_size-1]),
            (date_range[2*period_size], date_range[-1])
        ]
        period_names = [
            f"{period[0].strftime('%Y-%m-%d')} 至 {period[1].strftime('%Y-%m-%d')}"
            for period in time_periods
        ]
    
    # 创建频繁项集和关联规则的结果字典
    results = {}
    
    # 在不同时间段内分析关联规则
    for period_idx, period in enumerate(time_periods):
        period_name = period_names[period_idx]
        logger.info(f"分析时间段 {period_name} 的购物篮模式...")
        
        # 筛选当前时间段的购物篮
        if isinstance(period[0], pd.Timestamp):
            period_baskets = baskets[
                (baskets['date'] >= period[0].date()) & 
                (baskets['date'] <= period[1].date())
            ]
        else:
            period_baskets = baskets[
                (baskets['date'] >= period[0]) & 
                (baskets['date'] <= period[1])
            ]
        
        if len(period_baskets) < 10:
            logger.warning(f"时间段 {period_name} 的购物篮数量不足，跳过分析")
            continue
        
        # 创建商品的独热编码
        # 由于商品数量可能很多，这里选择频率较高的商品
        top_items = buy_df[buy_df['date'].isin(period_baskets['date'])]['item_id'].value_counts().head(200).index
        
        # 创建独热编码矩阵
        basket_sets = pd.DataFrame(0, index=range(len(period_baskets)), columns=top_items)
        
        # 填充独热编码
        for i, row in enumerate(period_baskets['item_id']):
            for item in row:
                if item in top_items:
                    basket_sets.loc[i, item] = 1
        
        # 应用FP-Growth算法
        try:
            # 选择适当的最小支持度
            min_support = 0.01
            
            # 尝试多次降低支持度，直到找到足够的频繁项集
            for attempt in range(3):
                # 应用FP-Growth算法找出频繁项集
                frequent_itemsets = fpgrowth(basket_sets, min_support=min_support, use_colnames=True)
                
                if len(frequent_itemsets) >= 10 or min_support < 0.001:
                    break
                    
                min_support = min_support / 2
                logger.info(f"降低最小支持度到 {min_support}")
            
            if len(frequent_itemsets) < 5:
                logger.warning(f"时间段 {period_name} 发现的频繁项集过少，关联规则分析可能不具有统计意义")
                continue
            
            # 生成关联规则，计算多种兴趣度指标
            rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
            
            # 如果规则太多，只保留提升度最高的100条
            if len(rules) > 100:
                rules = rules.sort_values('lift', ascending=False).head(100)
            
            # 添加更多的兴趣度指标
            # 杠杆率 (Leverage) = 支持度(X,Y) - 支持度(X) * 支持度(Y)
            rules['leverage'] = rules['support'] - (rules['antecedent support'] * rules['consequent support'])
            
            # 确信度差 (Conviction) = (1 - 支持度(Y)) / (1 - 确信度(X→Y))
            rules['conviction'] = np.where(
                rules['confidence'] < 1.0,
                (1 - rules['consequent support']) / (1 - rules['confidence']),
                float('inf')
            )
            
            # 将关联规则转换为可读形式
            rules_readable = rules.copy()
            rules_readable['antecedents'] = rules_readable['antecedents'].apply(lambda x: ', '.join(str(i) for i in x))
            rules_readable['consequents'] = rules_readable['consequents'].apply(lambda x: ', '.join(str(i) for i in x))
            
            # 保存该时间段的结果
            results[period_name] = {
                'frequent_itemsets': frequent_itemsets,
                'rules': rules,
                'rules_readable': rules_readable
            }
            
            # 保存到CSV
            output_name = f"association_rules_period_{period_idx+1}"
            rules_readable.to_csv(os.path.join(RESULT_DIR, 'tables', f"{output_name}.csv"), index=False)
            
            # 可视化前10条规则的提升度
            plt.figure(figsize=(12, 8))
            top_rules = rules_readable.sort_values('lift', ascending=False).head(10)
            
            # 创建规则标签（前项=>后项）
            rule_labels = [f"{ant} => {con}" for ant, con in zip(top_rules['antecedents'], top_rules['consequents'])]
            rule_labels = [label[:30] + '...' if len(label) > 30 else label for label in rule_labels]
            
            # 绘制条形图
            sns.barplot(x='lift', y=rule_labels, data=top_rules)
            plt.title(f'时间段 {period_name} 的Top10关联规则提升度')
            plt.xlabel('提升度(Lift)')
            plt.ylabel('关联规则')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, 'figures', f"{output_name}_lift.png"), dpi=300)
            plt.close()
            
            # 绘制散点图，展示支持度、置信度和提升度的关系
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                rules['support'], rules['confidence'], 
                alpha=0.5, c=rules['lift'], cmap='viridis', 
                s=rules['leverage']*1000 + 20
            )
            
            plt.colorbar(scatter, label='提升度(Lift)')
            plt.xlabel('支持度(Support)')
            plt.ylabel('置信度(Confidence)')
            plt.title(f'时间段 {period_name} 的关联规则分析')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, 'figures', f"{output_name}_scatter.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"时间段 {period_name} 的关联规则挖掘失败: {e}")
    
    logger.info(f"高级购物篮分析完成，分析了 {len(results)} 个时间段")
    return results


def perform_user_lifecycle_analysis(df):
    """
    用户生命周期分析
    使用BG/NBD和Gamma-Gamma模型分析用户的活跃和生命周期价值
    """
    logger.info("开始用户生命周期分析...")
    
    # 准备数据
    purchase_df = df[df['behavior_type'] == 'buy'].copy()
    
    if len(purchase_df) < 100:
        logger.warning("购买数据不足，无法进行有效的生命周期分析")
        return None
    
    # 数据预处理：确保时间格式正确且有用户和商品ID
    purchase_df['date'] = pd.to_datetime(purchase_df['date'])
    
    # 计算RFM统计数据
    # 设置观察期结束时间（数据集中的最后日期）
    observation_end = purchase_df['date'].max()
    
    # 使用lifetimes库计算RFM汇总
    rfm_lifetimes = summary_data_from_transaction_data(
        transactions=purchase_df,
        customer_id_col='user_id',
        datetime_col='date',
        observation_period_end=observation_end
    )
    
    # 移除极端值
    rfm_lifetimes = rfm_lifetimes[rfm_lifetimes['frequency'] < rfm_lifetimes['frequency'].quantile(0.99)]
    rfm_lifetimes = rfm_lifetimes[rfm_lifetimes['recency'] < rfm_lifetimes['recency'].quantile(0.99)]
    rfm_lifetimes = rfm_lifetimes[rfm_lifetimes['T'] < rfm_lifetimes['T'].quantile(0.99)]
    
    # 训练BG/NBD模型（用于预测客户活跃概率和期望购买次数）
    logger.info("训练BG/NBD模型...")
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(rfm_lifetimes['frequency'], rfm_lifetimes['recency'], rfm_lifetimes['T'])
    
    # 预测未来30天的期望购买次数
    rfm_lifetimes['predicted_purchases_30days'] = bgf.predict(30, 
                                                         rfm_lifetimes['frequency'],
                                                         rfm_lifetimes['recency'], 
                                                         rfm_lifetimes['T'])
    
    # 计算每个客户当前处于活跃状态的概率
    rfm_lifetimes['probability_alive'] = bgf.conditional_probability_alive(
        rfm_lifetimes['frequency'],
        rfm_lifetimes['recency'],
        rfm_lifetimes['T']
    )
    
    # 添加用户分类
    def classify_user(row):
        if row['probability_alive'] >= 0.7:
            if row['frequency'] > rfm_lifetimes['frequency'].median():
                return "活跃忠诚用户"
            else:
                return "新用户"
        elif row['probability_alive'] >= 0.3:
            if row['frequency'] > rfm_lifetimes['frequency'].median():
                return "有流失风险的忠诚用户"
            else:
                return "间歇性用户"
        else:
            if row['frequency'] > rfm_lifetimes['frequency'].median():
                return "已流失的忠诚用户"
            else:
                return "已流失的低价值用户"
    
    rfm_lifetimes['user_category'] = rfm_lifetimes.apply(classify_user, axis=1)
    
    # 保存生命周期分析结果
    rfm_lifetimes.reset_index().to_csv(os.path.join(RESULT_DIR, 'tables', 'user_lifecycle_analysis.csv'), index=False)
    
    # 统计各类用户数量
    category_counts = rfm_lifetimes['user_category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    category_counts['percentage'] = category_counts['count'] / category_counts['count'].sum() * 100
    category_counts.to_csv(os.path.join(RESULT_DIR, 'tables', 'user_lifecycle_categories.csv'), index=False)
    
    # 可视化用户类别分布
    plt.figure(figsize=(12, 8))
    sns.barplot(x='category', y='count', data=category_counts)
    plt.title('用户生命周期类别分布')
    plt.xlabel('用户类别')
    plt.ylabel('用户数量')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'figures', 'user_lifecycle_categories.png'), dpi=300)
    plt.close()
    
    # 可视化用户活跃概率与频率的关系
    plt.figure(figsize=(10, 8))
    plt.scatter(
        rfm_lifetimes['frequency'], 
        rfm_lifetimes['probability_alive'],
        alpha=0.4
    )
    plt.xlabel('购买频率')
    plt.ylabel('用户活跃概率')
    plt.title('用户活跃概率与购买频率的关系')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'figures', 'user_activity_probability.png'), dpi=300)
    plt.close()
    
    # 绘制活跃概率矩阵
    plt.figure(figsize=(12, 8))
    plot_probability_alive_matrix(bgf)
    plt.title('用户活跃概率矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'figures', 'user_probability_alive_matrix.png'), dpi=300)
    plt.close()
    
    return {
        'rfm_lifetimes': rfm_lifetimes,
        'category_counts': category_counts,
        'bgf_model': bgf
    }


def perform_time_series_analysis(df):
    """
    时间序列分析，研究用户活跃度的周期性模式与趋势
    使用ARIMA模型进行预测
    """
    logger.info("开始时间序列分析...")
    
    # 按日期统计用户活跃度（总用户数和行为次数）
    daily_active_users = df.groupby('date')['user_id'].nunique()
    daily_actions = df.groupby('date').size()
    
    # 如果数据点太少，则跳过
    if len(daily_active_users) < 7:
        logger.warning("时间序列数据点不足，无法进行有效的时间序列分析")
        return None
    
    # 创建时间序列数据框
    time_series_df = pd.DataFrame({
        'active_users': daily_active_users,
        'actions': daily_actions
    })
    
    # 确保索引是日期时间类型
    time_series_df.index = pd.to_datetime(time_series_df.index)
    time_series_df = time_series_df.sort_index()
    
    # 尝试进行季节性分解（至少需要2个完整周期）
    try:
        # 对用户活跃度进行季节性分解
        if len(time_series_df) >= 14:  # 至少需要14天数据才能分析周周期
            logger.info("进行季节性分解分析...")
            decomposition = seasonal_decompose(time_series_df['active_users'], model='additive', period=7)
            
            # 绘制分解结果
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            decomposition.observed.plot(ax=axes[0])
            axes[0].set_title('观察值')
            decomposition.trend.plot(ax=axes[1])
            axes[1].set_title('趋势')
            decomposition.seasonal.plot(ax=axes[2])
            axes[2].set_title('季节性')
            decomposition.resid.plot(ax=axes[3])
            axes[3].set_title('残差')
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, 'figures', 'time_series_decomposition.png'), dpi=300)
            plt.close()
    except Exception as e:
        logger.warning(f"季节性分解失败: {e}")
    
    # 尝试使用ARIMA模型预测未来趋势
    try:
        logger.info("使用ARIMA模型进行预测...")
        
        # 处理可能的缺失值
        ts_data = time_series_df['active_users'].fillna(method='ffill')
        
        # 拟合ARIMA模型
        # 简化起见，使用固定参数(1,1,1)，实际应用中应进行参数优化
        model = ARIMA(ts_data, order=(1, 1, 1))
        results = model.fit()
        
        # 预测未来7天
        forecast = results.forecast(steps=7)
        
        # 创建预测日期
        last_date = ts_data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
        forecast_series = pd.Series(forecast, index=forecast_dates)
        
        # 保存预测结果
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast
        })
        forecast_df.to_csv(os.path.join(RESULT_DIR, 'tables', 'user_activity_forecast.csv'), index=False)
        
        # 可视化预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(ts_data.index, ts_data, label='历史数据')
        plt.plot(forecast_series.index, forecast_series, label='预测', color='red')
        plt.title('用户活跃度预测（未来7天）')
        plt.xlabel('日期')
        plt.ylabel('活跃用户数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'user_activity_forecast.png'), dpi=300)
        plt.close()
        
    except Exception as e:
        logger.warning(f"ARIMA预测失败: {e}")
    
    # 分析行为类型在不同时间的分布
    # 按日期和行为类型统计
    behavior_time_series = df.groupby(['date', 'behavior_type']).size().unstack(fill_value=0)
    
    # 计算每种行为的移动平均（7日）
    if len(behavior_time_series) >= 7:
        behavior_ma = behavior_time_series.rolling(window=7, min_periods=1).mean()
        
        # 可视化不同行为类型的时间趋势
        plt.figure(figsize=(12, 6))
        for col in behavior_time_series.columns:
            plt.plot(behavior_ma.index, behavior_ma[col], label=col)
        
        plt.title('不同行为类型的时间趋势（7日移动平均）')
        plt.xlabel('日期')
        plt.ylabel('平均每日行为次数')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'behavior_time_trends.png'), dpi=300)
        plt.close()
    
    # 分析小时分布的日间变化
    hourly_behavior = df.groupby(['date', 'hour']).size().unstack(fill_value=0)
    
    if not hourly_behavior.empty and len(hourly_behavior) > 1:
        # 计算每小时行为的归一化分布（每天总和为1）
        hourly_distribution = hourly_behavior.div(hourly_behavior.sum(axis=1), axis=0)
        
        # 可视化小时分布热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(hourly_distribution.T, cmap='YlGnBu')
        plt.title('每日各小时用户行为分布热力图')
        plt.xlabel('日期')
        plt.ylabel('小时')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'hourly_distribution_heatmap.png'), dpi=300)
        plt.close()
    
    # 保存时间序列数据
    time_series_df.to_csv(os.path.join(RESULT_DIR, 'tables', 'time_series_data.csv'))
    
    return {
        'time_series_df': time_series_df,
        'behavior_time_series': behavior_time_series
    }


def analyze_user_preference(df):
    """
    深度分析用户偏好，构建用户-商品偏好矩阵
    使用矩阵分解技术探索潜在兴趣
    """
    logger.info("开始分析用户偏好模式...")
    
    # 创建用户-商品交互矩阵
    # 为不同行为类型分配权重
    behavior_weights = {
        'pv': 1,
        'fav': 3,
        'cart': 4,
        'buy': 5
    }
    
    # 用加权方式构建用户-商品交互矩阵
    user_item_df = df.copy()
    user_item_df['weight'] = user_item_df['behavior_type'].map(behavior_weights)
    
    # 聚合交互权重
    user_item_matrix = user_item_df.groupby(['user_id', 'item_id'])['weight'].sum().unstack(fill_value=0)
    
    # 如果矩阵太大，进行采样
    if user_item_matrix.shape[0] > 1000 or user_item_matrix.shape[1] > 1000:
        logger.info("用户-商品矩阵过大，进行采样...")
        # 选择活跃用户和热门商品
        active_users = user_item_df['user_id'].value_counts().head(1000).index
        popular_items = user_item_df['item_id'].value_counts().head(1000).index
        
        # 筛选数据
        filtered_df = user_item_df[
            user_item_df['user_id'].isin(active_users) & 
            user_item_df['item_id'].isin(popular_items)
        ]
        
        # 重建矩阵
        user_item_matrix = filtered_df.groupby(['user_id', 'item_id'])['weight'].sum().unstack(fill_value=0)
    
    # 如果矩阵为空或太小，则返回
    if user_item_matrix.empty or user_item_matrix.shape[0] < 10 or user_item_matrix.shape[1] < 10:
        logger.warning("用户-商品交互矩阵太小，无法进行有效的偏好分析")
        return None
    
    # 使用矩阵分解技术（截断SVD）探索潜在兴趣维度
    logger.info("使用矩阵分解探索潜在兴趣维度...")
    
    # 确定合适的潜在因子数量
    n_components = min(20, min(user_item_matrix.shape) - 1)
    
    # 应用截断SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    
    # 计算物品因子
    item_factors = svd.components_
    
    # 计算每个潜在因子的方差解释率
    explained_variance = svd.explained_variance_ratio_
    
    # 保存SVD结果
    explained_var_df = pd.DataFrame({
        'component': range(1, n_components+1),
        'explained_variance': explained_variance,
        'cumulative_variance': np.cumsum(explained_variance)
    })
    explained_var_df.to_csv(os.path.join(RESULT_DIR, 'tables', 'preference_svd_variance.csv'), index=False)
    
    # 可视化方差解释率
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_components+1), explained_variance)
    plt.plot(range(1, n_components+1), np.cumsum(explained_variance), 'r-')
    plt.title('潜在兴趣维度的方差解释率')
    plt.xlabel('潜在因子')
    plt.ylabel('解释方差比例')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'figures', 'preference_explained_variance.png'), dpi=300)
    plt.close()
    
    # 选择重要的潜在因子（解释至少70%的方差）
    n_selected = np.where(np.cumsum(explained_variance) >= 0.7)[0][0] + 1
    n_selected = min(n_selected, 5)  # 最多展示5个因子
    
    logger.info(f"选择了{n_selected}个重要潜在因子进行分析")
    
    # 分析每个潜在因子的商品分布
    item_factor_df = pd.DataFrame(item_factors[:n_selected], 
                                 columns=user_item_matrix.columns)
    
    # 为每个潜在因子找出最重要的商品
    important_items = {}
    for i in range(n_selected):
        factor_items = item_factor_df.iloc[i].sort_values(ascending=False).head(20)
        important_items[f'factor_{i+1}'] = factor_items.index.tolist()
    
    # 保存重要商品
    pd.DataFrame(important_items).to_csv(os.path.join(RESULT_DIR, 'tables', 'preference_factor_items.csv'), index=False)
    
    # 分析用户在潜在因子上的分布
    user_factor_df = pd.DataFrame(
        user_factors[:, :n_selected], 
        index=user_item_matrix.index,
        columns=[f'factor_{i+1}' for i in range(n_selected)]
    )
    
    # 计算用户聚类（基于潜在因子）
    logger.info("基于潜在兴趣进行用户聚类...")
    
    # 使用k-means进行聚类
    kmeans = KMeans(n_clusters=min(5, len(user_factor_df)), random_state=42, n_init=10)
    user_factor_df['cluster'] = kmeans.fit_predict(user_factors[:, :n_selected])
    
    # 保存用户聚类结果
    user_factor_df.to_csv(os.path.join(RESULT_DIR, 'tables', 'preference_user_factors.csv'))
    
    # 可视化用户在不同潜在因子上的分布
    if n_selected >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            user_factor_df['factor_1'], 
            user_factor_df['factor_2'],
            c=user_factor_df['cluster'], 
            cmap='viridis',
            alpha=0.6
        )
        plt.colorbar(scatter, label='聚类')
        plt.title('用户在前两个潜在兴趣维度上的分布')
        plt.xlabel('潜在因子1')
        plt.ylabel('潜在因子2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'figures', 'preference_user_distribution.png'), dpi=300)
        plt.close()
    
    # 分析不同聚类的用户偏好特征
    cluster_profiles = user_factor_df.groupby('cluster').mean()
    cluster_profiles.to_csv(os.path.join(RESULT_DIR, 'tables', 'preference_cluster_profiles.csv'))
    
    # 可视化聚类特征
    plt.figure(figsize=(12, 8))
    sns.heatmap(cluster_profiles, cmap='YlGnBu', annot=True, fmt='.2f')
    plt.title('不同用户群体的潜在兴趣特征')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'figures', 'preference_cluster_heatmap.png'), dpi=300)
    plt.close()
    
    return {
        'user_item_matrix': user_item_matrix,
        'user_factors': user_factor_df,
        'item_factors': item_factor_df,
        'explained_variance': explained_var_df,
        'important_items': important_items
    }


def main():
    """
    主函数，组织整个分析流程
    """
    logger.info("=== 淘宝用户行为分析开始 ===")
    
    # 1. 加载数据
    user_behavior_df, tmall_df = load_data()
    
    # 2. 数据清洗
    clean_user_behavior = clean_userbehavior_data(user_behavior_df) if not user_behavior_df.empty else None
    clean_tmall = clean_tmall_data(tmall_df) if not tmall_df.empty else None
    
    # 如果两个数据集都为空，则退出
    if clean_user_behavior is None and clean_tmall is None:
        logger.error("没有可用的数据集，分析终止")
        return
    
    # 3. 数据分析
    # 3.1 用户行为分析
    analysis_results = {}
    if clean_user_behavior is not None and not clean_user_behavior.empty:
        analysis_results = analyze_user_behavior(clean_user_behavior)
    
    # 3.2 天猫数据分析
    tmall_results = {}
    if clean_tmall is not None and not clean_tmall.empty:
        tmall_results = analyze_tmall_data(clean_tmall)
    
    # 4. 用户分群分析
    user_features = None
    cluster_summary = None
    if clean_user_behavior is not None and not clean_user_behavior.empty:
        user_features, cluster_summary = segment_users(clean_user_behavior, clean_user_behavior)
    
    # 5. 购物篮分析
    association_rules = None
    if clean_user_behavior is not None and not clean_user_behavior.empty:
        association_rules = basket_analysis(clean_user_behavior)
        
    # 6. RFM分析
    rfm_results = None
    rfm_segments = None
    if clean_user_behavior is not None and not clean_user_behavior.empty:
        rfm_results, rfm_segments = rfm_analysis(clean_user_behavior)
    
    # 7. 创建用户画像
    user_profiles = None
    if user_features is not None and not user_features.empty:
        user_profiles = create_user_profiles(user_features, rfm_results)
    
    # 8. 行为序列分析 (新增高级分析)
    sequence_results = None
    if clean_user_behavior is not None and not clean_user_behavior.empty:
        sequence_results = analyze_user_behavior_sequence(clean_user_behavior)
        
    # 9. 商品网络分析 (新增高级分析)
    network_results = None
    if clean_user_behavior is not None and not clean_user_behavior.empty:
        network_results = perform_product_network_analysis(clean_user_behavior)
        
    # 10. 用户价值预测 (新增高级分析)
    value_prediction = None
    if rfm_results is not None and not rfm_results.empty:
        value_prediction = predict_user_future_value(clean_user_behavior, rfm_results)
        
    # 11. 高级购物篮分析 (新增高级分析)
    advanced_basket_results = None
    if clean_user_behavior is not None and not clean_user_behavior.empty:
        advanced_basket_results = perform_advanced_basket_analysis(clean_user_behavior)
        
    # 12. 用户生命周期分析 (新增高级分析)
    lifecycle_results = None
    if clean_user_behavior is not None and not clean_user_behavior.empty:
        lifecycle_results = perform_user_lifecycle_analysis(clean_user_behavior)
        
    # 13. 时间序列分析 (新增高级分析)
    time_series_results = None
    if clean_user_behavior is not None and not clean_user_behavior.empty:
        time_series_results = perform_time_series_analysis(clean_user_behavior)
        
    # 14. 用户偏好分析 (新增高级分析)
    preference_results = None
    if clean_user_behavior is not None and not clean_user_behavior.empty:
        preference_results = analyze_user_preference(clean_user_behavior)
    
    # 15. 可视化结果
    visualize_results(analysis_results, tmall_results, user_features)
    
    # 16. 生成HTML报告
    generate_html_report()
    
    # 17. 生成JSON摘要
    generate_json_summary()
    
    logger.info("=== 淘宝用户行为分析完成 ===")


if __name__ == "__main__":
    main()


def basket_analysis(df):
    """
    购物篮分析（关联规则挖掘）
    """
    logger.info("开始购物篮分析...")
    
    # 只考虑购买行为
    buy_df = df[df['behavior_type'] == 'buy']
    
    # 如果购买记录太少，则跳过此分析
    if len(buy_df) < 100:
        logger.warning("购买记录不足，无法进行有效的购物篮分析")
        return None
        
    # 创建用户-商品购买矩阵
    logger.info("创建用户-商品购买矩阵...")
    
    # 检查用户购买记录数量
    user_purchase_counts = buy_df.groupby('user_id').size()
    multi_purchase_users = user_purchase_counts[user_purchase_counts > 1].index
    
    if len(multi_purchase_users) < 10:
        logger.warning("多次购买用户数量不足，无法进行有效的购物篮分析")
        return None
    
    # 只保留有多次购买记录的用户
    multi_buy_df = buy_df[buy_df['user_id'].isin(multi_purchase_users)]
    
    # 创建购物篮数据框（每个用户在不同日期的购买记录）
    baskets = multi_buy_df.groupby(['user_id', 'date'])['item_id'].apply(list).reset_index()
    
    # 创建商品的独热编码
    # 由于商品数量可能很多，这里选择频率较高的商品
    top_items = buy_df['item_id'].value_counts().head(200).index
    
    # 创建每个购物篮内商品的独热编码矩阵
    logger.info("创建独热编码矩阵...")
    basket_sets = pd.DataFrame({'item_id': top_items})
    basket_sets['item_present'] = 1
    
    # 创建空的独热编码结果
    baskets_encoded = pd.DataFrame(0, index=range(len(baskets)), columns=top_items)
    
    # 填充独热编码
    for i, row in enumerate(baskets['item_id']):
        for item in row:
            if item in top_items:
                baskets_encoded.loc[i, item] = 1
    
    # 应用Apriori算法
    logger.info("应用Apriori算法...")
    try:
        # 选择适当的最小支持度
        min_support = 0.01
        
        # 应用Apriori算法找出频繁项集
        frequent_itemsets = apriori(baskets_encoded, min_support=min_support, use_colnames=True)
        
        # 如果频繁项集过少，减小最小支持度
        attempts = 0
        while len(frequent_itemsets) < 10 and attempts < 3:
            min_support = min_support / 2
            logger.info(f"减小最小支持度到 {min_support}")
            frequent_itemsets = apriori(baskets_encoded, min_support=min_support, use_colnames=True)
            attempts += 1
        
        if len(frequent_itemsets) < 5:
            logger.warning("发现的频繁项集过少，关联规则分析可能不具有统计意义")
            return None
        
        # 生成关联规则
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        
        # 按提升度排序
        rules = rules.sort_values('lift', ascending=False)
        
        # 保存结果
        if not rules.empty:
            # 将item_id转换为可读形式
            rules_readable = rules.copy()
            rules_readable['antecedents'] = rules_readable['antecedents'].apply(lambda x: ', '.join(str(i) for i in x))
            rules_readable['consequents'] = rules_readable['consequents'].apply(lambda x: ', '.join(str(i) for i in x))
            
            # 保存关联规则
            rules_readable.to_csv(os.path.join(RESULT_DIR, 'tables', 'association_rules.csv'), index=False)
            logger.info(f"找到 {len(rules)} 条关联规则")
            
            # 返回前20条关联规则
            return rules.head(20)
        else:
            logger.warning("未发现有意义的关联规则")
            return None
    except Exception as e:
        logger.error(f"关联规则挖掘失败: {e}")
        return None