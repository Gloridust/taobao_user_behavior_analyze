# Taobao User Behavior Analysis

## Data Sources

- [Taobao User Shopping Behavior Dataset - Alibaba Tianchi - UserBehavior](https://tianchi.aliyun.com/dataset/649)
- [Tmall Recommendation Dataset - Alibaba Tianchi - Tianchi_2014002_Rec_Tmall_Log](https://tianchi.aliyun.com/dataset/140281)
- [Taobao User Shopping Behavior Analysis Project Dataset - tb-behavior.txt](https://gitcode.com/open-source-toolkit/15c09/?utm_source=tools_gitcode&index=bottom&type=card&)

### Taobao User Behavior Dataset Overview

1. Overview  

UserBehavior is a dataset of Taobao user behavior provided by Alibaba for research on implicit feedback recommendation problems.

1. Introduction  

| Filename          | Description                  | Features                             |
|-------------------|------------------------------|--------------------------------------|
| UserBehavior.csv   | Contains all user behavior data | User ID, Item ID, Category ID, Behavior Type, Timestamp |

This dataset contains all behaviors (including clicks, purchases, adding to cart, and favorites) of approximately one million random users who were active between November 25 and December 3, 2017. The dataset is organized similarly to MovieLens-20M, with each line representing a user behavior record consisting of user ID, item ID, category ID, behavior type, and timestamp, separated by commas. The detailed description of each column is as follows:

| Column Name    | Description                                   |
|----------------|-----------------------------------------------|
| User ID        | Integer type, serialized user ID              |
| Item ID        | Integer type, serialized item ID              |
| Category ID    | Integer type, serialized category ID of the item |
| Behavior Type  | String, enumeration type, including ('pv', 'buy', 'cart', 'fav') |
| Timestamp      | Timestamp when the behavior occurred          |

There are four types of user behaviors:

| Behavior Type | Description                           |
|---------------|---------------------------------------|
| pv            | Page view of item details, equivalent to clicks |
| buy           | Item purchase                        |
| cart          | Add item to shopping cart            |
| fav           | Favorite item                        |

Some information about the dataset size:

| Dimension        | Quantity    |
|------------------|-------------|
| Number of Users  | 987,994     |
| Number of Items  | 4,162,024   |
| Number of Categories | 9,439    |
| Total Behaviors  | 100,150,807 |

### Tianchi_2014002_Rec_Tmall_Log

User behaviors while browsing Tmall reflect their preferences for items. This dataset contains 1,333,729,303 rows, recording interactions between users and items. The features of each row are shown below.

| Column Name | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| Item_id     | An integer between [1, 8133507], representing a unique item.                |
| User_id     | A string like "u9774184", representing a unique user.                      |
| Action      | Behavior type, a string like "click", "collect", "cart", "alipay", representing "click", "add to favorites", "add to shopping cart", and "purchase" respectively. |
| Vtime       | Timestamp of the behavior, a string in the format "yyyy-mm-dd hh:mm:ss".   |
| Typical Research Topics | a) Matrix Completion  b) Ranking                                |

### Taobao User Shopping Behavior Analysis Project Dataset - tb-behavior.txt

Dataset Introduction

This repository provides a resource file named "Taobao User Behavior Dataset", which contains 3,182,257 user behavior records. These data can be used for recommendation systems, data analysis, and other scenarios. The fields in the dataset include:

- `id`: User ID
- `uidagegender`: User age and gender
- `item_id`: Item ID
- `behavior_type`: User behavior type (1. Browse, 2. Favorite, 3. Add to cart, 4. Purchase)
- `item_category`: Item category
- `date`: Date when the behavior occurred
- `province`: User's province

#### Dataset Usage

This dataset can be used for the following analyses and applications:

1. **Recommendation Systems**: Personalized recommendations based on user behavior data.
2. **Data Analysis**: Analysis of user behavior patterns, item popularity, etc.
3. **Statistical Analysis**:
  - **Top 10 Popular Products by Province**: Statistics on the top 10 products with the highest total number of views, favorites, cart additions, and purchases.
  - **Top 10 Purchased Products by Province**: Statistics on the top 10 products with the highest sales.
  - **Top 10 Best-Selling Product Categories by Province**: Statistics on the top 10 product categories with the highest sales.
  - **Male and Female User Count by Province**: Statistics on the number of male and female users in each province (`gender` field: 0 for male, 1 for female, 2 for unknown).

#### Dataset Field Description

- `id`: Unique identifier for the user.
- `uidagegender`: User age and gender information.
- `item_id`: Unique identifier for the item.
- `behavior_type`: User behavior type, with values ranging from 1 to 4, representing browsing, favoriting, adding to cart, and purchasing, respectively.
- `item_category`: Category to which the item belongs.
- `date`: Date when the user behavior occurred.
- `province`: The province where the user is located.

#### Dataset Example

Below is an example record from the dataset:

```
id, uidagegender, item_id, behavior_type, item_category, date, province
1, 25_1, 12345, 4, 101, 2023-01-01, Zhejiang Province
```

## Results - Output Files Explained

This document provides a detailed explanation of all result files generated by the Taobao User Behavior Analysis System and their purposes. All output files are organized in different subdirectories under the `result/` directory by type.

## Data Tables (result/tables/)

### Data Cleaning and Basic Statistics

1. **cleaned_userbehavior.csv**
   - Content: Cleaned Taobao user behavior data
   - Fields included: User ID, Item ID, Category ID, Behavior Type, Timestamp, Date, Hour, Day of Week
   - Purpose: Foundational dataset for all subsequent analyses

2. **userbehavior_descriptive_stats.csv**
   - Content: Descriptive statistics of the UserBehavior dataset
   - Statistics included: Count, Mean, Standard Deviation, Minimum, 25th Percentile, Median, 75th Percentile, Maximum
   - Purpose: Understanding the basic distribution characteristics of the dataset

3. **cleaned_tmall.csv**
   - Content: Cleaned Tmall log data
   - Fields included: Item ID, User ID, Action Type, Time, Date, Hour, Day of Week
   - Purpose: Foundational dataset for Tmall user behavior analysis

4. **tmall_descriptive_stats.csv**
   - Content: Descriptive statistics of the Tmall dataset
   - Statistics included: Similar to userbehavior_descriptive_stats.csv
   - Purpose: Understanding the basic distribution characteristics of the Tmall dataset

5. **basic_statistics.csv**
   - Content: Basic statistical summary of the UserBehavior dataset
   - Metrics included: Total users, Total items, Total categories, Total behaviors
   - Purpose: Quick understanding of the dataset scale and scope

### User Behavior Analysis

6. **behavior_distribution.csv**
   - Content: Distribution of different behavior types (pv, buy, cart, fav)
   - Fields included: Behavior type, Count, Percentage
   - Purpose: Understanding the overall distribution pattern of user behaviors

7. **daily_behaviors.csv**
   - Content: Behavior counts grouped by date and behavior type
   - Fields included: Date as index, behavior types as columns
   - Purpose: Analyzing behavior trends over time

8. **hourly_behaviors.csv**
   - Content: Behavior counts grouped by hour and behavior type
   - Fields included: Hour (0-23) as index, behavior types as columns
   - Purpose: Identifying user activity patterns at different times of the day

9. **day_of_week_behaviors.csv**
   - Content: Behavior counts grouped by day of week and behavior type
   - Fields included: Day of week (Monday to Sunday) as index, behavior types as columns
   - Purpose: Identifying user activity patterns on different days of the week

10. **conversion_rates.csv**
    - Content: Conversion rates between different behavior stages
    - Fields included: From behavior, To behavior, Conversion rate (%)
    - Purpose: Analyzing user behavior conversion funnel, identifying conversion bottlenecks

11. **top_categories.csv**
    - Content: Top 20 popular item categories ranked by interaction count
    - Fields included: Category ID, Interaction count
    - Purpose: Understanding the most popular item categories

12. **purchase_frequency.csv**
    - Content: Distribution of user purchase frequency
    - Fields included: Purchase count, Number of users
    - Purpose: Analyzing the frequency distribution of user purchase behavior

13. **user_engagement_statistics.csv**
    - Content: Statistics on user engagement
    - Fields included: Statistic, Value
    - Purpose: Understanding the degree of user engagement with the platform

14. **user_engagement_distribution.csv**
    - Content: Distribution of user engagement levels
    - Fields included: Engagement level (Very Low, Low, Medium, High, Very High), Number of users
    - Purpose: Analyzing the distribution of users across different engagement levels

### Tmall Data Analysis

15. **tmall_basic_statistics.csv**
    - Content: Basic statistical summary of Tmall data
    - Metrics included: Total users, Total items, Total behaviors
    - Purpose: Understanding the scale of the Tmall dataset

16. **tmall_action_distribution.csv**
    - Content: Distribution of action types in Tmall data
    - Fields included: Action type, Count, Percentage
    - Purpose: Understanding the behavior distribution of Tmall users

17. **tmall_daily_actions.csv**
    - Content: Action counts in Tmall data grouped by date and action type
    - Fields included: Date as index, action types as columns
    - Purpose: Analyzing date patterns of Tmall user behaviors

18. **tmall_hourly_actions.csv**
    - Content: Action counts in Tmall data grouped by hour and action type
    - Fields included: Hour as index, action types as columns
    - Purpose: Analyzing Tmall user activity patterns throughout the day

19. **tmall_day_of_week_actions.csv**
    - Content: Action counts in Tmall data grouped by day of week and action type
    - Fields included: Day of week as index, action types as columns
    - Purpose: Analyzing Tmall user activity patterns throughout the week

20. **tmall_interaction_statistics.csv**
    - Content: Statistics on user-item interaction frequency in Tmall data
    - Fields included: Statistic, Value
    - Purpose: Understanding the frequency distribution of user interactions with items

### User Segmentation and Clustering

21. **user_segmentation.csv**
    - Content: User segmentation results based on RFM model
    - Fields included: User ID, Recency (days since last purchase), Frequency, R-Score, F-Score, RFM Score, User Group
    - Purpose: Analyzing user value based on recency of purchase and purchase frequency

22. **segment_summary.csv**
    - Content: Summary information for each user segment
    - Fields included: Customer Group, Average Recency, Min/Max Recency, Average Frequency, Min/Max Frequency, User Count
    - Purpose: Understanding the characteristics of different user groups

23. **user_clustering.csv**
    - Content: User clustering results based on K-means
    - Fields included: User ID, Counts of each behavior type, Active days, Categories browsed, Cluster label
    - Purpose: Clustering users based on multidimensional behavioral features

24. **cluster_descriptions.txt**
    - Content: Textual descriptions of each cluster
    - Information included: Cluster number, User count, Percentage, Significant features and their values
    - Purpose: Providing human-understandable descriptions of each cluster's characteristics

### Purchase Behavior Analysis

25. **purchase_interval_statistics.csv**
    - Content: Statistics on user purchase intervals
    - Fields included: Statistic, Days
    - Purpose: Understanding the distribution of user purchase cycles

26. **user_category_statistics.csv**
    - Content: Statistics on the number of categories purchased by users
    - Fields included: Statistic, Category count
    - Purpose: Understanding the distribution of user purchase diversity

27. **category_cooccurrence.csv**
    - Content: Frequency of category co-occurrence (top 20 pairs)
    - Fields included: Category 1, Category 2, Co-occurrence count
    - Purpose: Discovering category combinations frequently purchased by the same user

### User Journey Analysis

28. **behavior_sequences.csv**
    - Content: User behavior sequence analysis results
    - Fields included: Initial behavior, Next behavior, Count, Transition probability
    - Purpose: Analyzing user behavior transition patterns

29. **avg_time_between_behaviors.csv**
    - Content: Average time intervals between different behaviors
    - Fields included: Initial behavior, Next behavior, Average time (minutes)
    - Purpose: Understanding the time intervals between users completing different behaviors

### Category Analysis

30. **category_behavior_analysis.csv**
    - Content: Behavior analysis and conversion rates by category
    - Fields included: Category ID as index, counts of each behavior type, PV to purchase conversion rate, Cart to purchase conversion rate
    - Purpose: Analyzing user behavior patterns and conversion performance across different categories

31. **behavior_distribution_by_category.csv**
    - Content: Behavior distribution within each category
    - Fields included: Category ID, Behavior type, Count, Total, Percentage
    - Purpose: Understanding the behavior distribution within each category

### Predictive Model Analysis

32. **purchase_prediction_report.csv**
    - Content: Performance report of the purchase prediction model
    - Fields included: Precision, Recall, F1-score, Support
    - Purpose: Evaluating the performance of the purchase behavior prediction model

33. **feature_importance.csv**
    - Content: Importance ranking of features in the prediction model
    - Fields included: Feature name, Importance
    - Purpose: Understanding which features are most important for predicting purchase behavior

## Visualizations (result/figures/)

### User Behavior Visualizations

1. **behavior_distribution.png**
   - Content: Bar chart of user behavior distribution
   - Purpose: Visually showing the distribution of different behavior types

2. **daily_behavior_patterns.png**
   - Content: Line chart of daily user behavior patterns
   - Purpose: Showing behavior trends over dates

3. **hourly_behavior_patterns.png**
   - Content: Line chart of hourly user behavior patterns
   - Purpose: Showing behavior distribution across different times of the day

4. **day_of_week_behavior_patterns.png**
   - Content: Bar chart of day-of-week user behavior patterns
   - Purpose: Showing behavior distribution across different days of the week

5. **purchase_frequency.png**
   - Content: Bar chart of user purchase frequency distribution
   - Purpose: Showing the distribution of users by purchase frequency

6. **user_engagement_distribution.png**
   - Content: Bar chart of user engagement level distribution
   - Purpose: Showing the distribution of users across different engagement levels

### Tmall Data Visualizations

7. **tmall_action_distribution.png**
   - Content: Bar chart of Tmall user action distribution
   - Purpose: Showing the distribution of different action types in Tmall data

8. **tmall_daily_action_patterns.png**
   - Content: Line chart of daily Tmall user action patterns
   - Purpose: Showing Tmall behavior trends over dates

9. **tmall_hourly_action_patterns.png**
   - Content: Line chart of hourly Tmall user action patterns
   - Purpose: Showing Tmall user activity patterns throughout the day

10. **tmall_day_of_week_action_patterns.png**
    - Content: Bar chart of day-of-week Tmall user action patterns
    - Purpose: Showing Tmall user activity patterns throughout the week

11. **tmall_interaction_frequency.png**
    - Content: Histogram of user-item interaction frequency
    - Purpose: Showing the distribution of interaction counts between users and items

### User Segmentation and Clustering Visualizations

12. **user_segment_distribution.png**
    - Content: Bar chart of user segment distribution
    - Purpose: Showing the distribution of users across RFM segments

13. **kmeans_elbow.png**
    - Content: Elbow method plot for K value selection
    - Purpose: Helping determine the optimal number of clusters for K-means clustering

14. **cluster_radar_charts.png**
    - Content: Radar charts of features for each cluster
    - Purpose: Showing feature patterns of each cluster in multiple dimensions

15. **cluster_sizes.png**
    - Content: Bar chart of user count in each cluster
    - Purpose: Showing the size of each cluster in the clustering result

16. **cluster_conversion_rates.png**
    - Content: Bar chart of average purchase conversion rates by cluster
    - Purpose: Comparing purchase tendencies of users in different clusters

### Purchase Behavior Visualizations

17. **purchase_interval_distribution.png**
    - Content: Histogram of average purchase intervals for users
    - Purpose: Showing the distribution of user purchase cycles

18. **user_category_distribution.png**
    - Content: Bar chart of category count distribution for users
    - Purpose: Showing the distribution of user purchase diversity

### User Journey Visualizations

19. **behavior_transitions.png**
    - Content: Bar chart of behavior transition probabilities
    - Purpose: Showing the probability of users transitioning from one behavior to another

20. **time_between_behaviors.png**
    - Content: Bar chart of average time between user behaviors
    - Purpose: Showing the time intervals between different behaviors

### Category Analysis Visualizations

21. **top_categories.png**
    - Content: Bar chart of top 20 popular item categories
    - Purpose: Showing the most popular item categories

22. **category_conversion_rates.png**
    - Content: Bar chart of PV to purchase conversion rates by category
    - Purpose: Comparing conversion performance across different categories

23. **top5_category_behavior_distribution.png**
    - Content: Bar chart of behavior distribution for top 5 categories
    - Purpose: Showing behavior distribution within popular categories

### Predictive Model Visualizations

24. **feature_importance.png**
    - Content: Bar chart of feature importance
    - Purpose: Showing the importance of each feature for purchase prediction

### Comprehensive Visualizations

25. **conversion_funnel.png**
    - Content: Bar chart of user behavior conversion funnel
    - Purpose: Showing the overall conversion funnel from browsing to purchase

26. **user_activity_time_distribution.png**
    - Content: Composite chart of user activity time distribution
    - Purpose: Showing user activity patterns by hour and day of week

27. **hourly_daily_activity_heatmap.png**
    - Content: Heatmap of user activity by day and hour
    - Purpose: Showing user activity levels across time dimensions in heatmap form

## Executive Summary

**executive_summary.md**
- Content: Executive summary of analysis results, summarizing key findings and optimization recommendations
- Sections included:
  1. Data Overview: Basic information about the datasets
  2. Key Findings: Main analysis results
  3. User Behavior Insights: User segmentation and behavior patterns
  4. Product and Category Insights: Popular categories and conversion performance
  5. Optimization Recommendations: Improvement measures based on analysis results
- Purpose: Providing concise analysis conclusions and recommendations for management decision-making

These output files collectively form a comprehensive Taobao user behavior analysis system, analyzing user behavior patterns from multiple dimensions to provide data support for precision marketing, personalized recommendations, platform optimization, and decision-making.