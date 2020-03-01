import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE

user_df = pd.read_csv('takehome_users.csv', encoding="ISO-8859-1", parse_dates=['creation_time', 'last_session_creation_time'])
logins_df = pd.read_csv('takehome_user_engagement.csv', encoding="ISO-8859-1")
# print(user_df.isnull().any(), logins_df.isnull().any())

# There seems to be two fields that have null values: last_session_creation_time, and invited_by_user_id
# In this scenario, forward filling or imputing invited_by_user_id would likely skew the data rather than balancing it
# This is why I will be replaceing it with zeros instead of the null values, and forward filling the timestamps for last_session_creation_time

user_df['invited_by_user_id'] = user_df['invited_by_user_id'].fillna(0)
user_df['last_session_creation_time'] = user_df['last_session_creation_time'].fillna(method='ffill')

# Changing categories to numerical to play better with classifier
user_df['creation_source'] = user_df['creation_source'].astype('category')
cat_columns = user_df.select_dtypes(['category']).columns
user_df[cat_columns] = user_df[cat_columns].apply(lambda x: x.cat.codes)

# Changing dates to play better with classifier
user_df['creation_time'] = user_df['creation_time'].astype(np.int64)
user_df['last_session_creation_time'] = user_df['last_session_creation_time'].astype(np.int64)

# normalizing the datetime to see which users are already adopted
logins_df['time_stamp'] = pd.to_datetime(logins_df['time_stamp'])
logins_df['time_stamp'] = logins_df['time_stamp'].dt.normalize().astype(np.int64)

# Dropping dupes to see which consecutives days were logged in for users via rolling windows
# Creating a timedelta between each 3 logins to see if they were all within a single week
# End by creating a new column by which we can see in the user_df who is adopted and who isn't typed by 1/0
logins_df = logins_df.sort_values(['user_id', 'time_stamp']).drop_duplicates()
three_day_chunks = logins_df.groupby(['user_id'])['time_stamp'].rolling(window=3)
day_interval = pd.to_timedelta(three_day_chunks.max() - three_day_chunks.min()).dt.days
adopted_users = list(dict.fromkeys(day_interval[day_interval <= 7].index.get_level_values('user_id').to_list()))
user_df['adopted'] = np.where(user_df['object_id'].isin(adopted_users), 1, 0)

# I would rather try to use a different form of feature selection that would allow me to gauge which features would be important.
# But as I have wrangled the data to provide a column that does acknowledge whether or not the user is adopted,
# I am more willing to fit a model on it and conduct feature elimination via that.
# My initial thoughts are to use RFE, or Decision Trees after thinking this through.
# The following features have been removed as they seem redundant in inclusion, or provide no empirical predictive value:
# 'object_id', 'name', and 'email'

# Data selection for model
X = user_df[[
    'creation_time', #1
    'creation_source', #3
    'last_session_creation_time', #1
    'opted_in_to_mailing_list', #4
    'enabled_for_marketing_drip', #5
    'org_id', #1
    'invited_by_user_id' #2
    ]]
y = user_df['adopted']

# Decision Tree Model
clf = DecisionTreeClassifier(random_state=0)
# Recursive feature selection
estimator = clf
selector = RFE(estimator, 3, step=1)
selector = selector.fit(X, y)
# Ranking of features
print("Feature Ranking: ", selector.ranking_)
# [1 3 1 4 5 1 2]
# The features that seems to be most important are the 'creation_time', 'last_session_creation_time', and 'org_id'
# The feature that seemed to follow close behind was 'invited_by_user_id'
# These features seem to be the most important when predicting future adoption of users
