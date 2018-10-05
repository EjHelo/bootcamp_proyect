# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 23:00:02 2018

@author: Amanda
"""

import pandas as pd
import numpy as np
from sklearn import linear_model,cross_validation
from sklearn.model_selection import train_test_split

#df1 = pd.read_csv("https://s3.amazonaws.com/fordgobike-data/2017-fordgobike-tripdata.csv",parse_dates=True)
df2 = pd.read_csv("data/201801-fordgobike-tripdata.csv",parse_dates=True)
df3 = pd.read_csv("data/201802-fordgobike-tripdata.csv",parse_dates=True)
df4 = pd.read_csv("data/201803-fordgobike-tripdata.csv",parse_dates=True)
df5 = pd.read_csv("data/201804-fordgobike-tripdata.csv",parse_dates=True)
df6 = pd.read_csv("data/201805-fordgobike-tripdata.csv",parse_dates=True)
df7 = pd.read_csv("data/201806-fordgobike-tripdata.csv",parse_dates=True)
df8 = pd.read_csv("data/201807-fordgobike-tripdata.csv",parse_dates=True)

df = pd.concat([df2, df3, df4, df5, df6, df7, df8], axis=0)

mediana = df['member_birth_year'].median()
df['member_birth_year'] = df['member_birth_year'].replace(np.NaN, mediana)
df['member_gender'] = df['member_gender'].fillna('Other')
df = df[np.isfinite(df['end_station_id'])]

dummies_user_type = pd.get_dummies(df.user_type)
merged_user_type = pd.concat([df, dummies_user_type],axis='columns')
df_temporal = merged_user_type.drop(['user_type'], axis='columns')
dummies_member_gender = pd.get_dummies(df.member_gender)
dummies_bike_share = pd.get_dummies(df.bike_share_for_all_trip)
merged_gender = pd.concat([df_temporal, dummies_member_gender,dummies_bike_share],axis='columns')
df_temporal2 = merged_gender.drop(['member_gender',"end_time","start_time","bike_share_for_all_trip","end_station_name","start_station_name"], axis='columns')




y = df_temporal2.pop('duration_sec')
X = df_temporal2
#X["end_station_name"] = X["end_station_name"].astype("category") 
#X["start_station_name"] = X["start_station_name"].astype("category") 

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.70)

#clf = linear_model.LinearRegression()
#clf = linear_model.Lasso(alpha = 0.1)
clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
clf.score(X_test, y_test)

y_pred = clf.predict(X_test)

score = cross_validation.cross_val_score(clf, X, y)
print(score)

