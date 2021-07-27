import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

train = pd.read_csv('train.csv')
train_df = train.copy()
train_df.head()

train_df.info()

store = pd.read_csv('store.csv')
store_df = store.copy()
store_df.head()

store_df.info()

test_ = pd.read_csv('test.csv')
test_df = test_.copy()
test_df.head()

test_df.info()

train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df.info()

test_df['Date'] = pd.to_datetime(test_df['Date'])
test_df.info()

closed_store_df = train_df[train_df['Open']==0].copy()
closed_store_df['Sales'].value_counts()

closed_store_df['Customers'].value_counts()

train_df['Year'] = pd.DatetimeIndex(train_df['Date']).year
train_df['Month'] = pd.DatetimeIndex(train_df['Date']).month
train_df['Day'] = pd.DatetimeIndex(train_df['Date']).day
train_df.head()

test_df['Year'] = pd.DatetimeIndex(test_df['Date']).year
test_df['Month'] = pd.DatetimeIndex(test_df['Date']).month
test_df['Day'] = pd.DatetimeIndex(test_df['Date']).day
test_df.head()

train_df.hist(bins = 20, figsize = (20,20), color = 'r')

test_df.hist(bins = 20, figsize = (20,20), color = 'r')

store_df.hist(bins = 20, figsize = (20,20), color = 'r')

train_df.head()

test_df.head()

test_df.drop('Id',axis=1,inplace=True)
test_df.head()

test_df.info()

test_df[test_df['Open'].isnull()]

mask = (train_df['Year'] ==2014) & (train_df['Month'] ==8) & (train_df['Store']==622)
train_df.loc[mask].head(100)

test_df['Open'].fillna(1,inplace=True)
test_df.info()

store_df.head()

train_df['StoreType'] = train_df['Store']
train_df.head()

test_df['StoreType'] = test_df['Store']
test_df.head()

dict_ = pd.Series(store_df.StoreType.values, index= store_df.Store).to_dict()
train_df['StoreType'].replace(dict_, inplace=True)
train_df.head(10)

test_df['StoreType'].replace(dict_, inplace=True)
test_df.head(10)

store_df.info()

train_df['Assortment'] = train_df['Store']
train_df.head()

test_df['Assortment'] = test_df['Store']
test_df.head()

dict_1 = pd.Series(store_df.Assortment.values, index= store_df.Store).to_dict()
train_df['Assortment'].replace(dict_1, inplace=True)
train_df.head(10)

test_df['Assortment'].replace(dict_1, inplace=True)
test_df.head(10)

store_df.head()

store_df.info()

store_df[store_df['CompetitionDistance'].isnull()]

store_df['CompetitionDistance'].describe()

train_df[train_df['Store']==879].head(10)

test_df[test_df['Store']==879].head(10)

store_df.head(20)

store_df.info()

train_df['CompetitionDistance'] = train_df['Store']
train_df.head()

test_df['CompetitionDistance'] = test_df['Store']
test_df.head()

dict_2 = pd.Series(store_df.CompetitionDistance.values, index=store_df.Store).to_dict()
train_df['CompetitionDistance'].replace(dict_2,inplace=True)
train_df.head()

test_df['CompetitionDistance'].replace(dict_2,inplace=True)
test_df.head()

train_df.info()

train_df['CompetitionDistance'].fillna(0, inplace=True)
train_df.info()

test_df.info()

test_df['CompetitionDistance'].fillna(0, inplace=True)
test_df.info()

train_df.Year.max()

store_df.info()

store_df['CompetitionOpenSinceYear'].fillna(2015, inplace=True)
store_df.info()

train_df['Comp_Year'] = train_df['Store']
train_df.head()

test_df['Comp_Year'] = test_df['Store']
test_df.head()

dict_3 = pd.Series(store_df.CompetitionOpenSinceYear.values, index=store_df.Store).to_dict()
train_df['Comp_Year'].replace(dict_3, inplace=True)
train_df.head()

test_df['Comp_Year'].replace(dict_3, inplace=True)
test_df.head()

train_df['2015'] = 2015
train_df.head()

test_df['2015'] = 2015
test_df.head()

train_df['Comp_Year'] = train_df['2015'] - train_df['Comp_Year']
train_df.head()

test_df['Comp_Year'] = test_df['2015'] - test_df['Comp_Year']
test_df.head()

train_df.info()

test_df.info()

store_df.info()

train_df['Promo2'] = train_df['Store']
train_df.head()

test_df['Promo2'] = test_df['Store']
test_df.head()

dict_4 = pd.Series(store_df.Promo2.values, index = store_df.Store).to_dict()
train_df['Promo2'].replace(dict_4, inplace=True)
train_df.head()

test_df['Promo2'].replace(dict_4, inplace=True)
test_df.head()

store_df['Promo2SinceWeek'].fillna(52, inplace=True)
store_df.info()

train_df['Promo2Week'] = train_df['Store']
train_df.head()

test_df['Promo2Week'] = test_df['Store']
test_df.head()

dict_5 = pd.Series(store_df.Promo2SinceWeek.values, index= store_df.Store).to_dict()
train_df['Promo2Week'].replace(dict_5, inplace=True)
train_df.head()

test_df['Promo2Week'].replace(dict_5, inplace=True)
test_df.head()

train_df['52'] = 52
train_df.head()

test_df['52'] = 52
test_df.head()

train_df['Promo2Week'] = train_df['52'] - train_df['Promo2Week']
train_df.head()

test_df['Promo2Week'] = test_df['52'] - test_df['Promo2Week']
test_df.head()

train_df.drop('52', axis=1,inplace=True)
train_df.head()

test_df.drop('52', axis=1,inplace=True)
test_df.head()

train_df.info()

train_df['Promo2Year'] = train_df['Store']
train_df.head()

test_df['Promo2Year'] = test_df['Store']
test_df.head()

dict_6 = pd.Series(store_df.Promo2SinceYear.values, index= store_df.Store).to_dict()
train_df['Promo2Year'].replace(dict_6, inplace=True)
train_df.head()

test_df['Promo2Year'].replace(dict_6, inplace=True)
test_df.head()

train_df['Promo2Year'].fillna(2015, inplace=True)
train_df.info()

test_df['Promo2Year'].fillna(2015, inplace=True)
test_df.info()

train_df['Promo2Year'] = train_df['2015'] - train_df['Promo2Year']
test_df['Promo2Year'] = test_df['2015'] - test_df['Promo2Year']
train_df.head()

test_df.head()

train_df.drop('2015', axis=1,inplace=True)
test_df.drop('2015', axis=1,inplace=True)
train_df.info()

test_df.info()

train_df.drop('Customers', axis=1, inplace=True)
train_df.info()

test_df['Open'] = test_df['Open'].astype('int64')
test_df.info()

test_df.head()

from fbprophet import Prophet

school_holidays = train_df[train_df['SchoolHoliday'] == 1].loc[:, 'Date'].values
state_holidays = train_df [ (train_df['StateHoliday'] == 'a') | (train_df['StateHoliday'] == 'b') | (train_df['StateHoliday'] == 'c')  ].loc[:, 'Date'].values

state_holidays = pd.DataFrame({'ds': pd.to_datetime(state_holidays),
                               'holiday': 'state_holiday'})
state_holidays.head()

school_holidays = pd.DataFrame({'ds': pd.to_datetime(school_holidays),
                                'holiday': 'school_holiday'})
school_holidays.head()

school_state_holidays = pd.concat((state_holidays, school_holidays))
school_state_holidays.tail(5)

sales_df = train_df[ train_df['Store'] == 10 ]
sales_df = sales_df[['Date', 'Sales']].rename(columns = {'Date': 'ds', 'Sales':'y'})
sales_df = sales_df.sort_values('ds')

model= Prophet(holidays = school_state_holidays)
model.fit(sales_df)
future   = model.make_future_dataframe(periods = 60)
forecast = model.predict(future)
figure   = model.plot(forecast, xlabel='Date', ylabel='Sales')
figure2  = model.plot_components(forecast)

from fbprophet.diagnostics import cross_validation
df_cv = cross_validation(model, initial='365 days', period ='180 days', horizon= '365 days')
# initial parametresi ile cv yapılacak model için eğitim periyodunu belirttik
# period parametresi ile de 180 günde bir test edileceğini belirtmiş olduk
# horizon ile de 365 gün için tahmin gerçekleştirmek istediğimizi belirtmiş olduk
df_cv.head()

from fbprophet.diagnostics import performance_metrics
df_p = performance_metrics(df_cv)
df_p.head()

from fbprophet.plot import plot_cross_validation_metric
fig = plot_cross_validation_metric(df_cv, metric = 'rmse')

train_df.head()

test_df.head()

train_df.drop('Date', axis=1, inplace=True)
test_df.drop('Date', axis=1, inplace=True)
train_df.info()

test_df.info()

train_ = pd.get_dummies(train_df, columns=['StateHoliday','StoreType','Assortment'])
train_.info()

train_df['StateHoliday'].replace({0:'0'}, inplace=True)
train_df['StateHoliday'].value_counts()

train_ = pd.get_dummies(train_df, columns=['StateHoliday','StoreType','Assortment'])
train_.info()

test_df_ = pd.get_dummies(test_df, columns=['StateHoliday','StoreType','Assortment'])
test_df_.info()

train_.drop(train_.loc[:,['StateHoliday_b','StateHoliday_c']],axis=1, inplace=True)
train_.info()

from sklearn.model_selection import train_test_split
y = train_['Sales'].copy()
x = train_.drop('Sales', axis = 1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)

pip install lightgbm

from lightgbm import LGBMRegressor

lgbm = LGBMRegressor()

lgbm_grid = {
    'learning_rate': [0.1, 0.5],
    'n_estimators': [1000]}
from sklearn.model_selection import GridSearchCV
lgbm_cv_model = GridSearchCV(lgbm, lgbm_grid,cv=10, n_jobs = -1, verbose = 2)
lgbm_cv_model.fit(x_train, y_train)

lgbm_cv_model.best_params_

lgbm_tuned = LGBMRegressor(learning_rate= 0.5, n_estimators= 1000)
lgbm_tuned.fit(x_train, y_train)

from sklearn.metrics import mean_squared_error
y_pred = lgbm_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))

predictions = lgbm_tuned.predict(test_df_)

sample_submission = pd.read_csv('sample_submission.csv')
sample_submission.head()

sample_submission['Sales'] = predictions
sample_submission.head()

sample_submission.to_csv('sample_submission2.csv',index=False)


