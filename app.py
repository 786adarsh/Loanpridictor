from flask import Flask, render_template,url_for,request
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
import gc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


start_df = pd.read_csv('loan.csv', low_memory=False)

df = start_df.copy(deep=True)
df.head()

cuurent_loan= df[(df['loan_status'] == 'Issued')].copy()

df.drop(df[(df['loan_status'] == 'Does not meet the credit policy. Status:Fully Paid')].index, inplace = True) 
df.drop(df[(df['loan_status'] == 'Does not meet the credit policy. Status:Charged Off')].index, inplace = True) 
df.drop(df[(df['loan_status'] == 'Issued')].index, inplace = True)
df.drop(df[(df['loan_status'] == 'Current')].index, inplace = True)
df['loan_status']

#target_list = [ 0 if i=='Fully Paid' else 1 for i in df['loan_status']]
bad_loan = ["Charged Off", "Default","Late (16-30 days)", "Late (31-120 days)"]


lst = [df]
df['emp_length_int'] = np.nan

for col in lst:
    col.loc[col['emp_length'] == '10+ years', "emp_length_int"] = 10
    col.loc[col['emp_length'] == '9 years', "emp_length_int"] = 9
    col.loc[col['emp_length'] == '8 years', "emp_length_int"] = 8
    col.loc[col['emp_length'] == '7 years', "emp_length_int"] = 7
    col.loc[col['emp_length'] == '6 years', "emp_length_int"] = 6
    col.loc[col['emp_length'] == '5 years', "emp_length_int"] = 5
    col.loc[col['emp_length'] == '4 years', "emp_length_int"] = 4
    col.loc[col['emp_length'] == '3 years', "emp_length_int"] = 3
    col.loc[col['emp_length'] == '2 years', "emp_length_int"] = 2
    col.loc[col['emp_length'] == '1 year', "emp_length_int"] = 1
    col.loc[col['emp_length'] == '< 1 year', "emp_length_int"] = 0.5
    col.loc[col['emp_length'] == 'n/a', "emp_length_int"] = 0
    
df['TARGET'] = np.nan  
def loan_condition(status):
    if status in bad_loan:
        return 1
    else:
        return 0
    
    
df['TARGET'] = df['loan_status'].apply(loan_condition)
df.drop('loan_status',axis=1,inplace=True)
df['emp_length_int'].fillna(value=0,inplace=True)


df.drop(['id','member_id','emp_title','title','zip_code','url',
    'mths_since_last_record',
    'mths_since_last_major_derog',
    'annual_inc_joint',
    'dti_joint',
    'verification_status_joint',
    'open_acc_6m',
        'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il',
        'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc',
        'all_util', 'inq_fi', 'total_cu_tl',
        'inq_last_12m','tot_coll_amt', 'desc', 'tot_cur_bal', 'total_rev_hi_lim', 
        'sub_grade','emp_length'],axis=1,inplace=True)

     
df['issue_d']= pd.to_datetime(df['issue_d']).apply(lambda x: int(x.strftime('%Y')))
df['last_pymnt_d']= pd.to_datetime(df['last_pymnt_d'].fillna('2016-01-01')).apply(lambda x: int(x.strftime('%m')))
df['last_credit_pull_d']= pd.to_datetime(df['last_credit_pull_d'].fillna("2016-01-01")).apply(lambda x: int(x.strftime('%m')))
df['earliest_cr_line']= pd.to_datetime(df['earliest_cr_line'].fillna('2001-08-01')).apply(lambda x: int(x.strftime('%m')))
df['next_pymnt_d'] = pd.to_datetime(df['next_pymnt_d'].fillna(value = '2016-02-01')).apply(lambda x:int(x.strftime("%Y")))
df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(df['mths_since_last_delinq'].median())

for col in ('mths_since_last_delinq','revol_util', 
                'collections_12_mths_ex_med', 'open_acc', 'inq_last_6mths',
            'acc_now_delinq'):
        df[col] = df[col].fillna(0)
        
df.isnull().sum().max()
    #df.dropna(inplace=True)

labelencoder_X = LabelEncoder()
df['term']=labelencoder_X.fit_transform(df['term'])
df['verification_status']=labelencoder_X.fit_transform(df['verification_status'])
df['pymnt_plan']=labelencoder_X.fit_transform(df['pymnt_plan'])
df['grade']=labelencoder_X.fit_transform(df['grade'])
df['home_ownership']=labelencoder_X.fit_transform(df['home_ownership'])
df['purpose']=labelencoder_X.fit_transform(df['purpose'])
df['addr_state']=labelencoder_X.fit_transform(df['addr_state'])
df['initial_list_status']=labelencoder_X.fit_transform(df['initial_list_status'])
df['application_type']=labelencoder_X.fit_transform(df['application_type'])

print(df.shape)
df['TARGET'].value_counts()

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
            
    elif train==False:
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))  
            
X_train, X_test, y_train, y_test = train_test_split(df.drop('TARGET',axis=1),df['TARGET'],test_size=0.25,random_state=101)

del start_df
gc.collect()

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)

sm = SMOTE(random_state=12, ratio = 1.0)
x_train_r, y_train_r = sm.fit_sample(X_train, y_train)
   

log_reg = LogisticRegression(C = 0.0001,random_state=21)
log_reg.fit(x_train_r, y_train_r)
    #log_reg.fit(X_train, y_train)
    #print_score(log_reg, X_train, y_train, X_test, y_test, train=False)
print_score(log_reg, x_train_r, y_train_r, X_test, y_test, train=False)
 
pickle.dump(log_reg, open('model.pkl','wb'))






   