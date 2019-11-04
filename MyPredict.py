from flask import Flask, request, render_template
import pandas as pd
import pickle
import io
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from flask import Flask, render_template, url_for

app = Flask(__name__,static_url_path='/static',static_folder = "static")


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        
        
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

        loaded_model = pickle.load(open("model.pkl","rb"))
        result = loaded_model.predict(df)
        total_loan=result.shape[0]
        good_loan=np.count_nonzero(result == 0)
        bad_loan=np.count_nonzero(result == 1)
        good_loan_per=  (good_loan/total_loan*100)
        bad_loan_per=   (bad_loan/total_loan*100)
        print('Percentage of Good Loan is: {:.2f}% '.format(good_loan_per))
        print('Percentage of Bad Loan is: {:.2f}% '.format(bad_loan_per))

        
        return render_template('upload.html', good_lo=good_loan_per, bad_lo=bad_loan_per)
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)