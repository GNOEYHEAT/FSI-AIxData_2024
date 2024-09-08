import os
import glob
import random
import numpy as np
import pandas as pd
import zipfile
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, make_scorer

import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    def __init__(self, frac=0.01, scaler="standard", cv=10, seed=826, **kwargs):
        self.frac = frac
        self.cv = cv
        self.seed = seed
        self.scaler = self._select_scaler(scaler)
        self._set_seeds(seed)
        
        self.train_df = None
        self.test_df = None
        self.syn_df = None
        
        self.train_x = None
        self.train_y = None
        self.test_x = None
        
        self.le_subclass = LabelEncoder()
        self.submission_id = f"stacking_{scaler}_cv{cv}_seed{seed}_frac{frac}"
        self.kwargs = kwargs
        
    def _select_scaler(self, scaler):
        if scaler == "standard":
            return StandardScaler()
        elif scaler == "minmax":
            return MinMaxScaler()
        elif scaler == "robust":
            return RobustScaler()
        
    def _set_seeds(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    def load_data_TH(self, train_path, test_path): #다른것
        print('Loading Data...')
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.preprocess_data_TH()
        
    def load_data_syn(self, train_path, test_path, syn_data_path): #다른것
        print('Loading Data...')
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.syn_df = pd.read_csv(syn_data_path)
        self.train_df = self.train_df[self.syn_df.columns]
        self.preprocess_data_syn()
        
    def preprocess_data_TH(self): #다른것
        # Sampling and feature engineering
        normal_df = self.train_df[self.train_df['Fraud_Type']=='m'].sample(frac=self.frac)
        anormal_df = self.train_df[self.train_df['Fraud_Type']!='m']
        self.train_df = pd.concat([normal_df, anormal_df], axis=0).reset_index(drop=True)
        
        # Feature Engineering
        self._feature_engineering_TH(self.train_df)
        self._feature_engineering_TH(self.test_df)
        
        # Drop unnecessary columns
        drop_col = [
            'Customer_personal_identifier', 'Customer_identification_number',
            'Account_account_number', 'Account_initial_balance', 'Account_balance',
            'Account_amount_daily_limit', 'Account_remaining_amount_daily_limit_exceeded',
            'IP_Address', 'MAC_Address', 'Location',
            'Recipient_Account_Number', 'Another_Person_Account',
            'Customer_registration_datetime', 'Account_creation_datetime', 'Transaction_Datetime',
            'Last_atm_transaction_datetime', 'Last_bank_branch_transaction_datetime', 'Transaction_resumed_date',
        ]
        
        self.train_df = self.train_df.drop(drop_col, axis=1)
        self.test_df = self.test_df.drop(drop_col, axis=1)
    
        self.train_x = self.train_df.drop(['ID', 'Fraud_Type'], axis=1)
        self.test_x = self.test_df.drop(['ID'], axis=1)
        
    def preprocess_data_syn(self): #다른것
        normal_df = self.train_df[self.train_df['Fraud_Type'] == 'm'].sample(1000, random_state=self.seed)
        anormal_df = pd.DataFrame()
        for i in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']:
            temp_df = self.syn_df[self.syn_df['Fraud_Type'] == i].sample(1000, replace=True, random_state=self.seed)
            anormal_df = pd.concat([anormal_df, temp_df], axis=0)

        self.train_df = pd.concat([normal_df, anormal_df], axis=0).reset_index(drop=True)
        self.train_df = self.train_df.sample(2388, random_state=self.seed).reset_index(drop=True)

        # Feature Engineering
        self._feature_engineering_syn(self.train_df)
        self._feature_engineering_syn(self.test_df)

        # Drop unnecessary columns
        drop_col_train = ['Account_initial_balance', 'Account_balance', 'Account_amount_daily_limit', 'Account_remaining_amount_daily_limit_exceeded', 'Another_Person_Account']
        drop_col_test = drop_col_train + ['Customer_personal_identifier', 'Customer_identification_number', 'Account_account_number', 'IP_Address', 'MAC_Address', 'Location', 'Recipient_Account_Number', 'Customer_registration_datetime', 'Account_creation_datetime', 'Transaction_Datetime', 'Last_atm_transaction_datetime', 'Last_bank_branch_transaction_datetime', 'Transaction_resumed_date']

        self.train_df = self.train_df.drop(drop_col_train, axis=1)
        self.test_df = self.test_df.drop(drop_col_test, axis=1)

        self.train_x = self.train_df.drop(['Fraud_Type'], axis=1)
        self.test_x = self.test_df.drop(['ID'], axis=1)

    def _feature_engineering_TH(self, df): #다른것
        current_year = 2024
        df['Customer_Age'] = current_year - df['Customer_Birthyear']
        df['Age_Group'] = df['Customer_Age'].apply(self._age_group)
        df['Customer_credit_rating'] = df['Customer_credit_rating'].apply(self._preprocess_customer_credit_rating)
        df['Total_change_of_authentication'] = df.iloc[:, df.columns.str.startswith('Customer_flag_change_of_authentication')].sum(axis=1)
        df['Total_terminal_malicious_behavior'] = df.iloc[:, df.columns.str.startswith('Customer_flag_terminal_malicious_behavior')].sum(axis=1)
        df['Balance_change'] = df['Account_balance'] - df['Account_initial_balance']
        df['remaining_daily_limit'] = df['Account_amount_daily_limit'] - df['Account_remaining_amount_daily_limit_exceeded']
        df['Time_difference'] = pd.to_timedelta(df['Time_difference']).dt.total_seconds()
        df['rooting_jailbreak_Transaction_Amount'] = df['Customer_rooting_jailbreak_indicator'] * df['Transaction_Amount']
        df['mobile_roaming_Transaction_Amount'] = df['Customer_mobile_roaming_indicator'] * df['Transaction_Amount']
        df['Transaction_Amount_to_Daily_Limit_Ratio'] = df['Transaction_Amount'] / df['Account_amount_daily_limit']
        df['Transaction_UCL'] = abs(df['Transaction_Amount']) + df['Account_one_month_std_dev'] * 1.645
        df['Over_UCL'] = df.apply(lambda row: 1 if row['Account_one_month_max_amount'] > row['Transaction_UCL'] else 0, axis=1)
                
        # Stacking_TH에만 있는 것
        df['creation_to_registration_timedelta'] = (pd.to_datetime(df['Account_creation_datetime']) - pd.to_datetime(df['Customer_registration_datetime'])).dt.total_seconds().astype(int)
        df['transaction_to_creation_timedelta'] = (pd.to_datetime(df['Transaction_Datetime']) - pd.to_datetime(df['Account_creation_datetime'])).dt.total_seconds().astype(int)
        df['last_atm_timedelta'] = (pd.to_datetime(df['Transaction_Datetime']) - pd.to_datetime(df['Last_atm_transaction_datetime'])).dt.total_seconds().astype(int)
        df['last_bank_branch_timedelta'] = (pd.to_datetime(df['Transaction_Datetime']) - pd.to_datetime(df['Last_bank_branch_transaction_datetime'])).dt.total_seconds().astype(int)
        df['resumed_date_timedelta'] = (pd.to_datetime(df['Transaction_Datetime']) - pd.to_datetime(df['Transaction_resumed_date'])).dt.total_seconds().astype(int)

    def _feature_engineering_syn(self, df): #다른것
        current_year = 2024
        df['Customer_Age'] = current_year - df['Customer_Birthyear']
        df['Age_Group'] = df['Customer_Age'].apply(self._age_group)
        df['Customer_credit_rating'] = df['Customer_credit_rating'].apply(self._preprocess_customer_credit_rating)
        df['Total_change_of_authentication'] = df.iloc[:, df.columns.str.startswith('Customer_flag_change_of_authentication')].sum(axis=1)
        df['Total_terminal_malicious_behavior'] = df.iloc[:, df.columns.str.startswith('Customer_flag_terminal_malicious_behavior')].sum(axis=1)
        df['Balance_change'] = df['Account_balance'] - df['Account_initial_balance']
        df['remaining_daily_limit'] = df['Account_amount_daily_limit'] - df['Account_remaining_amount_daily_limit_exceeded']
        df['Time_difference'] = pd.to_timedelta(df['Time_difference']).dt.total_seconds()
        df['rooting_jailbreak_Transaction_Amount'] = df['Customer_rooting_jailbreak_indicator'] * df['Transaction_Amount']
        df['mobile_roaming_Transaction_Amount'] = df['Customer_mobile_roaming_indicator'] * df['Transaction_Amount']
        df['Transaction_Amount_to_Daily_Limit_Ratio'] = df['Transaction_Amount'] / df['Account_amount_daily_limit']
        df['Transaction_UCL'] = abs(df['Transaction_Amount']) + df['Account_one_month_std_dev'] * 1.645
        df['Over_UCL'] = df.apply(lambda row: 1 if row['Account_one_month_max_amount'] > row['Transaction_UCL'] else 0, axis=1)
        
    def _age_group(self, x):
        if x < 20:
            return 0
        elif 20 <= x < 30:
            return 1
        elif 30 <= x < 40:
            return 2
        elif 40 <= x < 50:
            return 3
        elif 50 <= x < 60:
            return 4
        else:
            return 5

    def _preprocess_customer_credit_rating(self, x):
        mapping = {'S': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
        return mapping.get(x, x)
    
    def encode_and_scale(self):
        
        self.train_y = self.le_subclass.fit_transform(self.train_df['Fraud_Type'])
        print("Label Encoding mapping:")
        for i, label in enumerate(self.le_subclass.classes_):
            print(f"Original label: {label}, Encoded label: {i}")

        le_col = ['Customer_Gender']
        for col in le_col:
            le = LabelEncoder()
            le.fit(self.train_x[col])
            self.train_x[col] = le.transform(self.train_x[col])

            for case in np.unique(self.test_x[col]):
                if case not in le.classes_:
                    le.classes_ = np.append(le.classes_, case)
            self.test_x[col] = le.transform(self.test_x[col])

        # Target Encoding
        te_col = ['Customer_credit_rating', 'Customer_loan_type',
                  'Account_account_type',
                  'Channel', 'Operating_System', 'Error_Code',
                  'Type_General_Automatic', 'Access_Medium'
        ]
        
        for col in te_col:
            te = TargetEncoder(target_type="continuous")
            te.fit(self.train_x[col].values[:, np.newaxis], self.train_y)
            self.train_x[col + '_te'] = te.transform(self.train_x[col].values[:, np.newaxis])
            self.test_x[col + '_te'] = te.transform(self.test_x[col].values[:, np.newaxis])

        # One-Hot Encoding
        ohe_col = [
            'Customer_credit_rating', 'Customer_loan_type',
            'Account_account_type',
            'Channel', 'Operating_System', 'Error_Code',
            'Type_General_Automatic', 'Access_Medium'
        ]
                
        train_ohe = []
        test_ohe = []

        for col in ohe_col:
            ohe = OneHotEncoder(handle_unknown="ignore")
            ohe.fit(self.train_x[col].values.reshape(-1, 1))
            train_ohe.append(ohe.transform(self.train_x[col].values.reshape(-1, 1)).toarray())
            test_ohe.append(ohe.transform(self.test_x[col].values.reshape(-1, 1)).toarray())

        self.train_x = self.train_x.drop(ohe_col, axis=1)
        self.test_x = self.test_x.drop(ohe_col, axis=1)

        self.train_x = self.scaler.fit_transform(self.train_x)
        self.test_x = self.scaler.transform(self.test_x)

        self.train_x = np.concatenate((self.train_x, np.hstack(train_ohe)), axis=1)
        self.test_x = np.concatenate((self.test_x, np.hstack(test_ohe)), axis=1)
        
        
    def load_meta_data(self, directory_path): #다른것
        train_files = sorted(glob.glob(os.path.join(directory_path, 'meta_ml_X_train*')))
        test_files = sorted(glob.glob(os.path.join(directory_path, 'meta_ml_X_test*')))
        
        meta_ml_X_train_parts = [np.load(path) for path in train_files]
        meta_ml_X_test_parts = [np.load(path) for path in test_files]
        
        print('meta_ml_X_train : ', train_files)
        print('meta_ml_X_test : ', test_files)

        self.meta_ml_X_train = np.concatenate(meta_ml_X_train_parts, axis=1)
        self.meta_ml_X_test = np.concatenate(meta_ml_X_test_parts, axis=1)
        
        print('meta_ml_X_train concatenated shape : ', self.meta_ml_X_train.shape)
        print('meta_ml_X_test concatenated shape : ', self.meta_ml_X_test.shape)
        
    def get_stacking_ml_datasets(self, model, X_train_n, y_train_n, X_test_n, n_folds, fitting=True):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        n_classes = 13
        train_fold_pred = np.zeros((X_train_n.shape[0], n_classes))
        test_pred = np.zeros((X_test_n.shape[0], n_folds, n_classes))

        for folder_counter, (train_index, valid_index) in enumerate(skf.split(X_train_n, y_train_n)):
            X_tr = X_train_n[train_index]
            y_tr = y_train_n[train_index]
            X_te = X_train_n[valid_index]
            
            if fitting:
                model.fit(X_tr, y_tr)
                
            train_fold_pred[valid_index, :] = model.predict_proba(X_te)
            test_pred[:, folder_counter] = model.predict_proba(X_test_n)

        test_pred_mean = np.mean(test_pred, axis=1)
        return train_fold_pred, test_pred_mean      

    def train_meta_model(self): #다른것
        meta_clf = LogisticRegression(n_jobs=-1, random_state=self.seed)
        meta_clf.fit(self.meta_ml_X_train, self.train_y)
        prediction = meta_clf.predict(self.meta_ml_X_test)

        predictions_label = self.le_subclass.inverse_transform(prediction)

        return predictions_label
    
    def gen_meta_syn(self): #다른것
        print('Generating Meta Data... ')
        base_ml = [
            XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_jobs=-1, random_state=self.seed),
        ]

        meta_ml_X_train, meta_ml_X_test = [], []

        for estimator in base_ml:
            temp_X_train, temp_X_test = self.get_stacking_ml_datasets(estimator, self.train_x, self.train_y, self.test_x, self.cv)
            meta_ml_X_train.append(temp_X_train)
            meta_ml_X_test.append(temp_X_test)

        meta_ml_X_train = np.hstack(meta_ml_X_train)
        meta_ml_X_test = np.hstack(meta_ml_X_test)

        return meta_ml_X_train, meta_ml_X_test
    
    def gen_meta_raw(self): #다른것
        print('Generating Meta Data... ')
        xgboost_params = {
            'n_estimators': 1976,
            'max_depth': 6,
            'learning_rate': 0.008294125648045027, 
            'gamma': 0.11044398100317245,
            'min_child_weight': 1,
            'subsample': 0.9,
            'sampling_method': 'gradient_based',
            'colsample_bytree': 0.8,
            'reg_alpha': 0.03296137174022581,
            'reg_lambda': 0.006095201538414734,
            'tree_method': 'gpu_hist',
            'n_jobs' : -1,
            'random_state': self.seed,
            'eval_metric' : self._macro_f1_scorer(),
        }

        xgboost = XGBClassifier(**xgboost_params)

        best_ml = [xgboost]
        meta_ml_X_train=[]
        meta_ml_X_test=[]

        for estimator in best_ml:
            temp_X_train, temp_X_test = self.get_stacking_ml_datasets(estimator, self.train_x, self.train_y, self.test_x, self.cv)
            meta_ml_X_train.append(temp_X_train)
            meta_ml_X_test.append(temp_X_test)
        
        meta_ml_X_train = np.hstack(meta_ml_X_train)
        meta_ml_X_test = np.hstack(meta_ml_X_test)
        
        return meta_ml_X_train, meta_ml_X_test
    
    def _macro_f1_scorer(self): #다른것
        return make_scorer(lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'), greater_is_better=True)
    
    def save_submission(self, predictions, sample_submission_path, output_dir): #다른것
        clf_submission = pd.read_csv(sample_submission_path)
        clf_submission["Fraud_Type"] = predictions
        
        all_synthetic_data = pd.read_csv("syn_data/ctgan_syn_submission.csv")

        clf_submission.to_csv(os.path.join(output_dir, 'clf_submission.csv'), encoding='UTF-8-sig', index=False)
        all_synthetic_data.to_csv(os.path.join(output_dir, 'syn_submission.csv'), encoding='UTF-8-sig', index=False)

        with zipfile.ZipFile(os.path.join(output_dir, f"{self.submission_id}.zip"), 'w') as submission:
            submission.write(os.path.join(output_dir, 'clf_submission.csv'))
            submission.write(os.path.join(output_dir, 'syn_submission.csv'))
            
        
        print('Submission saved successfully.')
        
    def save_syn_meta_data(self, train_data, test_data, filename_prefix="forestdiffusion"): #다른것
        np.save(f'meta_data/meta_ml_X_train_{filename_prefix}_{self.seed}.npy', train_data)
        np.save(f'meta_data/meta_ml_X_test_{filename_prefix}_{self.seed}.npy', test_data)
        print('metadata saved successfully.')
        
    def save_raw_meta_data(self, train_data, test_data): #다른것
        np.save(f'meta_data/meta_ml_X_train_{self.seed}.npy', train_data)
        np.save(f'meta_data/meta_ml_X_test_{self.seed}.npy', test_data)
        print('metadata saved successfully.')



    def ensemble(**kwargs):
        model = FraudDetectionModel(frac=kwargs.get('frac'), scaler=kwargs.get('scaler'), cv=kwargs.get('cv'), seed=kwargs.get('seed'))
        model.load_data_TH(train_path=kwargs.get('train_path'), test_path=kwargs.get('test_path'))
        model.encode_and_scale()

        meta_data_path = './meta_data/'

        model.load_meta_data(meta_data_path)

        predictions = model.train_meta_model()
        model.save_submission(predictions, "data/sample_submission.csv", "./submission/")
        
    def gen_syn_meta_data(**kwargs):
        model = FraudDetectionModel(frac=kwargs.get('frac'), scaler=kwargs.get('scaler'), cv=kwargs.get('cv'), seed=kwargs.get('seed'))
        model.load_data_syn(train_path=kwargs.get('train_path'), test_path=kwargs.get('test_path'), syn_data_path=kwargs.get('syn_data_path'))
        model.encode_and_scale()

        meta_ml_X_train, meta_ml_X_test = model.gen_meta_syn()
        model.save_syn_meta_data(meta_ml_X_train, meta_ml_X_test)
        
    def gen_raw_meta_data(**kwargs):
        model = FraudDetectionModel(frac=kwargs.get('frac'), scaler=kwargs.get('scaler'), cv=kwargs.get('cv'), seed=kwargs.get('seed'))
        model.load_data_TH(train_path=kwargs.get('train_path'), test_path=kwargs.get('test_path'))
        model.encode_and_scale()
        
        meta_ml_X_train, meta_ml_X_test = model.gen_meta_raw()
        model.save_raw_meta_data(meta_ml_X_train, meta_ml_X_test)
        