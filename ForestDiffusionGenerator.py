import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ForestDiffusion import ForestDiffusionModel
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings('ignore')

class ForestDiffusionGenerator:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.set_seeds(self.seed)
        self.submission_id = f"{args.diffusion_type}-n_t{args.n_t}-duplicate_K{args.duplicate_K}-seed{args.seed}-n_sample{args.n_sample}"
        self.label_encoders = {}
        
    def set_seeds(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        
    def load_data(self, train_path="data/train.csv", test_path="data/test.csv"):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        
    def preprocess_data(self):
        syn_drop_col = [
            "ID", 
            "Customer_personal_identifier",
            "Customer_identification_number",
            "Customer_registration_datetime",
            "Account_account_number",
            "Account_creation_datetime",
            "Transaction_Datetime",
            "IP_Address",
            "MAC_Address",
            "Location",
            "Recipient_Account_Number",
            "Last_atm_transaction_datetime",
            "Last_bank_branch_transaction_datetime",
            "Transaction_resumed_date"
        ]
        temp_df = self.train_df.drop(syn_drop_col, axis=1)
        temp_df['Time_difference'] = pd.to_timedelta(temp_df['Time_difference']).dt.total_seconds()
        self.temp_df = temp_df
        
        '''
        ['Customer_Gender', 'Customer_flag_change_of_authentication_1',
        'Customer_flag_change_of_authentication_2',
        'Customer_flag_change_of_authentication_3',
        'Customer_flag_change_of_authentication_4',
        'Customer_rooting_jailbreak_indicator',
        'Customer_mobile_roaming_indicator', 'Customer_VPN_Indicator',
        'Customer_flag_terminal_malicious_behavior_1',
        'Customer_flag_terminal_malicious_behavior_2',
        'Customer_flag_terminal_malicious_behavior_3',
        'Customer_flag_terminal_malicious_behavior_4',
        'Customer_flag_terminal_malicious_behavior_5',
        'Customer_flag_terminal_malicious_behavior_6',
        'Customer_inquery_atm_limit', 'Customer_increase_atm_limit',
        'Account_indicator_release_limit_excess',
        'Account_indicator_Openbanking', 'Account_release_suspention',
        'Error_Code', 'Transaction_Failure_Status', 'Type_General_Automatic',
        'Unused_terminal_status', 'Flag_deposit_more_than_tenMillion',
        'Unused_account_status', 'Recipient_account_suspend_status',
        'First_time_iOS_by_vulnerable_user']
        '''
        bin_indexes = [
        1, 3, 4, 5, 6,
        7, 8, 9, 11, 12,
        13, 14, 15, 16, 17,
        18, 22, 24, 26, 34,
        35, 36, 42, 43, 44,
        45, 48
        ]
        
        '''
        ['Customer_credit_rating', 'Customer_loan_type', 'Account_account_type',
        'Channel', 'Operating_System', 'Access_Medium', 'Distance']
        '''
        cat_indexes = [2, 10, 19, 33, 37]

        '''
        ['Customer_Birthyear', 'Account_initial_balance', 'Account_balance',
        'Account_amount_daily_limit',
        'Account_remaining_amount_daily_limit_exceeded',
        'Account_one_month_max_amount', 'Account_one_month_std_dev',
        'Account_dawn_one_month_max_amount', 'Account_dawn_one_month_std_dev',
        'Transaction_Amount', 'Transaction_num_connection_failure',
        'Another_Person_Account', 'Time_difference',
        'Number_of_transaction_with_the_account',
        'Transaction_history_with_the_account']
        '''
        int_indexes = [0, 20, 21, 23, 25, 27, 28, 29, 30, 31, 38, 41, 46, 47]
        
        self.bin_indexes, self.cat_indexes, self.int_indexes = bin_indexes, cat_indexes, int_indexes
        self.encode_labels()


    def encode_labels(self):
        le_col = self.temp_df.select_dtypes(include='object').columns.tolist()
        df_encoded = self.temp_df.copy()

        for column in le_col:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(self.temp_df[column])
            self.label_encoders[column] = le

        self.df_encoded = df_encoded

    def generate_synthetic_data(self):
        fraud_types = self.df_encoded['Fraud_Type'].unique()
        all_synthetic_data = pd.DataFrame()
        N_SAMPLE = self.args.n_sample

        for fraud_type in tqdm(fraud_types):
            subset = self.df_encoded[self.df_encoded["Fraud_Type"] == fraud_type]
            subset = subset.sample(n=N_SAMPLE, random_state=self.seed)

            X = subset.drop("Fraud_Type", axis=1).values
            y = subset["Fraud_Type"].values

            forest_model = ForestDiffusionModel(
                X,
                label_y=y,
                n_t=self.args.n_t,
                duplicate_K=self.args.duplicate_K,
                bin_indexes=self.bin_indexes,
                cat_indexes=self.cat_indexes,
                int_indexes=self.int_indexes,
                diffusion_type=self.args.diffusion_type, # "flow", "vp"
                n_jobs=-1,
                n_batch=1,
                seed=self.seed
            )

            Xy_fake = forest_model.generate(batch_size=N_SAMPLE)
            syn_df = pd.DataFrame(Xy_fake, columns=self.df_encoded.columns)
            all_synthetic_data = pd.concat([all_synthetic_data, syn_df], ignore_index=True)

        self.all_synthetic_data = all_synthetic_data

    def decode_labels(self):
        df_decoded = self.all_synthetic_data.copy()
        for column in self.label_encoders:
            le = self.label_encoders[column]
            df_decoded[column] = le.inverse_transform(self.all_synthetic_data[column].astype(int))
        
        df_decoded['Time_difference'] = pd.to_timedelta(df_decoded['Time_difference'], unit='s')
        self.df_decoded = df_decoded

    def save_synthetic_data(self, output_dir='syn_data'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, 'forestdiffusion.csv')
        self.df_decoded.to_csv(output_path, index=False)

def main():
    parser = argparse.ArgumentParser(description="syn")
    parser.add_argument('--n_sample', default=100, type=float)
    parser.add_argument('--n_t', default=50, type=int)
    parser.add_argument('--duplicate_K', default=100, type=int)
    parser.add_argument('--diffusion_type', default="flow", type=str)
    parser.add_argument('--seed', default=826, type=int)
    args = parser.parse_args('')

    generator = ForestDiffusionGenerator(args)
    generator.load_data()
    generator.preprocess_data()
    generator.generate_synthetic_data()
    generator.decode_labels()
    generator.save_synthetic_data()

if __name__ == "__main__":
    main()