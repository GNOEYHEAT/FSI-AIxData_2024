import os
import zipfile
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import xgboost as xgb
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import warnings
import random
import torch

# To ignore all warnings
warnings.filterwarnings('ignore')

SEED = 826

class CTGANGenerator:
    
    def __init__(self, train_path, test_path, n_sample=100, n_cls_per_gen=1000, seed=SEED):
        self.train_path = train_path
        self.test_path = test_path
        self.n_sample = n_sample
        self.n_cls_per_gen = n_cls_per_gen
        self.seed = seed
        self.train_data = None
        self.test_data = None
        self.synthetic_data = pd.DataFrame()
        
        self.set_seeds(seed)
        self.load_data()

    def set_seeds(self, seed):
        """ Set random seeds for reproducibility """
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_data(self):
        """ Load the train and test datasets """
        self.train_data = pd.read_csv(self.train_path).drop(columns="ID")
        self.test_data = pd.read_csv(self.test_path)
        
    def handle_outliers(self, series, n_std=3):
        """ Handle outliers using z-score method """
        mean = series.mean()
        std = series.std()
        z_scores = np.abs(stats.zscore(series))
        return series.mask(z_scores > n_std, mean)
    
    def preprocess_data(self):
        """ Preprocess the train dataset by handling time differences """
        # Convert Time_difference column to seconds
        self.train_data['Time_difference_seconds'] = pd.to_timedelta(self.train_data['Time_difference']).dt.total_seconds()
        # Handle outliers in Time_difference_seconds
        self.train_data['Time_difference_seconds'] = self.handle_outliers(self.train_data['Time_difference_seconds'])
        
    def generate_synthetic_data(self):
        """ Generate synthetic data for each fraud type """
        fraud_types = self.train_data['Fraud_Type'].unique()

        # For each fraud type, generate synthetic data
        for fraud_type in tqdm(fraud_types):
            subset = self.train_data[self.train_data["Fraud_Type"] == fraud_type]

            # Sample N_SAMPLE rows for each Fraud_Type
            subset = subset.sample(n=self.n_sample, random_state=self.seed)
            
            # Drop original Time_difference column
            subset = subset.drop('Time_difference', axis=1)

            # Create metadata for SDV
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(subset)
            metadata.set_primary_key(None)

            # Define column data types
            column_sdtypes = {
                'Account_initial_balance': 'numerical',
                'Account_balance': 'numerical',
                'Customer_identification_number': 'categorical',  
                'Customer_personal_identifier': 'categorical',
                'Account_account_number': 'categorical',
                'IP_Address': 'ipv4_address',  
                'Location': 'categorical',
                'Recipient_Account_Number': 'categorical',
                'Fraud_Type': 'categorical',
                'Time_difference_seconds': 'numerical',
                'Customer_Birthyear': 'numerical'
            }

            for column, sdtype in column_sdtypes.items():
                metadata.update_column(column_name=column, sdtype=sdtype)

            # Train the CTGAN synthesizer
            synthesizer = CTGANSynthesizer(metadata)
            synthesizer.fit(subset)

            # Generate synthetic data
            synthetic_subset = synthesizer.sample(num_rows=self.n_cls_per_gen)

            # Handle outliers in generated data
            synthetic_subset['Time_difference_seconds'] = self.handle_outliers(synthetic_subset['Time_difference_seconds'])
            
            # Convert Time_difference_seconds back to timedelta
            synthetic_subset['Time_difference'] = pd.to_timedelta(synthetic_subset['Time_difference_seconds'], unit='s')

            # Drop Time_difference_seconds column
            synthetic_subset = synthetic_subset.drop('Time_difference_seconds', axis=1)

            # Append generated data to all_synthetic_data
            self.synthetic_data = pd.concat([self.synthetic_data, synthetic_subset], ignore_index=True)
            
    def save_synthetic_data(self, output_path):
        """ Save the synthetic data to a CSV file """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.synthetic_data.to_csv(output_path, index=False)
        
    def run(self, output_path):
        """ Full pipeline to preprocess, generate synthetic data, and save it """
        self.preprocess_data()
        self.generate_synthetic_data()
        self.save_synthetic_data(output_path)


# Example usage:
if __name__ == "__main__":
    generator = CTGANGenerator(train_path="data/train.csv", test_path="data/test.csv")
    generator.run(output_path="syn_data/ctgan.csv")
