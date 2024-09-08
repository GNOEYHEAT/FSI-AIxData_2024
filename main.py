import os
import argparse

from LLM_Masking import LLM_Masking
from FraudDetectionModel import FraudDetectionModel


def main(args, **kwargs):
    if args.masking == True:
        generator = LLM_Masking(
            model_id="yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
            condition_file='data/데이터_명세_및_생성조건.xlsx',
            train_file=args.train_path
        )
        generator.generate_synthetic_data()
        print("masking done")
    
    if args.mode == 'gen_syn_meta_data':
        FraudDetectionModel.gen_syn_meta_data(frac=args.frac, scaler=args.scaler, cv=args.cv, seed=args.seed, train_path=args.train_path, test_path=args.test_path, syn_data_path=args.syn_data_path, **kwargs)
        
    elif args.mode == 'gen_raw_meta_data':
        FraudDetectionModel.gen_raw_meta_data(frac=args.frac, scaler=args.scaler, cv=args.cv, seed=args.seed, train_path=args.train_path, test_path=args.test_path, syn_data_path=args.syn_data_path, **kwargs)
    
    if args.ensemble == True:
        FraudDetectionModel.ensemble(frac=args.frac, scaler=args.scaler, cv=args.cv, seed=args.seed, train_path=args.train_path, test_path=args.test_path, syn_data_path=args.syn_data_path, **kwargs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--frac', default=0.01, type=float)
    parser.add_argument('--scaler', default="standard", type=str) # standard or minmax or robust
    parser.add_argument('--cv', default=10, type=int)
    parser.add_argument('--seed', default=826, type=int)
    
    parser.add_argument('--train_path', default="data/train.csv", type=str)
    parser.add_argument('--test_path', default="data/test.csv", type=str)
    parser.add_argument('--syn_data_path', default="syn_data/forestdiffusion.csv", type=str)
    
    parser.add_argument('--masking', default=True, type=bool) # True or False
    parser.add_argument('--mode', default="gen_raw_meta_data", type=str) # gen_raw_meta_data or gen_syn_meta_data
    parser.add_argument('--ensemble', default=False, type=bool) # True or False
    
    
    args = parser.parse_args('')
    config = vars(args)
    print('------------ Options -------------')
    for k, v in sorted(config.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    
    main(args)