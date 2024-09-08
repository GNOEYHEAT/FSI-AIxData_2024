import os
import re
import random
import pandas as pd
import numpy as np
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import warnings

warnings.filterwarnings('ignore')


class LLM_Masking:
    def __init__(self, model_id, condition_file, train_file, seed=826):
        self.model_id = model_id
        self.condition_file = condition_file
        self.train_file = train_file
        self.seed = seed
        print('LLM Masking Start')
        self._set_seed()
        self._load_data()
        self._initialize_model()

    def _set_seed(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _load_data(self):
        self.train_all = pd.read_csv(self.train_file)
        self.train = self.train_all.drop(columns="ID")
        self.condition = pd.read_excel(self.condition_file)
        self.condition = self.condition.drop(columns=self.condition.columns[0:2])
        self.condition.columns = self.condition.iloc[0]
        self.condition = self.condition[1:]

    def _initialize_model(self):
        compute_dtype = getattr(torch, "bfloat16")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype="float16", device_map="auto", quantization_config=bnb_config)

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.7,
            return_full_text=False,
            max_new_tokens=512,
            do_sample=True
        )

        self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        self.prompt_template = PromptTemplate(
            input_variables=["data", "condition", "example"],
            template="""
                당신은 주어진 예시를 참조하여 새로운 값을 생성해주는 AI Assistant입니다.
                설정한 조건에 맞게 새로운 값을 생성해주세요.
                
                다음 조건에 맞게 새로운 {data}을(를) 생성하세요.
                {condition}
                
                다음 생성 예시를 참조하세요.
                {example}
                
                위 생성 예시와 완전히 다른 값으로 생성해주세요.
                조건에 명시된 숫자를 무조건 지켜주세요.
                절대로 설명, 참고 등의 추가적인 생성은 하지말고, 새로운 값만 짧게 생성해주세요.
                
                새로운 값:
                """
        )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            output_parser=StrOutputParser()
        )

    @staticmethod
    def _is_valid_customer_personal_identifier(sequence):
        pattern = r'^[가-힣]{1,5}$'
        return bool(re.match(pattern, sequence))

    @staticmethod
    def _is_valid_customer_identification_number(sequence):
        pattern = r'^[a-zA-Z]{6}-[a-zA-Z]{7}$'
        return bool(re.match(pattern, sequence))

    @staticmethod
    def _is_valid_alphabet_sequence(sequence):
        return sequence.isalpha() and len(sequence) == 10

    @staticmethod
    def _clean_list(strings):
        cleaned_list = []
        for string in strings:
            cleaned_string = string.replace("\n", "").replace("\t", "").replace("\r", "")
            cleaned_string = ' '.join(cleaned_string.split())
            if cleaned_string:
                cleaned_list.append(cleaned_string)
        return cleaned_list

    def _get_list(self, feature, num_iterations):
        filtered_row = self.condition[self.condition['항목명'] == feature]

        gen_data = filtered_row['항목 설명'].values[0]
        gen_condition = filtered_row['생성 조건 '].values[0]
        gen_example = filtered_row['데이터 예시'].values[0]

        results = []
        unique_values = set(self.train[feature].unique())

        while len(results) < num_iterations:
            result = self.chain.run({"data": gen_data, "condition": gen_condition, "example": gen_example})

            if feature == 'Customer_personal_identifier' and not self._is_valid_customer_personal_identifier(result):
                continue

            if feature == 'Customer_identification_number' and not self._is_valid_customer_identification_number(result):
                continue

            if feature == 'Account_account_number' and not self._is_valid_alphabet_sequence(result):
                continue

            if result not in unique_values:
                results.append(result)
                unique_values.add(result)

            if len(results) == num_iterations:
                break

        return results

    def generate_synthetic_data(self, num_iterations=3):
        print('LLM Masking... ')
        print('Customer_personal_identifier Masking... ')
        names = self._get_list('Customer_personal_identifier', num_iterations)
        names = self._clean_list(names)
        
        print('Customer_identification_number Masking... ')
        idf_number = self._get_list('Customer_identification_number', num_iterations)
        idf_number = self._clean_list(idf_number)

        print('Account_account_number Masking... ')
        account_number = self._get_list('Account_account_number', num_iterations)
        account_number = self._clean_list(account_number)

        print('IP_Address Masking... ')
        ip = self._get_list('IP_Address', 3 * num_iterations)
        ip = self._clean_list(ip)

        print('MAC_Address Masking... ')
        mac = self._get_list('MAC_Address', 3 * num_iterations)
        mac = self._clean_list(mac)

        print('Location Masking... ')
        loc = self._get_list('Location', 3 * num_iterations)
        loc = self._clean_list(loc)

        syn_person = pd.DataFrame(columns=['Customer_personal_identifier', 'Customer_identification_number', 'Account_account_number'])

        for i in range(len(names)):
            syn_person.loc[i] = {
                'Customer_personal_identifier': names[i],
                'Customer_identification_number': idf_number[i],
                'Account_account_number': account_number[i]
            }

        syn_set = pd.read_csv("syn_data/ctgan.csv")
        for i in range(len(syn_set)):
            random_index = np.random.choice(syn_person.index)
            syn_set.loc[i, syn_person.columns] = syn_person.loc[random_index]

        syn_set['IP_Address'] = np.random.choice(ip, size=len(syn_set))
        syn_set['MAC_Address'] = np.random.choice(mac, size=len(syn_set))
        syn_set['Location'] = np.random.choice(loc, size=len(syn_set))

        syn_set = syn_set[self.train.columns]
        syn_set.to_csv('syn_data/ctgan_syn_submission.csv', index=False)
        print('LLM Masking Done')
        print('LLM Masked Data saved successfully.')


#Example usage:
# if __name__ == "__main__":
#     generator = LLM_Masking(
#         model_id="yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
#         condition_file='data/데이터_명세_및_생성조건.xlsx',
#         train_file='data/train.csv'
#     )
#     generator.generate_synthetic_data()
