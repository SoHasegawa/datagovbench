import argparse
import datetime
import pickle
import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, DistilBertModel

from src.fti.fti.configs import TwoStepConfigs
from src.fti.fti.extract_statistical_features import FeaturizeFile
from src.fti.fti.utils import is_num


class FTIModel:
    def __init__(self,
                 dataset_train_path,
                 dataset_test_path,
                 session_path,
                 first_method,
                 second_method,
                 pretrain_session=None,
                 txt_feature_only=False,
                 original_numerical=False,
                 pca=False,
                 top_n=30):
        self.train_path = dataset_train_path
        self.test_path = dataset_test_path
        self.first_text_method = first_method
        self.second_text_method = second_method
        self.session_path = session_path
        self.txt_feature_only = txt_feature_only
        self.top_n = top_n
        self.original_numerical = original_numerical
        self.pca = pca
        self.map_to_unite = TwoStepConfigs().map_to_unite

        # Model initialization
        if pretrain_session is None:
            self.model_first = LGBMClassifier()
            self.model_second = LGBMClassifier()
        else:
            self.model_first = pickle.load(open(f"src/fti/sessions/{pretrain_session}/first_model.pickle", "rb"))
            self.model_second = pickle.load(open(f"src/fti/sessions/{pretrain_session}/second_model.pickle", "rb"))

        # Text feature extractor initialization
        if pretrain_session is None:
            if self.first_text_method == "bow":
                self.name_vectorizer_first = CountVectorizer()
                self.cell_vectorizer_first = CountVectorizer()
            elif self.first_text_method == "tfidf":
                self.name_vectorizer_first = TfidfVectorizer()
                self.cell_vectorizer_first = TfidfVectorizer()
            if self.second_text_method == "bow":
                self.name_vectorizer_second = CountVectorizer()
                self.cell_vectorizer_second = CountVectorizer()
            elif self.second_text_method == "tfidf":
                self.name_vectorizer_second = TfidfVectorizer()
                self.cell_vectorizer_second = TfidfVectorizer()
            if self.pca:
                self.pca_model = PCA(n_components=70)
        else:
            if self.first_text_method == "bow" or self.first_text_method == "tfidf":
                self.name_vectorizer_first = pickle.load(open(f"src/fti/sessions/{pretrain_session}/name_vectorizer_first.pickle", "rb"))
                self.cell_vectorizer_first = pickle.load(open(f"src/fti/sessions/{pretrain_session}/cell_vectorizer_first.pickle", "rb"))
            if self.second_text_method == "bow" or self.second_text_method == "tfidf":
                self.name_vectorizer_second = pickle.load(open(f"src/fti/sessions/{pretrain_session}/name_vectorizer_second.pickle", "rb"))
                self.cell_vectorizer_second = pickle.load(open(f"src/fti/sessions/{pretrain_session}/cell_vectorizer_second.pickle", "rb"))
            if self.pca:
                self.pca_model = pickle.load(open(f"src/fti/sessions/{pretrain_session}/pca.pickle", "rb"))

        self.columns_first = ["numeric", "categorical", "datetime", "sentence", "url", "embed",
                              "list", "not-generalizable", "unit", "range", "sign"]
        self.columns_second = ["datetime", "sentence", "url", "list", "embed", "unit", "sign", "range"]

        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased").to(self.device).eval()

    @staticmethod
    def process_text(__dataset: pd.DataFrame, method="bow", numerical_original=False) -> pd.DataFrame:
        for _col in ["Attribute_name", "sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]:
            process_text = [str(t).lower() for t in __dataset[_col]]
            if method == "bow" or method == "tfidf":
                #if not numerical_original:
                process_text = [re.sub(r'\d+', 'num', t) for t in process_text]
            __dataset[_col] = process_text

        return __dataset

    @staticmethod
    def process_text_bert(__dataset: pd.DataFrame, numerical_original=False) -> pd.DataFrame:
        for _col in ["Attribute_name", "sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]:
            process_text = [str(t) for t in __dataset[_col]]
            if not numerical_original:
                process_text = [re.sub(r'\d+', 'num', t) for t in process_text]
            __dataset[_col] = process_text

        return __dataset

    def extract_word_embedding(self, df: pd.DataFrame) -> np.array:
        def construct_sentence(attribute, sample_values):
            return f"Name of the column of tabular data is {attribute}, and sample values are {sample_values[0]}, {sample_values[1]}, {sample_values[2]}, {sample_values[3]}, and {sample_values[4]}."
        attributes_list = df["Attribute_name"].values.tolist()
        samples_list = df[["sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]].values.tolist()

        ys = []

        for attribute, samples in zip(attributes_list, samples_list):
            x = construct_sentence(attribute, samples)
            inputs = self.tokenizer(x,
                                    return_tensors="pt",
                                    max_length=512,
                                    padding='max_length',
                                    truncation=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # First token is utilized as the embedding
            y = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
            ys.append(y.reshape(-1))

            del y

        ys = np.array(ys)

        return ys

    def _prepare_first_stage(self, df, stage="train", add_is_numeric=False):
        df["Attribute_name"] = df["Attribute_name"].fillna("column")
        df["sample_1"] = df["sample_1"].fillna("nothing")
        df["sample_2"] = df["sample_2"].fillna("nothing")
        df["sample_3"] = df["sample_3"].fillna("nothing")
        df["sample_4"] = df["sample_4"].fillna("nothing")
        df["sample_5"] = df["sample_5"].fillna("nothing")

        if add_is_numeric:
            # Add float type or not
            sample_values = df[["sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]].values.tolist()
            add_is_numeric_list = []
            for sv_list in sample_values:
                numeric_form_list = []
                for sv in sv_list:
                    if isinstance(sv, datetime.date):
                        numeric_form_list.append(sv)
                        continue
                    if is_num(sv):
                        numeric_form_list.append(float(sv))
                    else:
                        numeric_form_list.append(sv)
                add_is_numeric_list.append(is_numeric_dtype(np.array(numeric_form_list)))

            df["is_numeric"] = add_is_numeric_list

        if self.first_text_method == "bow" or self.first_text_method == "tfidf":
            df = self.process_text(df, self.original_numerical)

            sample_values1 = df["sample_1"].values.tolist()
            sample_values2 = df["sample_2"].values.tolist()
            sample_values = sample_values1 + sample_values2
            if stage == "train":
                self.cell_vectorizer_first.fit(sample_values)
            sv1_vector = self.cell_vectorizer_first.transform(sample_values1).toarray()
            sv2_vector = self.cell_vectorizer_first.transform(sample_values2).toarray()

            name_values = df["Attribute_name"].values.tolist()
            if stage == "train":
                self.name_vectorizer_first.fit(name_values)
            name_vector = self.name_vectorizer_first.transform(name_values).toarray()

        elif self.first_text_method == "bert":
            txt_feature = self.extract_word_embedding(df)

        df["std_dev"] = df["std_dev"].fillna(0)

        columns = list(df.columns)
        columns.remove("Attribute_name")
        columns.remove("sample_1")
        columns.remove("sample_2")
        columns.remove("sample_3")
        columns.remove("sample_4")
        columns.remove("sample_5")
        if "subcategory" in columns:
            columns.remove("subcategory")
        remains = df[columns].values

        if self.first_text_method == "bow" or self.first_text_method == "tfidf":
            if self.pca:
                txt_vectors = np.concatenate([name_vector, sv1_vector, sv2_vector], axis=1)
                if stage == "train":
                    txt_vectors = self.pca_model.fit_transform(txt_vectors)
                else:
                    txt_vectors = self.pca_model.transform(txt_vectors)
                x = np.concatenate([txt_vectors, remains], axis=1)
            else:
                x = np.concatenate([name_vector, sv1_vector, sv2_vector, remains], axis=1)
        elif self.first_text_method == "bert":
            x = np.concatenate([txt_feature, remains], axis=1)

        return df, x

    def _prepare_second_stage(self, df, stage="train"):
        df["Attribute_name"] = df["Attribute_name"].fillna("column")
        df["sample_1"] = df["sample_1"].fillna("nothing")
        df["sample_2"] = df["sample_2"].fillna("nothing")
        df["sample_3"] = df["sample_3"].fillna("nothing")
        df["sample_4"] = df["sample_4"].fillna("nothing")
        df["sample_5"] = df["sample_5"].fillna("nothing")

        sample_values = df[["sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]].values.tolist()
        num_number_characters = []
        num_separated_tokens = []
        comma_include = []
        for sv_list in sample_values:
            number_characters = 0
            separated_tokens = 0
            comma = 0
            for sv in sv_list:
                sv = str(sv)
                for s in sv:
                    if s in "0123456789": number_characters += 1
                separated_tokens += len(re.findall(r'([-]?[0-9]+\.?[0-9]*)', sv))
                if "," in sv: comma += 1
            num_number_characters.append(number_characters/5)
            num_separated_tokens.append(separated_tokens/5)
            comma_include.append(comma)

        df["num_number_characters"] = num_number_characters
        df["num_separated_tokens"] = num_separated_tokens
        df["comma_include"] = comma_include

        if self.second_text_method == "bow" or self.second_text_method == "tfidf":
            df = self.process_text(df, self.original_numerical)

            sample_values1 = df["sample_1"].values.tolist()
            sample_values2 = df["sample_2"].values.tolist()
            sample_values = sample_values1 + sample_values2
            if stage == "train":
                self.cell_vectorizer_second.fit(sample_values)
            sv1_vector = self.cell_vectorizer_second.transform(sample_values1).toarray()
            sv2_vector = self.cell_vectorizer_second.transform(sample_values2).toarray()

            name_values = df["Attribute_name"].values.tolist()
            if stage == "train":
                self.name_vectorizer_second.fit(name_values)
            name_vector = self.name_vectorizer_second.transform(name_values).toarray()

        elif self.second_text_method == "bert":
            df = self.process_text_bert(df, self.original_numerical)
            txt_feature = self.extract_word_embedding(df)

        df["std_dev"] = df["std_dev"].fillna(0)

        columns = list(df.columns)
        columns.remove("Attribute_name")
        columns.remove("sample_1")
        columns.remove("sample_2")
        columns.remove("sample_3")
        columns.remove("sample_4")
        columns.remove("sample_5")
        columns.remove("total_vals")
        columns.remove("num_nans")
        columns.remove("%_nans")
        columns.remove("num_of_dist_val")
        columns.remove("%_dist_val")
        columns.remove("mean")
        columns.remove("std_dev")
        columns.remove("min_val")
        columns.remove("max_val")
        if "subcategory" in columns:
            columns.remove("subcategory")
        remains = df[columns].values

        if self.second_text_method == "bow" or self.second_text_method == "tfidf":
            x = np.concatenate([name_vector, sv1_vector, sv2_vector, remains], axis=1)
        elif self.second_text_method == "bert":
            if self.txt_feature_only:
                x = txt_feature
            else:
                x = np.concatenate([txt_feature, remains], axis=1)

        return df, x

    @staticmethod
    def _filter(df):
        #df_drop = df[(df["subcategory"] == "categorical") & (df["%_dist_val"] >= 50)]
        df_drop = df[(df["subcategory"] == "numeric") & (df["mean"] == 0) & (df["std_dev"] == 0) & (df["min_val"] == 0) & (df["max_val"] == 0)]
        df = df.drop(df_drop.index)

        return df

    def _first_step(self):
        df_train = pd.read_csv(self.train_path).copy()
        df_test = pd.read_csv(self.test_path).copy()

        df_train = self._filter(df_train)

        extract_columns = self.columns_first
        df_train = df_train.query('subcategory in @extract_columns')
        df_test = df_test.query('subcategory in @extract_columns')

        df_train, x_train = self._prepare_first_stage(df_train, stage="train", add_is_numeric=True)
        df_test, x_test = self._prepare_first_stage(df_test, stage="test", add_is_numeric=True)
        mapping = {"numeric": 0, "categorical": 1, "datetime": 2, "sentence": 2, "url": 2,
                   "embed": 2, "list": 2, "unit": 2, "range": 2,
                   "sign": 2, "not-generalizable": 3, "formatted-id": 2}
        y_train = np.array(df_train["subcategory"].map(mapping).values.tolist())
        y_test = np.array(df_test["subcategory"].map(mapping).values.tolist())

        self.model_first.fit(x_train, y_train)
        y_pred = self.model_first.predict(x_test)

        print(self.model_first.score(x_test, y_test))
        print(confusion_matrix(y_test, y_pred))

    def _second_step(self):
        df_train = pd.read_csv(self.train_path).copy()
        df_test = pd.read_csv(self.test_path).copy()

        extract_columns = self.columns_second
        df_train = df_train.query('subcategory in @extract_columns')
        df_test = df_test.query('subcategory in @extract_columns')

        df_train, x_train = self._prepare_second_stage(df_train, stage="train")
        df_test, x_test = self._prepare_second_stage(df_test, stage="test")
        mapping = {"datetime": 0, "sentence": 1, "url": 2, "list": 3, "embed": 4,
                   "unit": 5, "range": 6, "sign": 7, "formatted-id": 8}
        y_train = np.array(df_train["subcategory"].map(mapping).values.tolist())
        y_test = np.array(df_test["subcategory"].map(mapping).values.tolist())

        self.model_second.fit(x_train, y_train)
        y_pred = self.model_second.predict(x_test)

        print(self.model_second.score(x_test, y_test))
        print(confusion_matrix(y_test, y_pred))

    def _save(self):
        pickle.dump(self.model_first, open(f"src/fti/sessions/{self.session_path}/first_model.pickle", "wb"))
        pickle.dump(self.model_second, open(f"src/fti/sessions/{self.session_path}/second_model.pickle", "wb"))

        if self.first_text_method == "bow" or self.first_text_method == "tfidf":
            pickle.dump(self.name_vectorizer_first, open(f"src/fti/sessions/{self.session_path}/name_vectorizer_first.pickle", "wb"))
            pickle.dump(self.cell_vectorizer_first, open(f"src/fti/sessions/{self.session_path}/cell_vectorizer_first.pickle", "wb"))
        if self.second_text_method == "bow" or self.second_text_method == "tfidf":
            pickle.dump(self.name_vectorizer_second, open(f"src/fti/sessions/{self.session_path}/name_vectorizer_second.pickle", "wb"))
            pickle.dump(self.cell_vectorizer_second, open(f"src/fti/sessions/{self.session_path}/cell_vectorizer_second.pickle", "wb"))

        if self.pca:
            pickle.dump(self.pca_model, open(f"src/fti/sessions/{self.session_path}/pca.pickle", "wb"))

    def train(self):
        self._first_step()
        print("First stage is finished.")
        self._second_step()
        print("Second stage is finished.")
        self._save()

    @staticmethod
    def candidate_generation(y_proba: np.array,
                             first_stage=False) -> list[list[int]]:
        candidates = []
        probabilities = []
        num_candidates = 1

        for ft_order, proba_column in enumerate(y_proba):
            candidate = []
            fti_probability = []
            for i, proba in enumerate(proba_column):
                threshold = 1 / proba_column.shape[0]
                if proba > threshold:
                    candidate.append(i)
                    fti_probability.append(proba)
            candidates.append(candidate)
            probabilities.append(fti_probability)
            num_candidates *= len(candidate)

        fti_candidates = [list(candidate) for candidate in product(*candidates)]
        fti_probabilities = []
        for fti_candidate in fti_candidates:
            fti_probability = []
            for i, c in enumerate(fti_candidate):
                fti_probability.append(y_proba[i, c])
            fti_probabilities.append(fti_probability)

        return fti_candidates, fti_probabilities

    @staticmethod
    def _detect_mixed_type(df: pd.DataFrame, remaining_features: pd.DataFrame) -> pd.DataFrame:
        """Detect whether the column is mixed with string and numerical values
           If detected, string values are prioritized to be used as the sample values
        Args:
            df (pd.DataFrame): original dataframe
            remaining_features (pd.DataFrame): SortingHat feature
        Returns:
            pd.DataFrame: Rewritten SortingHat features
        """

        for i, column in enumerate(df.columns):
            series = df[column]
            series = series.replace([np.inf, -np.inf], np.nan)
            original_series = series.loc[series.notnull()]
            series = pd.to_numeric(original_series, errors="coerce")
            ratio = series.notnull().sum() / len(series.index)
            is_mixed = ratio > 0 and ratio < 1
            if is_mixed:
                count = 0
                for cell in original_series.values.tolist():
                    if not is_num(cell):
                        count += 1
                        remaining_features.loc[i, f"sample_{count}"] = cell
                        if count == 5: break

        return remaining_features

    def predict(self, df):
        first_df = df.copy()
        remain_features = FeaturizeFile(df.copy())
        remain_features = self._detect_mixed_type(df.copy(), remain_features.copy())
        percent_nans = remain_features["%_nans"].values.tolist()
        percent_dist_val = remain_features["%_dist_val"].values.tolist()
        remain_features = remain_features.drop(["%_nans", "%_dist_val"], axis=1)
        remain_features.insert(3, "%_nans", percent_nans)
        remain_features.insert(5, "%_dist_val", percent_dist_val)

        _, x = self._prepare_first_stage(remain_features, stage="test", add_is_numeric=True)
        y_first = self.model_first.predict(x)

        remain_columns = []
        remain_indices = []
        for i, (ann, column) in enumerate(zip(y_first, first_df.columns)):
            if ann == 2:
                remain_columns.append(column)
                remain_indices.append(i)
            for i, ann in enumerate(y_first):
                if ann == 3: y_first[i] = 7

        if len(remain_columns) == 0: return y_first

        second_df = df.copy()
        text_df = second_df[remain_columns]

        remain_features = FeaturizeFile(text_df)
        remain_features = self._detect_mixed_type(text_df.copy(), remain_features.copy())
        percent_nans = remain_features["%_nans"].values.tolist()
        percent_dist_val = remain_features["%_dist_val"].values.tolist()
        remain_features = remain_features.drop(["%_nans", "%_dist_val"], axis=1)
        remain_features.insert(3, "%_nans", percent_nans)
        remain_features.insert(5, "%_dist_val", percent_dist_val)

        _, x = self._prepare_second_stage(remain_features, stage="test")
        y = self.model_second.predict(x)

        for i, index in enumerate(remain_indices):
            y_first[index] = self.map_to_unite[y[i]]

        return y_first

    def predict_multiple(self, df):
        # Prepare input features
        first_df = df.copy()
        remain_features = FeaturizeFile(df)
        remain_features = self._detect_mixed_type(df, remain_features.copy())
        percent_nans = remain_features["%_nans"].values.tolist()
        percent_dist_val = remain_features["%_dist_val"].values.tolist()
        remain_features = remain_features.drop(["%_nans", "%_dist_val"], axis=1)
        remain_features.insert(3, "%_nans", percent_nans)
        remain_features.insert(5, "%_dist_val", percent_dist_val)

        # First model
        _, x = self._prepare_first_stage(remain_features.copy(), stage="test", add_is_numeric=True)
        y_first = self.model_first.predict(x)
        y_proba = self.model_first.predict_proba(x)
        best_proba = np.max(y_proba, axis=1).tolist()

        # Second model
        _, x = self._prepare_second_stage(remain_features.copy(), stage="test")

        y_second = self.model_second.predict(x)
        y_second_proba = self.model_second.predict_proba(x)

        first_fti_candidates, first_fti_probabilities = self.candidate_generation(y_proba,
                                                                                  first_stage=True)

        final_fti_candidates = []
        final_fti_probabilities = []

        for fti_candidate, fti_probability in zip(first_fti_candidates, first_fti_probabilities):
            remain_columns = []
            remain_indices = []
            for i, (ann, column) in enumerate(zip(fti_candidate, first_df.columns)):
                if ann == 2:
                    remain_columns.append(column)
                    remain_indices.append(i)
                    fti_candidate[i] = 3
                if ann == 3:
                    fti_candidate[i] = 7

            if len(remain_columns) == 0:
                final_fti_candidates.append(fti_candidate.copy())
                final_fti_probabilities.append(sum(fti_probability.copy()))
                continue

            # Second model
            y = y_second[remain_indices]
            y_proba = y_second_proba[remain_indices]
            second_fti_candidates, second_fti_probabilities = self.candidate_generation(y_proba)

            for second_fti_candidate, second_fti_probability in zip(second_fti_candidates, second_fti_probabilities):
                fti_candidate_copy = fti_candidate.copy()
                fti_probability_copy = fti_probability.copy()
                for i, index in enumerate(remain_indices):
                    fti_candidate_copy[index] = self.map_to_unite[second_fti_candidate[i]]
                    fti_probability_copy[index] = second_fti_probability[i]

                final_fti_candidates.append(fti_candidate_copy)
                final_fti_probabilities.append(sum(fti_probability_copy))

        if self.top_n is not None:
            final_fti_candidates = np.array(final_fti_candidates)
            final_fti_probabilities = np.array(final_fti_probabilities)
            top_idx = np.argsort(-final_fti_probabilities)
            final_fti_candidates = final_fti_candidates[top_idx][:self.top_n]

        return final_fti_candidates
