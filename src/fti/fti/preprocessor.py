import math
import re
import string
from collections import defaultdict
from itertools import cycle
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from src.fti.fti.configs import PreprocessorConfigs
from src.fti.fti.jptime.jptime import from_str
from src.fti.fti.llm import LargeLanguageModel
from src.fti.fti.utils import is_num


class NestedTransformers:
    def __init__(self, transformers):
        self.transformers = transformers

    def transform(self, X):
        for transformer, column in self.transformers:
            X = transformer.transform(X, column)

        return X


class TextTfIdfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, na_const='NA', pre_op=1):
        self.na_const = na_const
        self._max_features = 100
        self._vectorizer = TfidfVectorizer(max_features=self._max_features, stop_words=None, ngram_range=(1, 1))
        self.pre_op = pre_op

    def process_text_single_col(self, __dataset):
        process_text = [str(t).lower() for t in __dataset]
        # strip all punctuation
        table = str.maketrans('', '', string.punctuation)
        process_text = [t.translate(table) for t in process_text]
        # convert all numbers in text to 'num'
        process_text = [re.sub(r'\d+', 'num', t) for t in process_text]
        __dataset = pd.Series(process_text)
        return __dataset

    def fit(self, X, y=None):
        X = self.process_text_single_col(X)

        if any(X.isnull()):
            X.fillna(self.na_const, inplace=True)
        self._vectorizer.fit(X.values.astype('U'))
        return self

    def transform(self, X, y=None):
        X = self.process_text_single_col(X)
        vect = self._vectorizer.transform(X.values.astype('U'))
        return vect.toarray()


# modify from Mljar-Supervised
# Reference: https://medium.com/codex/building-a-mixed-type-preprocessing-pipeline-with-scikit-learn-f4d90f5919fa
class DateTimeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, columns):
        return self

    @staticmethod
    def _judge_language(series):
        japanese_language = False
        for cell in series.values.tolist()[:10]:
            if isinstance(cell, str):
                if len(cell) != len(cell.encode('utf-8')):
                    japanese_language = True
                    break

        return japanese_language

    @staticmethod
    def _japanese_preprocess(X, column):
        if any(X[column] == 'None'):
            X[column] = X[column].replace('None', pd.NaT)

        date_data = {
            "year": [],
            "month": [],
            "day": [],
            "weekday": [],
            "hour": [],
            "minute": [],
            "second": []
        }
        for cell in X[column].values.tolist():
            date_format = from_str(str(cell)).to_datetime()

            date_data["year"].append(date_format.year)
            date_data["month"].append(date_format.month)
            date_data["day"].append(date_format.day)
            date_data["weekday"].append(date_format.weekday())
            date_data["hour"].append(date_format.hour)
            date_data["minute"].append(date_format.minute)
            date_data["second"].append(date_format.second)

        X[f"{column}_年"] = date_data["year"]
        X[f"{column}_月"] = date_data["month"]
        X[f"{column}_日"] = date_data["day"]
        X[f"{column}_曜日"] = date_data["weekday"]
        X[f"{column}_時"] = date_data["hour"]
        X[f"{column}_分"] = date_data["minute"]
        X[f"{column}_秒"] = date_data["second"]

        return X

    @staticmethod
    def _english_preprocess(X, column):
        if any(X[column] == 'None'):
            X[column] = X[column].replace('None', pd.NaT)

        new_X_col = pd.to_datetime(X[column], errors='coerce')

        try:
            year_values = new_X_col.dt.year
            month_values = new_X_col.dt.month
            day_values = new_X_col.dt.day
            weekday_values = new_X_col.dt.weekday
            hour_values = new_X_col.dt.hour
            minute_values = new_X_col.dt.minute
            second_values = new_X_col.dt.second
        except:
            year_values = new_X_col.apply(lambda x: 0 if pd.isnull(x) else x.year)
            month_values = new_X_col.apply(lambda x: 0 if pd.isnull(x) else x.month)
            day_values = new_X_col.apply(lambda x: 0 if pd.isnull(x) else x.day)
            weekday_values = new_X_col.apply(lambda x: 0 if pd.isnull(x) else x.weekday())
            hour_values = new_X_col.apply(lambda x: 0 if pd.isnull(x) else x.hour)
            minute_values = new_X_col.apply(lambda x: 0 if pd.isnull(x) else x.minute)
            second_values = new_X_col.apply(lambda x: 0 if pd.isnull(x) else x.second)

        X[f"{column}_year"] = year_values
        X[f"{column}_month"] = month_values
        X[f"{column}_day"] = day_values
        X[f"{column}_weekday"] = weekday_values
        X[f"{column}_hour"] = hour_values
        X[f"{column}_minute"] = minute_values
        X[f"{column}_second"] = second_values

        return X

    def transform(self, X, columns):
        for column in columns:
            try:
                if self._judge_language(X[column]):
                    X = self._japanese_preprocess(X, column)
                else:
                    X = self._english_preprocess(X, column)
            except Exception as e:
                print(e)
                new_column = f"{column}_as_sentence"
                X[new_column] = X[column]

            X.drop(column, axis=1, inplace=True)
        return X


class UrlTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, option='OrdinalEncoder', na_const='NA'):
        self.option = option
        self.na_const = na_const

        self._vectorizer = None
        if self.option == 'OrdinalEncoder':
            self._vectorizer = TfidfVectorizer()

    # 1.Domain of the URL (Domain)
    def getDomain(self, url):
        domain = urlparse(url).netloc
        # print('domain: ', domain)
        if domain:
            if re.match(r"^www.", domain):
                domain = domain.replace("www.", "")
        return domain

    def getPath(self, url):
        return urlparse(url).path

    def getLastPath(self, url):
        if urlparse(url).path:
            return urlparse(url).path.split('/')[-1]
        else:
            return self.na_const

    # 5.Gives number of '/' in URL (URL_Depth)
    def getDepth(self, url):
        depth = 0
        if urlparse(url).path:
            s = urlparse(url).path.split('/')
            for j in range(len(s)):
                if len(s[j]) != 0:
                    depth = depth + 1
        return depth

    def fit(self, X, columns):
        return self

    def transform(self, X, columns):
        for column in columns:
            if any(X[column].isnull()):
                X[column].fillna(self.na_const, inplace=True)
            domain_colname = column + "_Domain"
            path_colname = column + "_Path"
            lastpath_colname = column + "_LastPath"
            depth_colname = column + '_Depth'

            domain_colnames = []
            path_colnames = []
            lastpath_colnames = []
            depth_colnames = []

            for value in X[column].values.tolist():
                url = value
                domain = self.getDomain(url)
                if domain == "":
                    domain_colnames.append("example.com")
                    path_colnames.append("/example/example/")
                    lastpath_colnames.append("example")
                    depth_colnames.append(3)
                else:
                    domain_colnames.append(self.getDomain(url))
                    path_colnames.append(self.getPath(url))
                    lastpath_colnames.append(self.getLastPath(url))
                    depth_colnames.append(self.getDepth(url))

            datas = {
                domain_colname: domain_colnames,
                path_colname: path_colnames,
                lastpath_colname: lastpath_colnames,
                depth_colname: depth_colnames
            }

            return_X_df = pd.DataFrame(datas, index=X[column].index)

            X = X.drop([column], axis=1)
            X = pd.concat([X, return_X_df], axis=1)

        return X


class ListTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unique_category = {}

    def fit(self, X, columns):
        for column in columns:
            if any(X[column].isnull()):
                X[column] = X[column].fillna("0")
            partial_df = X[column].values.tolist()
            transform_flag, unique_category = self._determine_transform(partial_df)
            self.unique_category[column] = {"transform_flag": transform_flag,
                                            "unique_category": unique_category}

        return self

    @staticmethod
    def _is_num(s):
        try:
            float(s)
        except ValueError:
            return False
        else:
            return True

    @staticmethod
    def _find_splitter(partial_df):
        candidates = defaultdict(int)
        for cell in partial_df:
            row = cell.lstrip("[").rstrip("]")
            row = row.replace("'", "")
            splitter = re.sub(r"[^0-9a-zA-Z_]+", "@@", row)
            if len(splitter) > 1:
                for s in splitter:
                    candidates[s] += 1

        candidates = sorted(candidates.items(), key=lambda x: x[1])
        splitter = candidates[-1][0]

        return splitter

    @staticmethod
    def _split(row):
        if "[" in str(row):
            row = row.lstrip("[")
        if "]" in str(row):
            row = row.lstrip("]")
        row = str(row).replace("'", "")
        splitter = re.sub(r"[^0-9a-zA-Z_-]+", "@@", row)
        rows = splitter.split("@@")
        if len(rows) == 0:
            rows = [row]

        return rows

    def _determine_transform(self, X):
        all_elements = []
        all_list_elements = []
        elements_lengths = []
        for cell in X:
            elements = self._split(cell)
            all_elements += elements
            all_list_elements.append(elements)
            elements_lengths.append(len(elements))
        unique_category = list(set(all_elements))
        unique_lengths = list(set(elements_lengths))

        if len(unique_category) < 500: return "categorical", unique_category
        if len(unique_lengths) == 1: return "same_length", all_list_elements
        return "text", None

    def transform(self, X, columns):
        """
        Two ways to transform
        If the number of unique values in list is less than number of rows, make them categorical
        Else
            If the number of elements is the same, split based on the separator
            If not, just treat them as text
        """

        for column in columns:
            if any(X[column].isnull()):
                X[column] = X[column].fillna("0")
            partial_df = X[column].values.tolist()

            transform_flag = self.unique_category[column]["transform_flag"]
            unique_category = self.unique_category[column]["unique_category"]

            if transform_flag == "categorical":
                new_columns = [f"{column}_{attribute}" for attribute in unique_category]
                new_df = []
                for cell in partial_df:
                    record = self._split(cell)
                    new_record = [0] * len(unique_category)
                    for r in record:
                        if r in unique_category:
                            index = unique_category.index(r)
                            new_record[index] = 1
                    new_df.append(new_record)

                new_df = pd.DataFrame(new_df, columns=new_columns, index=X[column].index)
                X = pd.concat([X, new_df], axis=1)
                X = X.drop(columns=column)

            elif transform_flag == "same_length":
                num_columns = len(unique_category[0])
                new_columns = [f"{column}_{i}" for i in range(num_columns)]

                new_df = pd.DataFrame(unique_category, columns=new_columns, index=X[column].index)
                X = pd.concat([X, new_df], axis=1)
                X = X.drop(columns=column)

            else:
                new_df = pd.DataFrame(X[column].values.tolist(), columns=[f"{column}_as_sentence"], index=X[column].index)
                X = pd.concat([X, new_df], axis=1)
                X = X.drop(columns=column)

            pd.set_option('display.max_columns', 1000)

        return X


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, columns):
        # there is nothing to fit
        return self

    def transform(self, X:pd.DataFrame, columns):
        return X.drop(columns, axis=1)


class EmbedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, columns):
        return self

    @staticmethod
    def _to_num(row):
        if is_num(row):
            if np.isnan(float(row)) or np.isinf(float(row)): return row
        num = re.sub(r"[^\d.]", "", row)

        try:
            num = float(num)
        except:
            num = 0

        return num

    def transform(self, X, columns):
        for column in columns:
            X[column] = X[column].apply(lambda row: self._to_num(row))

        return X


class LocationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return self


class UnitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mapping = PreprocessorConfigs().unit_mapping

    def fit(self, X, columns):
        return self

    @staticmethod
    def _check_unit(df):
        values = df.values.tolist()
        units = []
        for v in values:
            v = str(v)
            if is_num(v):
                # Check NaN and Inf
                if np.isnan(float(v)) or np.isinf(float(v)):
                    continue
            sign = re.sub(r"\d+(?:\.\d+)?|\s", "", v)
            units.append(sign)
        units = list(set(units))

        return len(units) == 1

    @staticmethod
    def _to_num(row):
        row = str(row)
        if is_num(row):
            # Check NaN and Inf
            if np.isnan(float(row)) or np.isinf(float(row)): return row
        num = re.sub(r"[^\d.]", "", row)
        try:
            num = float(num)
        except:
            num = 0

        return num

    def _unit_conversion(self, row):
        row = str(row)
        if is_num(row):
            # Check NaN and Inf
            if np.isnan(float(row)) or np.isinf(float(row)): return row
        unit = re.sub(r"[0-9.]+", "", row).lstrip(" ").rstrip(" ")
        num = re.sub(r"[^\d.]", "", row)
        try:
            num = float(num)
        except:
            num = 0
        conversion = self.mapping.get(unit)

        if conversion is not None:
            num *= conversion

        return num

    def transform(self, X, columns):
        for column in columns:
            if self._check_unit(X[column]):
                X[column] = X[column].apply(self._to_num)
            else:
                X[column] = X[column].apply(self._unit_conversion)

        return X


class SignTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    @staticmethod
    def _check_unit(df):
        values = df.values.tolist()
        units = []
        for v in values:
            v = str(v)
            if is_num(v):
                # Check NaN and Inf
                if np.isnan(float(v)) or np.isinf(float(v)):
                    continue
            sign = re.sub(r"\d+(?:\.\d+)?|\s", "", v)
            units.append(sign)
        units = list(set(units))

        return len(units) == 1

    def fit(self, X, columns):
        return self

    @staticmethod
    def _remove_sign(row):
        row = str(row)
        if is_num(row):
            # Check NaN and Inf
            if np.isnan(float(row)) or np.isinf(float(row)): return row
        num = re.sub(r"[^\d.]", "", row)
        try:
            num = float(num)
        except:
            num = 0

        return num

    @staticmethod
    def _to_num(row):
        row = str(row)
        if is_num(row):
            if np.isnan(float(row)) or np.isinf(float(row)): return row
        num = re.sub(r"[^\d.]", "", row)

        try:
            num = float(num)
        except:
            num = 0

        sign = re.sub(r"\d+(?:\.\d+)?", "", row)
        sign = sign.lstrip(" ").rstrip(" ").lower()

        if sign in PreprocessorConfigs().lt_candidates:
            num -= 1
        elif sign in PreprocessorConfigs().gt_candidates:
            num += 1

        return num

    def transform(self, X, columns):
        for column in columns:
            if self._check_unit(X[column]):
                X[column] = X[column].apply(self._remove_sign)
            else:
                X[column] = X[column].apply(lambda row: self._to_num(row))

        return X


class RangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.force_split = {}

    def fit(self, X, columns):
        for column in columns:
            X, all_single = self._split(X, column)
            self.force_split[column] = all_single
        return self

    @staticmethod
    def _to_float(cell):
        try:
            cell = float(cell)
        except:
            cell = 0
        return cell

    def _find_range(self, cell):
        cell = str(cell)
        pattern = re.compile(r'([-]?\d*\.?\d+\D+[-]?\d*\.?\d+)')
        matches = pattern.findall(cell)
        split_enable = False
        if len(matches) > 0:
            match = matches[0]
            splitter = re.sub(r"\d+(?:\.\d+)?", "", match)
            start = self._to_float(match.split(splitter)[0])
            end = self._to_float(match.split(splitter)[1])
            split_enable = True
        else:
            start = self._to_float(re.sub(r"[^\d.]", "", cell))
            end = self._to_float(re.sub(r"[^\d.]", "", cell))

        return start, end, split_enable

    def _split(self, X, column):
        starts, ends = [], []
        partial_df = X[column].values.tolist()
        all_single = True
        for row in partial_df:
            if is_num(row):
                starts.append(self._to_float(row))
                ends.append(self._to_float(row))
                continue
            start, end, split_enable = self._find_range(row)
            starts.append(start)
            ends.append(end)
            if split_enable: all_single = False

        get = self.force_split.get(column)
        if get is not None:
            all_single = get

        if all_single:
            X[f"{column}_no_change"] = starts
        else:
            X[f"{column}_start"] = starts
            X[f"{column}_end"] = ends
        X = X.drop([column], axis=1)

        return X, all_single

    def transform(self, X, columns):
        for column in columns:
            X, _ = self._split(X, column)

        return X


class LLMSplitter(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.llm_output_container = {}
        self.llm_id_columns = {}
        self.llm = LargeLanguageModel()

    @staticmethod
    def _extend(sample_values):
        if len(sample_values) >= 5: return sample_values
        else:
            sample_values = sample_values * 5
            return sample_values

    def _extract_sample_values(self, X, column):
        # List with only numbers
        series = pd.to_numeric(X[column].copy(), errors="coerce")
        series = series.replace([np.inf, -np.inf], np.nan)
        series_num = series.loc[series.notnull()].values.tolist()

        # List with only strings
        series = X[column].copy()
        series = series.loc[series.notnull()].astype("string")
        series[series.str.isnumeric()] = math.nan
        series_str = series.loc[series.notnull()].values.tolist()

        if len(series_num) > 0 and len(series_str) == 0:
            sample_values = [int(i) for i in series_num[:5]]
        elif len(series_num) > 0 and len(series_str) > 0:
            zip_list = zip(series_num, cycle(series_str)) if len(series_num) > len(series_str) else zip(cycle(series_num), series_str)
            sample_values = []

            for i, (v_num, v_str) in enumerate(zip_list):
                sample_values.append(v_str)
                sample_values.append(int(v_num))
                if i == 2: break
        else:
            sample_values = series_str[:5]

        sample_values = self._extend(sample_values)

        print(sample_values)

        return sample_values

    @staticmethod
    def _to_numeric(data_dict):
        for k in data_dict.keys():
            data_dict[k] = pd.to_numeric(data_dict[k], errors="ignore")
        return data_dict

    @staticmethod
    def _detect_splittable(pd_series):
        pd_series = pd_series[pd_series.notnull()].values.tolist()
        split_lengths = []
        for cell in pd_series:
            cell = str(cell)
            splitter = re.findall(r"[^0-9a-zA-Z.]+", cell)
            if len(splitter) > 1:
                split_character = max(splitter, key=splitter.count)
                split_lengths.append(len(cell.split(split_character)))
            else:
                split_lengths.append(1)

        unique_lengths = list(set(split_lengths))

        if len(unique_lengths) > 1 or len(unique_lengths) == 0:
            return False, None
        else:
            if unique_lengths[0] == 1: return False, None
            result = {
                "standard_format": f"The column is split by {split_character}",
                "option": 2,
                "where to split": split_character,
                "reason": f"The column is split by {split_character}",
                "num_splits": unique_lengths[0]
            }
            return True, result

    def _construct_prompt(self, column_name, sample_values):
       prompt = f"""There is a column named '{column_name}' in a tabular data, and the sample values are '{sample_values[0]}', '{sample_values[1]}', '{sample_values[2]}', '{sample_values[3]}', and '{sample_values[4]}'. If any standard format exists for '{column_name}' type, tell me the way to split into the most detailed substrings by following only one of the options below. Otherwise, try the best to split into the most detailed substrings for the sample values (e.g. a combination of numeric and strings MUST BE separated into numeric and strings).
Option1: Split by the specific indices. Also tell me the specific indices to split in the form list of indices. Ignore the indices of extremes. (e.g. If the value is 'ABC12de34fg', the where to split becomes [3, 5, 7, 9]).
Option2: Split based on the specific separator. Also tell me the specific character in the form list of characters.
Option3: Select if option 1 and option 2 are not appropriate
### Output format ###
Option: [Option number]
Reason: [Please explain why you chose the option briefly as if you are a professional expert]
Where to split: [Specific indices if option 1 is selected, specific characters if option 2 is selected, or 'nowhere' if option 3 are selected]
"""

       return prompt

    @staticmethod
    def _post_process(generated_text, column_series):
        remains = generated_text.split("Option: ")[1]
        option, remains = generated_text.split("Reason: ")[0].rstrip("\n"), generated_text.split("Reason: ")[1]
        #option, remains = remains.split("Reason: ")[0].rstrip("\n"), remains.split("Reason: ")[1]
        reason, where = remains.split("Where to split: ")[0].rstrip("\n"), remains.split("Where to split: ")[1].rstrip("\n")
        standard_format = ""

        if "1" in option: option = 1
        elif "2" in option: option = 2
        elif "3" in option: option = 3
        elif "4" in option: option = 4

        where_judge = re.findall(r'\[.*?\]', where)
        if len(where_judge) == 0: option = 4
        elif len(where_judge[0]) > 20: option = 4

        if option == 1:
            where = re.findall(r'\[.*?\]', where)[0]
            where = where.lstrip("[").rstrip("]").split(",")
            try:
                where = [int(i.lstrip(" ").rstrip(" ")) for i in where]
            except:
                option = 3
                where = []
            split_num = 0
        elif option == 2:
            where = re.findall(r'\[.*?\]', where)[0]
            where = where.lstrip("[").rstrip("]").split(",")
            where = where[0].lstrip("'").rstrip("'")
            split_nums = [len(str(value).split(where)) for value in column_series.values.tolist()]
            split_num = max(split_nums)
        elif option == 3 or option == 4:
            split_num = 0

        result = {
            "standard_format": standard_format,
            "option": option,
            "where to split": where,
            "reason": reason,
            "num_splits": split_num
        }

        return result

    @staticmethod
    def _split_by_indices(series, split_indices, column_name):
        values = series.values.tolist()
        num_splits = len(split_indices) + 1
        generated_column_names = []

        # Initialize lists
        split_dict = {}
        for i in range(num_splits):
            split_dict[f"{column_name}_{i}"] = []
            generated_column_names.append(f"{column_name}_{i}")

        for value in values:
            start_index = 0
            if is_num(value):
                # Check NaN and Inf
                if np.isnan(float(value)) or np.isinf(float(value)):
                    for k in split_dict.keys():
                        split_dict[k].append(value)
                    continue

            value = str(value)
            for i, split_index in enumerate(split_indices):
                try:
                    portion = value[start_index: split_index]
                except Exception as e:
                    print(e)
                    portion = ""
                start_index = split_index
                split_dict[f"{column_name}_{i}"].append(portion)
            split_dict[f"{column_name}_{i+1}"].append(value[split_index:])

        return split_dict, generated_column_names

    @staticmethod
    def _split_by_characters(series, split_characters, column_name, split_num):
        values = series.values.tolist()
        split_dict = {}
        generated_column_names = []

        # Initialize lists
        for i in range(split_num):
            split_dict[f"{column_name}_{i}"] = []
            generated_column_names.append(f"{column_name}_{i}")

        for value in values:
            if is_num(value):
                # Check NaN or Inf
                if np.isnan(float(value)) or np.isinf(float(value)):
                    for k in split_dict.keys():
                        split_dict[k].append(value)
                    continue

            splits = str(value).split(split_characters)
            # Fill remaining lists
            if len(splits) != split_num:
                splits += [""] * (split_num - len(splits))

            for i in range(split_num):
                split_dict[f"{column_name}_{i}"].append(splits[i])

        return split_dict, generated_column_names

    @staticmethod
    def _split_into_num_str(series, column_name):
        values = series.values.tolist()
        split_dict = {f"{column_name}_num": [], f"{column_name}_str": []}
        generated_column_names = [f"{column_name}_num", f"{column_name}_str"]
        for value in values:
            if is_num(value):
                # Check NaN and Inf
                if np.isnan(float(value)) or np.isinf(float(value)):
                    for k in split_dict.keys():
                        split_dict[k].append(value)
                    continue

            value = str(value)
            num = re.sub(r"[^\d.]", "", value)
            sign = re.sub(r"\d+(?:\.\d+)?", "", value)

            split_dict[f"{column_name}_num"].append(num)
            split_dict[f"{column_name}_str"].append(sign)

        return split_dict, generated_column_names

    @staticmethod
    def _remove_id_columns(X, column_names):
        id_column_names = []
        for column_name in column_names:
            series = X[column_name]
            unique_values = pd.unique(series[series.notnull()])
            if len(unique_values) == 0 or len(unique_values) == 1:
                id_column_names.append(column_name)

        if len(id_column_names) > 0:
            X = X.drop(id_column_names, axis=1)

        return X, id_column_names

    def fit(self, X, columns):
        prompts = []
        llm_columns = []
        for column in columns:
            splittable, result = self._detect_splittable(X[column])

            if not splittable:
                sample_values = self._extract_sample_values(X, column)
                prompt = self._construct_prompt(column, sample_values)
                prompts.append(prompt)
                llm_columns.append(column)
            else:
                self.llm_output_container[column] = result
                continue

        generated_text = self.llm(prompts)

        for gt, col in zip(generated_text, llm_columns):
            result = self._post_process(gt, X[col])
            print(col, result)
            self.llm_output_container[col] = result

        return self

    def transform(self, X, columns):
        for column in columns:
            result = self.llm_output_container[column]
            where = result["where to split"]
            option = result["option"]
            num_splits = result["num_splits"]

            if option == 1:
                X_portion, generated_column_names = self._split_by_indices(X[column], where, column)
            elif option == 2:
                X_portion, generated_column_names = self._split_by_characters(X[column], where, column, num_splits)
            elif option == 3:
                X[f"{column}_unchanged"] = X.copy()[column]
                X = X.drop([column], axis=1)
                continue
            elif option == 4:
                X_portion, generated_column_names = self._split_into_num_str(X[column], column)

            if option in [1, 2, 4]:
                X_portion = self._to_numeric(X_portion)
                X_portion = pd.DataFrame.from_dict(X_portion)
                X_portion.index = X.index

                X = pd.concat([X, X_portion], axis=1)
                X = X.drop([column], axis=1)

                id_columns = self.llm_id_columns.get(column)
                if id_columns is None:
                    X, id_columns = self._remove_id_columns(X, generated_column_names)
                    if len(id_columns) > 0:
                        self.llm_id_columns[column] = id_columns
                else:
                    X = X.drop(id_columns, axis=1)

        return X

