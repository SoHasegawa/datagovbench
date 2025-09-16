import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.fti.fti.model import FTIModel
from src.fti.fti.preprocessor import (
    DateTimeTransformer,
    DropColumnsTransformer,
    EmbedTransformer,
    ListTransformer,
    LLMSplitter,
    NestedTransformers,
    RangeTransformer,
    SignTransformer,
    TextTfIdfTransformer,
    UnitTransformer,
    UrlTransformer,
)

warnings.filterwarnings("ignore")


def split_dataset(df, target_column_name, seed, test_size=0.25):
    X = df.drop(target_column_name, axis=1)
    y = df[target_column_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test


def sampling_rows(df: pd.DataFrame,
                  target_column: list[str],
                  task: str,
                  num_samples: int) -> pd.DataFrame:
    df = df.copy()
    if task == "classification":
        num_categories = df[target_column[0]].nunique()
        if num_categories > 25:
            num_samples = int(num_samples * 25 / num_categories)
        return df.groupby(target_column,
                          group_keys=False).apply(lambda x: x.sample(min(len(x),
                                                                         num_samples)))
    elif task == "regression":
        try:
            target_column = [target_column[0]]
            _, bins, = np.histogram(df[target_column].values, bins='doane')
        except Exception as e:
            print(f"Sampling error: {e}")
            return df
        tentative_classes = []
        num_classes = len(bins) - 1
        if num_classes > 25:
            num_samples = int(num_samples * 25 / num_classes)
        for v in df[target_column].values:
            for i in range(num_classes):
                if v >= bins[i] and v <= bins[i + 1]: tentative_classes.append(i)
        if len(df[target_column].values) != len(tentative_classes): return df
        df[f"{target_column[0]}_hist"] = tentative_classes
        target_column = [f"{target_column[0]}_hist"]
        df = df.groupby(target_column,
                        group_keys=False).apply(lambda x: x.sample(min(len(x),
                                                                       num_samples)))
        df = df.drop(columns=target_column)

        return df


class RuleBasedAutoML:
    def __init__(self, num_loops):
        self.mapping_fts_to_preprocessor = {
            0: StandardScaler(with_mean=True),
            1: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            2: DateTimeTransformer(),
            3: TextTfIdfTransformer(),
            4: UrlTransformer(),
            5: EmbedTransformer(),
            6: ListTransformer(),
            7: DropColumnsTransformer(),
            8: UnitTransformer(),
            9: SignTransformer(),
            10: RangeTransformer(),
            11: LLMSplitter()
        }
        self.num_loops = num_loops

    @staticmethod
    def _is_proceed_to_pipeline(feature_types: np.array) -> bool:
        allowed = True
        for feature_type in feature_types:
            if feature_type not in [0, 1, 3]: allowed = False
        return allowed

    @staticmethod
    def _aggregate_data_enrichment(feature_types, columns):
        aggregated_fts = defaultdict(list)
        updated_feature_types = []
        drop_columns = []

        for ft, column in zip(feature_types, columns):
            if ft in [2, 4, 6, 7, 10, 11]:
                drop_columns.append(ft)
            if ft not in [0, 1, 3]:
                aggregated_fts[ft].append(column)

        for ft, column in zip(feature_types, columns):
            if ft in [2, 4, 6, 7, 10, 11]: continue
            if ft in [5, 8, 9]:
                updated_feature_types.append(0)
                continue
            updated_feature_types.append(ft)

        return aggregated_fts, drop_columns, updated_feature_types

    @staticmethod
    def _aggregate_ml_pipeline(feature_types, columns):
        aggregated_fts = defaultdict(list)

        for ft, column in zip(feature_types, columns):
            if ft in [0, 1, 3]:
                aggregated_fts[ft].append(column)

        return aggregated_fts

    @staticmethod
    def _remove_columns(columns, drop_columns):
        columns = columns.values.tolist()
        for dc in drop_columns:
            columns.remove(dc)

        return columns

    @staticmethod
    def _extract_diff_columns(previous_columns: list[str], updated_columns: list[str]) -> list[str]:
        diff_columns = []
        for updated_column in updated_columns:
            if updated_column not in previous_columns:
                diff_columns.append(updated_column)

        return diff_columns

    @staticmethod
    def _diff_column_submatch(diff_column, split_columns):
        matches = [1 for c in split_columns if c in diff_column]
        match = len(matches) > 0

        return match

    def enricher(self,
                 feature_types,
                 columns, df, fti,
                 llm_transformer=None
                 ):
        ml_pipeline_enable = self._is_proceed_to_pipeline(feature_types)
        iteration = 0

        transformers = []

        # Force feature types of following columns to be sentence
        passthrough_columns = [f"{column}_as_sentence" for column, ft in zip(df.columns, feature_types) if ft == 6]
        passthrough_columns += [f"{column}_Domain" for column, ft in zip(df.columns, feature_types) if ft == 4]
        passthrough_columns += [f"{column}_Path" for column, ft in zip(df.columns, feature_types) if ft == 4]
        passthrough_columns += [f"{column}_LastPath" for column, ft in zip(df.columns, feature_types) if ft == 4]
        passthrough_columns += [f"{column}_as_sentence" for column, ft in zip(df.columns, feature_types) if ft == 2]
        passthrough_columns += [f"{column}_unchanged" for column, ft in zip(df.columns, feature_types) if ft == 11]

        split_columns = [column for column, ft in zip(df.columns, feature_types) if ft == 11]
        original_columns = df.columns

        while not ml_pipeline_enable:
            if iteration == self.num_loops: break
            fts_data_enrichment, _, updated_feature_types = self._aggregate_data_enrichment(feature_types, columns)
            previous_columns = df.columns
            for i, (feature_type, columns) in enumerate(fts_data_enrichment.items()):
                if feature_type != 11:
                    mapping = self.mapping_fts_to_preprocessor[feature_type].fit(df, columns)
                else:
                    mapping = llm_transformer
                df = mapping.transform(df, columns)
                tf_tuple = (mapping, columns)
                transformers.append(tf_tuple)

            updated_columns = df.columns
            diff_columns = self._extract_diff_columns(previous_columns, updated_columns)
            if len(diff_columns) > 0:
                diff_feature_types = fti.predict(df[diff_columns])
                for i, diff_column in enumerate(diff_columns):
                    if diff_column in passthrough_columns and diff_feature_types[i] != 7:
                        diff_feature_types[i] = 3
                    elif diff_feature_types[i] == 11:
                        diff_feature_types[i] = 3
                feature_types = updated_feature_types + diff_feature_types.tolist()
            else:
                feature_types = updated_feature_types

            columns = df.columns
            iteration += 1

            ml_pipeline_enable = self._is_proceed_to_pipeline(feature_types)

        transformers = NestedTransformers(transformers)

        return df, feature_types, transformers

    def pipeliner(self, feature_types, columns, df, task, model_dict,
                  multiple_target_columns=False):
        fts_ml_pipeline = self._aggregate_ml_pipeline(feature_types, columns)
        ml_pipelines = []
        sequence_count = 0
        for i, (feature_type, columns) in enumerate(fts_ml_pipeline.items()):
            if feature_type == 3:
                for column in columns:
                    ml_pipeline_tuple = (
                        f"ml_pipeline_{sequence_count}",
                        self.mapping_fts_to_preprocessor[feature_type],
                        column
                    )
                    sequence_count += 1
                    ml_pipelines.append(ml_pipeline_tuple)
            else:
                ml_pipeline_tuple = (
                    f"ml_pipeline_{sequence_count}",
                    self.mapping_fts_to_preprocessor[feature_type],
                    columns
                )
                sequence_count += 1
                ml_pipelines.append(ml_pipeline_tuple)

        ml_pipelines = ColumnTransformer(ml_pipelines, remainder='passthrough')
        #x = ml_pipelines.fit_transform(df)

        if task == "classification":
            lgbm_model = LGBMClassifier(num_iterations=model_dict["num_iterations"],
                                        n_estimators=model_dict["n_estimators"],
                                        num_leaves=model_dict["num_leaves"],
                                        max_depth=model_dict["max_depth"],
                                        learning_rate=model_dict["learning_rate"])
            rf_model = RandomForestClassifier()
            if multiple_target_columns:
                lgbm_model = MultiOutputClassifier(lgbm_model)
                rf_model = MultiOutputClassifier(rf_model)
            lgbm_pipelines = Pipeline([
                ('preprocess', ml_pipelines),
                ('model', lgbm_model)
            ])
            rf_pipelines = Pipeline([
                ('preprocess', ml_pipelines),
                ('model', rf_model)
            ])
        elif task == "regression":
            lgbm_model = LGBMRegressor(num_iterations=model_dict["num_iterations"],
                                       n_estimators=model_dict["n_estimators"],
                                       num_leaves=model_dict["num_leaves"],
                                       max_depth=model_dict["max_depth"],
                                       learning_rate=model_dict["learning_rate"])
            rf_model = RandomForestRegressor()
            if multiple_target_columns:
                lgbm_model = MultiOutputRegressor(lgbm_model)
                rf_model = MultiOutputRegressor(rf_model)
            lgbm_pipelines = Pipeline([
                ('preprocess', ml_pipelines),
                ('model', lgbm_model)
            ])
            rf_pipelines = Pipeline([
                ('preprocess', ml_pipelines),
                ('model', rf_model)
            ])

        return lgbm_pipelines, rf_pipelines


class FTI:
    def __init__(self, args):
        self.fti = FTIModel(dataset_train_path=None,
                            dataset_test_path=None,
                            session_path=None,
                            first_method=args.first_method,
                            second_method=args.second_method,
                            pretrain_session=args.session,
                            txt_feature_only=False,
                            original_numerical=True,
                            pca=True,
                            top_n=50
                            )
        self.automl = RuleBasedAutoML(args.loops)
        self.multiple = args.multiple
        self.train_samples = args.num_train
        self.val_samples = args.num_val
        self.llm_transformer = None

        self.model_dict = {
            "num_iterations": 30,
            "n_estimators": 100,
            "max_depth": 3,
            "num_leaves": 7,
            "learning_rate": 0.25
        }

    @staticmethod
    def _imputer(df: pd.DataFrame) -> pd.DataFrame:
        imputer_enable = df.isna().any()
        if imputer_enable.any():
            for column in df.columns:
                df[column] = df[column].replace([np.inf, -np.inf], np.nan)
                if len(df[column].mode()) == 0:
                    df[column] = df[column].fillna("nothing")
                else:
                    df[column] = df[column].fillna(df[column].mode()[0])

        return df

    @staticmethod
    def _copy(X_train, X_test, y_train, y_test):
        X_train = X_train.copy()
        X_test = X_test.copy()
        y_train = y_train.copy()
        y_test = y_test.copy()

        return X_train, X_test, y_train, y_test

    @staticmethod
    def _include_2nd_feature_types(feature_type):
        if 4 in feature_type: return True
        if 5 in feature_type: return True
        if 6 in feature_type: return True
        if 8 in feature_type: return True
        if 9 in feature_type: return True
        if 10 in feature_type: return True
        if 11 in feature_type: return True
        return False

    def _sample(self,
                X_train, X_val, y_train, y_val,
                target_column_name,
                task):
        train_sample = sampling_rows(pd.concat([X_train, y_train], axis=1),
                                     target_column_name,
                                     task,
                                     num_samples=self.train_samples)
        val_sample = sampling_rows(pd.concat([X_val, y_val], axis=1),
                                   target_column_name,
                                   task,
                                   num_samples=self.val_samples)
        y_train_sample = train_sample[target_column_name]
        X_train_sample = train_sample.drop(columns=target_column_name)
        y_val_sample = val_sample[target_column_name]
        X_val_sample = val_sample.drop(columns=target_column_name)

        return X_train_sample, X_val_sample, y_train_sample, y_val_sample

    def _split_dataset(self,
                       df,
                       target_column_name):
        X_trainval, X_test_org, y_trainval, y_test_org = split_dataset(df, target_column_name, seed=0, test_size=0.25)
        X_train_org, X_val, y_train_org, y_val = split_dataset(pd.concat([X_trainval, y_trainval], axis=1), target_column_name, seed=0, test_size=0.20)
        x_columns = X_train_org.columns

        X_train, X_test, y_train, y_test = self._copy(X_train_org,
                                                      X_test_org,
                                                      y_train_org,
                                                      y_test_org)

        return X_train, X_test, y_train, y_test

    def _validation(self,
                    fti_candidates,
                    X_train_sample, X_val_sample, y_train_sample, y_val_sample,
                    task):
        score_lists = []
        score_indices = []

        # Pre-run LLM transformer fit for saving time
        columns = X_train_sample.columns
        split_columns = []
        for fti_candidate in fti_candidates:
            for i, ft in enumerate(fti_candidate):
                if ft == 11: split_columns.append(columns[i])
        split_columns = list(set(split_columns))

        if len(split_columns) > 0:
            print(split_columns)
            transformer = LLMSplitter().fit(X_train_sample, split_columns)
            self.llm_transformer = transformer
        else:
            transformer = None

        multiple_target_columns = len(y_train_sample.columns) > 1
        print(multiple_target_columns)

        for i, feature_types in enumerate(fti_candidates):
            print(feature_types)
            try:
                X_train, X_val, y_train, y_val = self._copy(X_train_sample,
                                                            X_val_sample,
                                                            y_train_sample,
                                                            y_val_sample)
                x_columns = X_train.columns
                updated_X_train, updated_feature_types, transformers = self.automl.enricher(feature_types,
                                                                                            x_columns,
                                                                                            X_train,
                                                                                            self.fti,
                                                                                            llm_transformer=transformer)
                updated_X_val = transformers.transform(X_val)
                lgbm_pipelines, rf_pipelines = self.automl.pipeliner(updated_feature_types,
                                                                        updated_X_train.columns,
                                                                        updated_X_train,
                                                                        task,
                                                                        model_dict=self.model_dict,
                                                                        multiple_target_columns=multiple_target_columns
                                                                        )
                updated_X_train = self._imputer(updated_X_train)
                updated_X_val = self._imputer(updated_X_val)
                y_train = self._imputer(y_train)
                y_val = self._imputer(y_val)

                lgbm_pipelines.fit(updated_X_train, y_train)
                lgbm_score = lgbm_pipelines.score(updated_X_val, y_val)

                score_lists.append(lgbm_score)
                score_indices.append(i)
            except Exception as e:
                print(f"Validation error: {e}")

        return score_lists, score_indices

    def _generate_single_candidate(self, df):
        return self.fti.predict(df)

    def _generate_multiple_candidates(self, df, target_column_name, task):
        X_train, X_val, y_train, y_val = self._split_dataset(df, target_column_name)

        fti_candidates = self.fti.predict_multiple(X_train)
        print(fti_candidates)
        y_train = self._imputer(y_train)
        y_val = self._imputer(y_val)
        X_train_sample, X_val_sample, y_train_sample, y_val_sample = self._sample(X_train, X_val,
                                                                                  y_train, y_val,
                                                                                  target_column_name,
                                                                                  task)
        score_lists, score_indices = self._validation(fti_candidates,
                                                      X_train_sample, X_val_sample,
                                                      y_train_sample, y_val_sample,
                                                      task)
        print(score_lists)
        if len(score_lists) == 0:
            best_feature_type = self.fti.predict(X_train)
        else:
            best_indices = np.flatnonzero(score_lists == np.max(score_lists))
            best_indices = list(np.array(score_indices)[best_indices])
            break_flag = False
            for best_index in best_indices:
                best_feature_type = fti_candidates[best_index]
                if self._include_2nd_feature_types(best_feature_type):
                    break_flag = True
                    break

            if not break_flag:
                best_feature_type = fti_candidates[best_indices[0]]

        return best_feature_type

    def __call__(self,
                 df: pd.DataFrame,
                 target_column_name=None,
                 task=None) -> np.array:
        if self.multiple:
            return self._generate_multiple_candidates(df, target_column_name, task)
        else:
            return self._generate_single_candidate(df)
