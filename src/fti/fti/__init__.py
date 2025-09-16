from src.fti.fti.feature_type_inference import FTI


class Config:
    multiple = True
    first_method = "tfidf"
    second_method = "bert"
    num_train = 200
    num_val = 80
    loops = 3
    session = "dwv2"
    pca = True


def fti(df, target_column_names=None, task=None):
    config = Config()

    if target_column_names is None or task is None:
        config.multiple = False

    _fti = FTI(config)

    if target_column_names is not None and task is not None:
        feature_types = _fti(df, target_column_names, task)
        return feature_types

    print("INFO: Since target columns and the task are not specified, FTI will be executed without considering multiple candidates.")
    feature_types = _fti(df)
    return feature_types
