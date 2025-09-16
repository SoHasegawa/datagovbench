import json

from pandas.api.types import is_numeric_dtype


def is_category_column(c):
    c2 = c.loc[c.notnull()]
    num = c2.nunique()
    if num <= 2:
        return "binary"
    elif num <= 20 and (num / c2.shape[0] < 0.3):
        return "small"
    elif c2.value_counts(normalize=True).head(20).sum() > 0.8:
        return "large"
    else:
        return False


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def is_numeric_type(c):
    if is_numeric_dtype(c):
        return True
    else:
        return False


def json_read(json_path):
    with open(json_path) as f:
        fs = json.load(f)

    return fs
