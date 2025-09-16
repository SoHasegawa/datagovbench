from src.fti.fti import fti

from pandas.api.types import infer_dtype


def to_sequence_descriptive(df):
    feature_type_mapping = {
            0: "Numerical",
            1: "Categorical",
            2: "Datetime",
            3: "Sentence",
            4: "URL",
            5: "Embedded numbers",
            6: "List",
            7: "ID",
            8: "Numbers with Unit",
            9: "Numbers with Sign",
            10: "Range of Numbers",
            11: "Formatted ID"
        }
    feature_types = fti(df)
    columns = df.columns

    num_columns = len(columns)
    num_rows = len(df.index)

    sequence = f"Number of columns: {num_columns}\nNumber of rows: {num_rows}\n"
    sequence += f"Feature type, pandas type, ratio of missing values, and feature type-specific information is given for each column as below.\n"
    for column, feature_type in zip(columns, feature_types):
        if df[column].dropna().nunique() == 1: feature_type = 1
        feature_type_name = feature_type_mapping[feature_type]
        series = df[column]
        nan_percentage = series.isna().sum() * 100.0 / (len(series.index) + 1e-9)
        if nan_percentage > 0 and nan_percentage < 1:
            nan_percentage = 1
        else:
            nan_percentage = int(nan_percentage)
        pandas_dtype = infer_dtype(df[column].dropna())
        if feature_type == 0 or feature_type == 2:
            try:
                max_value, min_value = df[column].max(), df[column].min()
            except Exception as e:
                print(e)
                values = df[column].values.tolist()
                max_value, min_value = values[-1], values[0]
            if feature_type == 0:
                sequence += f"{column} (feature type: {feature_type_name}) (pandas type: {pandas_dtype}) (ratio of missing values: {nan_percentage}%): Value range is [{min_value}, {max_value}].\n"
            elif feature_type == 2:
                sequence += f"{column} (feature type: {feature_type_name}) (pandas type: {pandas_dtype}) (ratio of missing values: {nan_percentage}%): Start date is {min_value}, and end date is {max_value}.\n"
        elif feature_type == 1:
            categories = list(set(df[column].dropna().values.tolist()))
            if len(categories) > 20:
                categories = [str(c) for c in categories][:20]
                categories = ", ".join(categories)
                sequence += f"{column} (feature type: {feature_type_name}) (pandas type: {pandas_dtype}) (ratio of missing values: {nan_percentage}%): Selected 20 categories are [{categories}].\n"
            else:
                categories = [str(c) for c in categories]
                categories = ", ".join(categories)
                sequence += f"{column} (feature type: {feature_type_name}) (pandas type: {pandas_dtype}) (ratio of missing values: {nan_percentage}%): All categories are [{categories}].\n"
        elif feature_type in [4]:
            sequence += f"{column} (feature type: {feature_type_name}) (pandas type: {pandas_dtype}) (ratio of missing values: {nan_percentage}%)\n"
        else:
            sample_values = list(set(df[column].dropna().values.tolist()))[:10]
            sample_values = [str(v) for v in sample_values]
            sample_values = ", ".join(sample_values)
            sequence += f"{column} (feature type: {feature_type_name}) (pandas type: {pandas_dtype}) (ratio of missing values: {nan_percentage}%): Selected 10 unique examples are [{sample_values}].\n"
    
    return sequence


def sequence_tables(dfs, metadata, head_only=False):
    table_information = []
    distributions = metadata["distribution"]
    for i, distribution in enumerate(distributions):
        df = dfs[i]
        file_title = distribution["file_title"]
        file_description = distribution["file_description"]

        if head_only:
            table_sequence = df.head(10).to_string(index=False)
        else:
            table_sequence = to_sequence_descriptive(df)

        if i == 0: prefix = "1st table"
        elif i == 1: prefix = "2nd table"
        elif i == 2: prefix = "3rd table"
        else: prefix = f"{i+1}th table"

        prompt_for_file = f"""{prefix}
Dataset title: {file_title}
Dataset description: {file_description}

Headers and values:
{table_sequence}
"""
            
        table_information.append(prompt_for_file)

    return table_information


def sequence_external_knowledge(external_knowledges):
    knowledge = ""
    for external_knowledge in external_knowledges:
        if external_knowledge is None: continue
        if isinstance(external_knowledge, dict):
            for column_name, description in external_knowledge.items():
                if isinstance(description, str):
                    v = description
                elif isinstance(description, dict):
                    v = ""
                    for item, desc in description.items():
                        if isinstance(desc, float):
                            desc = "none"
                        else:
                            if item != "values":
                                desc = desc.split("/")[0]
                            v += f"{item} is {desc}. "
                    v = v.rstrip(" ")
                knowledge += f"{column_name}: {v}\n"
        elif isinstance(external_knowledge, str):
            knowledge += external_knowledge
        knowledge += "\n"

    return knowledge


def sequence_qa_pairs(qa_pairs):
    pairs = ""
    image_paths = []

    image_num = 0

    for i, qa_pair in enumerate(qa_pairs):
        if i % 2 == 0: pairs += f"Q: {qa_pair}\n"
        if i % 2 == 1:
            if isinstance(qa_pair, list):
                images = ""
                for i, qa in enumerate(qa_pair):
                    if image_num == 0: images += "1st image, "
                    elif image_num == 1: images += "2nd image, "
                    elif image_num == 2: images += "3rd image, "
                    else: images += f"{image_num+1}th image, "
                    image_paths.append(qa)
                    image_num += 1
                images = images[:-2]
                pairs += f"A: {images}\n"
            else:
                pairs += f"A: {qa_pair}\n\n"

    return pairs, image_paths


def sequence_questions(qa_pairs):
    questions = ""

    for i, qa_pair in enumerate(qa_pairs):
        if i % 2 == 0: questions += f"{qa_pair}\n"

    return questions