class PipelineConfig:
    map_to_name = {
        0: "Numeric",
        1: "Categorical",
        2: "Datetime",
        3: "Sentence",
        4: "URL",
        5: "Embed",
        6: "List",
        7: "Drop",
            #"location": 8,
        8: "Unit",
        9: "Sign",
        10: "Range",
        11: "Formatted-id"
    }


class TwoStepConfigs:
    map_to_name = {
        0: "Numeric",
        1: "Categorical",
        2: "Datetime",
        3: "Sentence",
        4: "URL",
        5: "Embed",
        6: "List",
        7: "Drop",
        8: "Unit",
        9: "Sign",
        10: "Range",
        11: "Formatted-id"
    }

    map_to_unite = {0: 2, 1: 3, 2: 4, 3: 6, 4: 5, 5: 8, 6: 10, 7: 9, 8: 11}


class PreprocessorConfigs:
    unit_mapping = {
            "g": 1,
            "grams": 1,
            "gram": 1,
            "kg": 1000,
            "kilograms": 1000,
            "kilogram": 1000,
            "t": 1000000,
            "tonne": 1000000,
            "mg": 0.001,
            "cm": 0.01,
            "centimeters": 0.01,
            "centimeter": 0.01,
            "m": 1,
            "meter": 1,
            "meters": 1,
            "km": 1000,
            "kilometers": 1000,
            "kilometer": 1000,
            "year": 365,
            "yr": 365,
            "years": 365,
            "month": 30,
            "months": 30,
            "days": 1,
            "day": 1,
            "mm": 0.001,
            "K": 1000,
            "M": 1000000,
            "グラム": 1,
            "キログラム": 1000,
            "ミリグラム": 0.001,
            "トン": 1000000,
            "メートル": 1,
            "キロメートル": 1000,
            "ミリメートル": 0.001,
            "年": 365,
            "月": 30,
            "日": 1,
            "週": 7,
            "分": 60,
            "秒": 1,
            "キロバイト": 1000,
            "メガバイト": 1000000,
            "ギガバイト": 1000000000
        }
    lt_candidates = ["less than",
                     "Less than",
                     "Smaller than",
                     "smaller than"
                     "<",
                     "<=",
                     "lt",
                     "LT"]
    gt_candidates = ["greater than",
                     "larger than",
                     "Larger than",
                     "Greater than",
                     ">",
                     ">=",
                     "over",
                     "Over",
                     "gt",
                     "GT",
                     "+"]
    split_candidates = ["-",
                        ":",
                        "to",
                        "～",
                        "から"]
