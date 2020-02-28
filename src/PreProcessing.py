import pandas as pd
import os
from sklearn.model_selection import train_test_split

TRAINING_DATA = os.environ.get('TRAINING_DATA')


def pre_processing():
    df = pd.read_csv(TRAINING_DATA)

    df.dropna(axis=0, inplace=True)

    df.drop('sku', axis=1, inplace=True)

    df = type_cast_to_cat(df)

    df = dummy(df)

    x, y = df.drop('went_on_backorder', axis=1), df.went_on_backorder

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=123)

    return x_train, x_test, y_train, y_test


def type_cast_to_cat(data_frame):
    obj_attrs = get_obj_attrs(data_frame)

    for item in obj_attrs:
        data_frame[item] = data_frame[item].astype('category')
        return data_frame


def dummy(data_frame):
    obj_attrs = get_obj_attrs(data_frame)

    obj_attrs.remove('went_on_backorder')

    cat_attrs = obj_attrs

    data_frame = pd.get_dummies(data_frame, columns=cat_attrs, prefix=cat_attrs, prefix_sep='_', drop_first=True)

    return data_frame


def get_obj_attrs(data_frame):
    return list(data_frame.select_dtypes('object').columns)
