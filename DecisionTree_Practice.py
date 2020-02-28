import os
import pandas as pd
from sklearn.model_selection import train_test_split

TRAINING_DATA = os.environ.get('TRAINING_DATA')

if __name__ == '__main__':
    
    preprocessing(TRAINING_DATA)

def preprocessing():

    df = pd.read_csv(TRAINING_DATA)

    df.dropna(axis=0, inplace=True)

    df.drop('sku', axis=0, inplace=True)

    df = type_cast_to_cat(df)

    df = dummify(df)

    X, y = df.drop('went_on_backorder', axis=1), df.went_on_backorder

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    return X_train, X_test, y_train, y_test

def type_cast_to_cat(dataframe):
    
    obj_attrs = get_obj_attrs(dataframe)

    for item in obj_attrs:
        dataframe[item] = dataframe[item].astype('category')
        return dataframe

def dummify(dataframe):

    obj_attrs = get_obj_attrs(dataframe)

    obj_attrs.remove('went_on_backorder')

    cat_attrs = obj_attrs

    dataframe = pd.get_dummies(dataframe, columns=cat_attrs, prefix=cat_attrs,\
        prefix_sep='_', drop_first=True)
    
    return dataframe

def get_obj_attrs(dataframe):

    return list(dataframe.select_dtypes('object').columns)