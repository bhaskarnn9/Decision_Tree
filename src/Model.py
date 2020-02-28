from sklearn.tree import DecisionTreeClassifier


def model(x_train, y_train):

    model_dt = DecisionTreeClassifier()

    model_dt.fit(x_train, y_train)

    return model_dt
