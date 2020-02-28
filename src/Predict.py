from . import ExecutionEngine


def predict(x_train, x_test):
    y_train_pred = ExecutionEngine.model_dt.predict(x_train)
    y_test_pred = ExecutionEngine.model_dt.predict(x_test)

    return y_train_pred, y_test_pred
