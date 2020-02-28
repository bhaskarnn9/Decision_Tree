from . import PreProcessing
from . import Model
from . import Predict
from . import Metrics

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = PreProcessing.pre_processing()
    model_dt = Model.model(x_train, y_train)
    y_train_pred, y_test_pred = Predict.predict(x_train, x_test)
    Metrics.console_metric(y_train, y_train_pred, y_test, y_test_pred)