from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

from . import ExecutionEngine


def console_metric(y_train, y_train_pred, y_test, y_test_pred):
    print('confusion matrix for train prediction:')
    print('\n')
    print(confusion_matrix(y_true=y_train, y_pred=y_train_pred))
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('\n')
    print('confusion matrix for test prediction:')
    print('\n')
    print(confusion_matrix(y_true=y_test, y_pred=y_test_pred))
    print('\n')
    print('depth of model: ', ExecutionEngine.model_dt.get_depth())
    print('\n')
    print('recall score of train prediction')
    print('\n')
    print(recall_score(y_true=y_train, y_pred=y_train_pred, pos_label='Yes'))
    print('\n')
    print('recall score of test prediction')
    print('\n')
    print(recall_score(y_true=y_test, y_pred=y_test_pred, pos_label='Yes'))

