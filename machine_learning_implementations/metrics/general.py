import typing

AnyNum = typing.TypeVar('AnyNum', int, float)


def mean_absolute_error(predictions: typing.Iterable[AnyNum],
                        actuals: typing.Iterable[AnyNum]) -> float:
    error_sum = 0
    for prediction, actual in zip(predictions, actuals):
        error_sum += abs(prediction - actual)
    return error_sum / len(predictions)


def mean_squared_error(predictions: typing.Iterable[AnyNum],
                        actuals: typing.Iterable[AnyNum]) -> float:
    error_sum = 0
    for prediction, actual in zip(predictions, actuals):
        error_sum += (prediction - actual)**2
    return error_sum / len(predictions)
