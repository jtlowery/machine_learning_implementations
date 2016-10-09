import typing

AnyNum = typing.TypeVar('AnyNum', int, float)


def mean_absolute_error(predictions: typing.Iterable[AnyNum],
                        actuals: typing.Iterable[AnyNum]) -> float:
    sum = 0
    count = 0
    for prediction, actual in zip(predictions, actuals):
        sum += abs(prediction - actual)
        count += 1
    return sum / count
