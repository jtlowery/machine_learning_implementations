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


def average_precision(actuals, predictions, n: int) -> float:

    if not actuals:
        return 0.0

    actuals = set(actuals) if not isinstance(actuals, set) else actuals
    score = 0.0
    num_correct_preds = 0

    already_scored_preds = set()
    for idx, pred in enumerate(predictions):
        # guard against there being more predictions than allowed by n
        if idx + 1 > n:
            break
        if pred in actuals and pred not in already_scored_preds:
            already_scored_preds.add(pred)
            num_correct_preds += 1
            score += num_correct_preds / (idx + 1)
    return score / min(len(actuals), n)

