import typing
from collections import Counter
from math import log2

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


def average_precision(actuals: typing.Iterable,
                      predictions: typing.Iterable,
                      n: int) -> float:

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


def mean_average_precision(actuals: typing.Iterable[typing.Iterable],
                           predictions: typing.Iterable[typing.Iterable],
                           n: int) -> float:
    return (sum(average_precision(acts, preds, n) for acts, preds
                in zip(actuals, predictions)) / len(actuals))


def entropy(labels: typing.Iterable) -> float:
    label_counts = Counter(labels)
    num_labels = len(labels)
    probs = [float(l) / num_labels for l in label_counts.values()]
    entropy_val = 0.0
    for prob in probs:
        if prob > 0.0:
            entropy_val += -prob * log2(prob)
    return entropy_val
