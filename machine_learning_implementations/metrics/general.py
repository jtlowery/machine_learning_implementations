import typing
from collections import Counter
from math import log2
from machine_learning_implementations.lin_alg.lin_alg import squared_distance

AnyNum = typing.TypeVar('AnyNum', int, float)


def mean_absolute_error(predictions: typing.Iterable[AnyNum],
                        actuals: typing.Iterable[AnyNum]) -> float:
    error_sum = 0
    for prediction, actual in zip(predictions, actuals):
        error_sum += abs(prediction - actual)
    return error_sum / len(predictions)


def mean_squared_error(predictions: typing.Iterable[AnyNum],
                       actuals: typing.Iterable[AnyNum]) -> float:
    error_sum = squared_distance(predictions, actuals)
    return error_sum / len(predictions)


def root_mean_squared_error(predictions: typing.Iterable[AnyNum],
                            actuals: typing.Iterable[AnyNum]) -> float:
    return mean_squared_error(predictions, actuals) ** 0.5


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


def gini_impurity(labels: typing.Iterable) -> float:
    label_counts = Counter(labels)
    num_labels = len(labels)
    return 1 - sum(((float(label_count) / num_labels) ** 2
                    for label_count in label_counts.values()))


def misclassification_error(labels: typing.Iterable) -> float:
    label_counts = Counter(labels)
    num_labels = len(labels)
    return 1 - max(label_count / num_labels for label_count in label_counts.values())


def precision(true_positive: int, false_positive: int):
    return true_positive / (true_positive + false_positive)


def recall(true_positive: int, false_negative: int):
    return true_positive / (true_positive + false_negative)
