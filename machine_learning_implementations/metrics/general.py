def mean_absolute_error(predictions, actuals):
    sum = 0
    count = 0
    for prediction, actual in zip(predictions, actuals):
        sum += abs(prediction - actual)
        count += 1
    return sum / count
