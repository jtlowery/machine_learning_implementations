import machine_learning_implementations.metrics.general as metrics


class TestMeanAbsoluteError:

    def test_zero_case(self):
        actuals = [1.0, 1.0, 0.0]
        preds = [1.0, 1.0, 0.0]
        assert metrics.mean_absolute_error(actuals, preds) == 0.0

    def test_nonzero_case(self):
        actuals = [1.0, 1.0, 0.0]
        preds = [1.0, 1.0, 1.0]
        assert abs(metrics.mean_absolute_error(actuals, preds) - 0.33333) < 0.00001

class TestMeanSquaredError:

    def test_zero_case(self):
        actuals = [1.0, 1.0, 0.0]
        preds = [1.0, 1.0, 0.0]
        assert metrics.mean_squared_error(actuals, preds) == 0.0

    def test_nonzero_case(self):
        actuals = [1.0, 1.0, 0.0]
        preds = [1.0, 1.0, 1.0]
        assert abs(metrics.mean_squared_error(actuals, preds) - 0.33333) < 0.00001

    def test_single_element_case(self):
        actuals = [2.0]
        preds = [5.0]
        assert metrics.mean_squared_error(actuals, preds) == 9.0
