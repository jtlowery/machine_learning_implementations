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


class TestRootMeanSquaredError:

    def test_zero_case(self):
        actuals = [1.0]
        preds = [1.0]
        assert metrics.root_mean_squared_error(actuals, preds) == 0.0

    def test_nonzero_case(self):
        actuals = [1.0, 1.0, 0.0]
        preds = [1.0, 1.0, 1.0]
        assert metrics.root_mean_squared_error(actuals, preds) - 0.5773502 < 0.00001

        actuals = [1.0, 1.0, 1.0, 2.0]
        preds = [1.0, 1.0, 1.0, 4.0]
        assert metrics.root_mean_squared_error(actuals, preds) == 1.0


class TestAveragePrecision:

    def test_zero_case(self):
        actuals = [1, 2, 3]
        preds = [4, 5, 6]
        assert metrics.average_precision(actuals, preds, 3) == 0.0

        actuals = ['a']
        preds = ['b']
        assert metrics.average_precision(actuals, preds, 1) == 0.0

    def test_partially_correct_case(self):
        actuals = ['a', 'b']
        preds = ['a', 'c']
        assert metrics.average_precision(actuals, preds, 2) == 0.5

        actuals = ['a', 'b']
        preds = ['c', 'a']
        assert metrics.average_precision(actuals, preds, 2) == 0.25

        actuals = [1, 2, 3]
        preds = [1, 5, 6]
        assert abs(metrics.average_precision(actuals, preds, 3) -
                   0.33333) < 0.00001

    def test_all_correct_case(self):
        actuals = [1, 2, 3]
        preds = [1, 2, 3]
        assert metrics.average_precision(actuals, preds, 3) == 1.0

        actuals = [1.0, 2.0]
        preds = [2.0, 1.0]
        assert metrics.average_precision(actuals, preds, 2) == 1.0

    def test_n_cutoff(self):
        # test that n prevents excess predictions from being counted
        actuals = [1, 2, 3]
        preds = [1, 4, 5]
        assert metrics.average_precision(actuals, preds, 2) == 0.5


class TestMeanAveragePrecision:

    def test_single_element_case(self):
        actuals = [[1, 2]]
        preds = [[1, 4]]
        assert metrics.mean_average_precision(actuals, preds, 2) == 0.5

        actuals = [['a', 'b']]
        preds = [['c', 'a']]
        assert metrics.mean_average_precision(actuals, preds, 2) == 0.25

    def test_all_correct_case(self):
        actuals = [[1, 2], [3, 4]]
        preds = [[1, 2], [3, 4]]
        assert metrics.mean_average_precision(actuals, preds, 2) == 1.0

    def test_zero_case(self):
        actuals = [['a', 'b'], ['c', 'd']]
        preds = [['c', 'd'], ['a', 'b']]
        assert metrics.mean_average_precision(actuals, preds, 2) == 0.0

    def test_partially_correct_case(self):
        actuals = [['a', 'b'], ['c', 'd']]
        preds = [['a', 'b'], ['a', 'b']]
        assert metrics.mean_average_precision(actuals, preds, 2) == 0.5


class TestEntropy:

    def test_basic_cases(self):
        # evenly split data
        assert metrics.entropy([1, 1, 2, 2]) == 1.0
        assert metrics.entropy([1.0, 1.0, 2.0, 2.0]) == 1.0
        assert metrics.entropy(['a', 'a', 'b', 'b']) == 1.0
        assert metrics.entropy((1, 1, 2, 2)) == 1.0

        # all one label
        assert metrics.entropy([1, 1, 1, 1]) == 0.0
        assert metrics.entropy(('a', 'a')) == 0.0

        # unevenly split
        assert metrics.entropy((1, 1, 1, 2)) - .811278 < 0.00001


class TestGiniImpurity:

    def test_basic_cases(self):
        # evenly split
        assert metrics.gini_impurity((1, 1, 0, 0)) == 0.5
        assert metrics.gini_impurity(['a', 'a', 'z', 'z']) == 0.5

        # all one label
        assert metrics.gini_impurity([1, 1, 1, 1]) == 0.0
        assert metrics.gini_impurity((9.0, 9.0, 9.0, 9.0)) == 0.0
        assert metrics.gini_impurity([0]) == 0.0

        # unevenly split
        assert metrics.gini_impurity([1, 0, 0, 0, 0, 0]) - 0.277777 < 0.00001


class TestMisclassificationError:

    def test_basic_cases(self):
        # evenly split
        assert metrics.misclassification_error([1, 1, 0, 0]) == 0.5
        assert metrics.gini_impurity(['a', 'a', 'z', 'z']) == 0.5

        # all one label
        assert metrics.misclassification_error(['a']) == 0
        assert metrics.misclassification_error([1, 1, 1]) == 0

        # unevenly split
        assert (metrics.misclassification_error([1, 0, 0, 0, 0, 0])
                - 0.166667) < 0.00001


class TestPrecision:

    def test_0_tp(self):
        assert metrics.precision(true_positive=0, false_positive=1) == 0.0
    def test_0_fp(self):
        assert metrics.precision(true_positive=1, false_positive=0) == 1.0
    def test_basic_cases(self):
        assert metrics.precision(true_positive=2, false_positive=2) == 0.5
        assert metrics.precision(true_positive=10, false_positive=5) - 0.666666 < 0.00001


class TestRecall:

    def test_0_fn(self):
        assert metrics.recall(true_positive=1, false_negative=0) == 1.0
    def test_0_tp(self):
        assert metrics.recall(true_positive=0, false_negative=1) == 0.0
    def test_basic_cases(self):
        assert metrics.recall(true_positive=2, false_negative=2) == 0.5
        assert metrics.recall(true_positive=10, false_negative=5) - 0.666666 < 0.00001
