import machine_learning_implementations.lin_alg.lin_alg as lin_alg


class TestSquaredDistance:

    def test_zero_case(self):
        x = [1.0]
        y = [1.0]
        assert lin_alg.squared_distance(x, y) == 0.0

        x = [1, -1]
        y = [1, -1]
        assert lin_alg.squared_distance(x, y) == 0.0

    def test_nonzero_case(self):
        x = [1]
        y = [3]
        assert lin_alg.squared_distance(x, y) == 4.0

        x = [1.0, 1.0]
        y = [3.0, 5.0]
        assert lin_alg.squared_distance(x, y) == 20.0


class TestMeanVectors:

    def test_single_element_case(self):
        assert lin_alg.mean_vectors([[1.0]]) == [1.0]
        assert lin_alg.mean_vectors([[1, 0]]) == [1, 0]
        assert lin_alg.mean_vectors([[-9.0, 1.0, -4.5]]) == [-9.0, 1.0, -4.5]

    def test_multiple_element_case(self):
        vecs = [[1, 2, 3], [1, 2, 3]]
        assert lin_alg.mean_vectors(vecs) == [1, 2, 3]
        vecs = [[1.0, 2.0, 9.0], [2.0, 0.0, 0.0]]
        assert lin_alg.mean_vectors(vecs) == [1.5, 1.0, 4.5]

