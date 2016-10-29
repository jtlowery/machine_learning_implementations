from machine_learning_implementations.knn import k_nearest_neighbors

class TestKNN:

    def test_single_datapoint(self):
        train_point = [0.0, 0.0]
        train_label = 2
        knn = k_nearest_neighbors.KNearestNeighbors(n_neighbors=1)
        knn = knn.fit([(train_point, train_label)])
        assert knn.predict(train_point) == 2

    def test_two_datapoints(self):
        p1 = [1.0, 1.0]
        l1 = 1
        p2 = [-1.0, -1.0]
        l2 = -1
        knn = k_nearest_neighbors.KNearestNeighbors(n_neighbors=1)
        knn = knn.fit([(p1, l1), (p2, l2)])
        assert knn.predict(p1) == 1
        assert knn.predict(p2) == -1
        assert knn.predict([0.5, 0.5]) == 1
        assert knn.predict([-0.5, -0.5]) == -1

    def test_multiple_datapoints(self):
        p1 = [1, 1]
        l1 = 1
        p2 = [-1, 1]
        l2 = 2
        p3 = [-1, -1]
        l3 = 3
        p4 = [1, -1]
        l4 = 4
        knn = k_nearest_neighbors.KNearestNeighbors(n_neighbors=1)
        knn = knn.fit([(p1, l1), (p2, l2), (p3, l3), (p4, l4)])

        # predict on trained points
        assert knn.predict(p1) == 1
        assert knn.predict(p2) == 2
        assert knn.predict(p3) == 3
        assert knn.predict(p4) == 4

        # predict on new points
        assert knn.predict([0.5, 0.5]) == 1
        assert knn.predict([0.1, 0.1]) == 1
        assert knn.predict([0.25, 0.25]) == 1

        assert knn.predict([-0.5, 0.5]) == 2
        assert knn.predict([-0.1, 0.111]) == 2
        assert knn.predict([-0.333, 0.333]) == 2

        assert knn.predict([-0.5, -0.5]) == 3
        assert knn.predict([-0.1, -0.1]) == 3
        assert knn.predict([-0.75, -0.1]) == 3

        assert knn.predict([0.5, -0.5]) == 4
        assert knn.predict([0.05, -0.05]) == 4
        assert knn.predict([1.1, -1.1]) == 4

    def test_multiple_k(self):
        p1 = [1.0, 0.0]
        p2 = [5.0, 1.0]
        p3 = [2.0, -1.0]
        p4 = [-1.0, 0.0]
        p5 = [-5.0, -1.0]
        p6 = [-2.0, 1.0]
        group1 = zip([p1, p2, p3], (1, 1, 1))
        group2 = zip([p4, p5, p6], (2, 2, 2))
        knn = k_nearest_neighbors.KNearestNeighbors(n_neighbors=3)
        knn = knn.fit(list(group1) + list(group2))
        assert knn.predict([1.0, 0.0]) == 1
        assert knn.predict([-1.0, 0.0]) == 2
        assert knn.predict([0.6, 0]) == 1
        assert knn.predict([6.0, -1.0]) == 1
        assert knn.predict([-3, 0.0]) == 2

    def test_voting(self):
        p1 = [0, 0]
        p2 = [0, 0]
        p3 = [0, 0]
        knn = k_nearest_neighbors.KNearestNeighbors(n_neighbors=3)
        knn = knn.fit(zip([p1, p2, p3], (1, 1, 2)))
        assert knn.predict([0, 0]) == 1
