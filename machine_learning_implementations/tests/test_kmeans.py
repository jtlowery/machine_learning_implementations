from machine_learning_implementations.clustering.kmeans import KMeans


class TestKMeans:

    def test_single_point_single_cluster(self):
        classifier = KMeans(n_clusters=1, seed=1)
        assert classifier.n_clusters == 1
        assert classifier.seed == 1
        classifier = classifier.fit(data=[[1]])
        assert classifier.predict([1]) == 0
        assert classifier.predict([2]) == 0

    def test_two_points_two_clusters(self):
        classifier = KMeans(n_clusters=2, seed=1)
        classifier = classifier.fit(data=[[-1], [1]])
        assert classifier.predict([-1]) != classifier.predict([1])
        assert classifier.predict([-2]) == classifier.predict([-1])
        assert classifier.predict([3]) == classifier.predict([1])

    def test_four_points_four_clusters(self):
        classifier = KMeans(n_clusters=4, seed=1)
        assert classifier.n_clusters == 4
        p1 = [1.0, 1.0]
        p2 = [-1.0, -1.0]
        p3 = [-1.0, 1.0]
        p4 = [1.0, -1.0]
        classifier = classifier.fit([p1, p2, p3, p4])
        p1_class = classifier.predict(p1)
        p2_class = classifier.predict(p2)
        p3_class = classifier.predict(p3)
        p4_class = classifier.predict(p4)
        assert len(set((p1_class, p2_class, p3_class, p4_class))) == 4
        assert classifier.predict([2.0, 2.0]) == p1_class
        assert classifier.predict([-0.5, -0.5]) == p2_class
        assert classifier.predict([-1.1, 1.0]) == p3_class
        assert classifier.predict([0.9, -2.0]) == p4_class

    def test_fewer_clusters_than_points(self):
        classifier = KMeans(n_clusters=2, seed=1)
        # points for a cluster
        p1 = [1, 1]
        p2 = [2, 2]
        p3 = [1, 2]
        p4 = [2, 1]
        p5 = [3, 3]
        # points for another cluster
        p6 = [-1, -1]
        p7 = [-2, -2]
        p8 = [-3, -3]
        p9 = [-2, -1]
        p10 = [-3, -2]
        data = [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]
        classifier = classifier.fit(data)

        # check the points have been clustered as expected
        trained_classes = {classifier.predict(point) for point in data}
        assert len(trained_classes) == 2
        assert classifier.predict(p1) != classifier.predict(p7)
        classes_p1_to_5 = {classifier.predict(point) for point in data[0:5]}
        assert len(classes_p1_to_5) == 1
        classes_p6_to_10 = {classifier.predict(point) for point in data[5:]}
        assert len(classes_p6_to_10) == 1

        # check some new points
        assert classifier.predict([4, 4]) == classifier.predict([1, 1])
        assert classifier.predict([3, 2]) == classifier.predict([1, 1])
        assert classifier.predict([-5, -4]) == classifier.predict([-1, -1])
        assert classifier.predict([-2, -4]) == classifier.predict([-1, -1])

