import typing
import random
from machine_learning_implementations.lin_alg.lin_alg import squared_distance, mean_vectors

AnyNum = typing.TypeVar('AnyNum', int, float)


class KMeans:

    def __init__(self, n_clusters: int, seed: int):
        self.n_clusters = n_clusters
        self.seed = seed
        self.initial_means = None
        self.cluster_means = None
        self.labels = None
        self.data = None

    def fit(self, data: typing.Iterable[typing.Iterable[AnyNum]]):
        random.seed(self.seed)
        self.data = data

        # seed clusters with randomly chosen data points
        sample_indexes = random.sample(range(len(self.data)),
                                        self.n_clusters)
        self.initial_means = [self.data[index] for index in sample_indexes]
        self.cluster_means = [list(x) for x in self.initial_means]

        converged = False
        while not converged:
            # classify points
            new_labels = [self.predict(x) for x in self.data]
            # if labels are same as last run it has converged
            if self.labels == new_labels:
                converged = True

            self.labels = list(new_labels)

            # if convergence hasn't been reached
            # recalculate the cluster means
            if not converged:
                for cluster_idx, _ in enumerate(self.cluster_means):
                    # find all data with this label
                    cluster_data = []
                    for data_idx, label in enumerate(self.labels):
                        if label == cluster_idx:
                            cluster_data.append(self.data[data_idx])
                    # calc the new mean for this cluster
                    self.cluster_means[cluster_idx] = mean_vectors(cluster_data)

        return self

    def predict(self, data_point: typing.Iterable[AnyNum]):
        return min((idx for idx, _ in enumerate(self.cluster_means)),
            key = lambda x: squared_distance(data_point, self.cluster_means[x]))

