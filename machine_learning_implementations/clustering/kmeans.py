import typing
import random

AnyNum = typing.TypeVar('AnyNum', int, float)


class KMeans:

    def __init__(self, n_clusters: int, seed: int):
        self.n_clusters = n_clusters
        self.seed = seed
        self.means = None
        self.data = None

    def fit(self, data: typing.Iterable[typing.Iterable[AnyNum]]):
        random.seed(self.seed)
        self.data = data
        self.means = random.sample(data, self.n_clusters)
        return self

    #def predict(self, datapoint: typing.Iterable[AnyNum]):
