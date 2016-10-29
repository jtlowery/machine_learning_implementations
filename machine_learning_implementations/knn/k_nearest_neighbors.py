import typing
from math import sqrt
from collections import Counter

AnyNum = typing.TypeVar('AnyNum', int, float)


def distance_between_points(p1: typing.Iterable[AnyNum],
                            p2: typing.Iterable[AnyNum]) -> float:
    squared_differences = 0.0
    for p1_elem, p2_elem in zip(p1, p2):
        squared_differences += (p1_elem - p2_elem) ** 2
    return sqrt(squared_differences)


class KNearestNeighbors:

    def __init__(self, n_neighbors: int):
        self.n_neighbors = n_neighbors
        self.data = None

    def fit(self, data: typing.Iterable[tuple]):
        """

        Parameters
        ----------
        data - an array-like of tuples of form (point, label)

        Returns
        -------
        self
        """
        self.data = data
        return self

    def predict(self, row_data: typing.Iterable):
        sorted_data = sorted(self.data,
                             key=lambda x: distance_between_points(row_data, x[0]))
        k_nearest = sorted_data[:self.n_neighbors]
        label_counter = Counter(label for (point, label) in k_nearest)
        return label_counter.most_common(1)[0][0]
