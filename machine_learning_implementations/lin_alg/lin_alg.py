import typing

AnyNum = typing.TypeVar('AnyNum', int, float)


def squared_distance(x: typing.Iterable[AnyNum],
                     y: typing.Iterable[AnyNum]) -> float:
    return sum(((a - b) ** 2 for a, b in zip(x, y)))


def mean_vectors(vectors: typing.Iterable[typing.Iterable[AnyNum]]) -> typing.Iterable[AnyNum]:
    num_vectors = len(vectors)
    sums = [0] * len(vectors[0])
    for vector in vectors:
        for idx, val in enumerate(vector):
            sums[idx] += val
    return [sum_component / num_vectors for sum_component in sums]
