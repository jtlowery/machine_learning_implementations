import typing
from machine_learning_implementations.metrics.general import entropy


class Leaf:
    def __init__(self, label: int):
        self.label = label

    def predict(self, data: typing.Iterable):
        return self.label


class Branch:
    def __init__(self,
                 left,
                 right,
                 depth,
                 curr_info_gain,
                 split_col,
                 split_value,
                 split_rows):
        """

        Parameters
        ----------
        left - left subtree
        right - right subtree
        depth - current depth
        curr_info_gain - info gain at this split
        split_col - column index of original data being split on
        split_value - value to make the split on
        split_rows - row indexes of original data used
        """
        self.left = left
        self.right = right
        self.depth = depth
        self.curr_info_gain = curr_info_gain
        self.split_col = split_col
        self.split_value = split_value
        self.split_rows = split_rows

    def predict(self, data: typing.Iterable):
        if isinstance(self.value, int):
            if data[self.split_col] == self.split_value:
                return self.left.predict(data)
            else:
                return self.right.predict(data)
        if isinstance(self.split_value, float):
            if data[self.split_col] < self.split_value:
                return self.left.predict(data)
            else:
                return self.right.predict(data)




def information_gain(parent_labels: typing.Iterable,
                     left_split_labels: typing.Iterable,
                     right_split_labels) -> float:
    total_length = len(parent_labels)
    left_split_entropy = (len(left_split_labels) / total_length) * entropy(left_split_labels)
    right_split_entropy = (len(right_split_labels) / total_length) * entropy(right_split_labels)
    return entropy(parent_labels) - (left_split_entropy + right_split_entropy)

