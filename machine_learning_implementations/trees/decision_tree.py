import typing
from machine_learning_implementations.metrics.general import entropy


class Leaf:
    def __init__(self, label: int):
        self.label = label

    def predict(self, data: typing.List):
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

    def predict(self, data: typing.List):
        if isinstance(self.split_value, int):
            if data[self.split_col] == self.split_value:
                return self.left.predict(data)
            else:
                return self.right.predict(data)
        if isinstance(self.split_value, float):
            if data[self.split_col] < self.split_value:
                return self.left.predict(data)
            else:
                return self.right.predict(data)


class Tree:
    def __init__(self, max_depth: int, min_samples_split: int):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, data: typing.List[typing.List]):
        return self

    def _find_best_split_on_feature(self, column_data: typing.List):
        # finds best split on a feature/column
        if isinstance(column_data[0], int):
            # split as categorical
            pass
        if isinstance(column_data[0], float):
            # split as continuous
            pass

    def _split_categorical_feature(self, column_data: typing.List):
        possible_vals = set(column_data)
        for possible_val in possible_vals:
            left_split, right_split = [], []
            for pt in column_data:
                if pt == possible_val:
                    left_split.append(pt)
                else:
                    right_split.append(pt)
            # check both splits have at least min_samples_split
            if (len(left_split) < self.min_samples_split or
                        len(right_split) < self.min_samples_split):
                continue
            else:
                
        return NotImplemented

    def predict(self):
        return self

def information_gain(parent_labels: typing.Iterable,
                     left_split_labels: typing.Iterable,
                     right_split_labels) -> float:
    total_length = len(parent_labels)
    left_split_entropy = (len(left_split_labels) / total_length) * entropy(left_split_labels)
    right_split_entropy = (len(right_split_labels) / total_length) * entropy(right_split_labels)
    return entropy(parent_labels) - (left_split_entropy + right_split_entropy)

