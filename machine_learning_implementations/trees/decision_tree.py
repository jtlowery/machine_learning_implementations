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
        self.tree = None

    def fit(self, data: typing.List[typing.List], labels: typing.List):
        continue_splitting = True
        while continue_splitting:
            best_col_split = self._find_best_split_on_data(data)
            if best_col_split is None:
                continue_splitting = False
            else:
                Branch()
        return self

    def _find_best_split_on_data(self,
                                 data: typing.List[typing.List],
                                 row_indexes: typing.List):
        # finds best split over all columns/features
        num_columns = len(data[0])
        possible_col_splits = []
        for col in range(num_columns):
            column_data = [row[col] for row in data]
            if isinstance(column_data[0], int):
                # split as categorical
                split = self._split_categorical_feature(column_data, row_indexes)
            elif isinstance(column_data[0], float):
                # split as continuous
                split = self._split_continuous_feature(column_data, row_indexes)
            else:
                raise TypeError('Expected either float or int')
            if split:
                possible_col_splits.append((col, split))
        if len(possible_col_splits) == 0:
            return None
        else:
            return max(possible_col_splits, key=lambda x: x[1]['info_gain'])

    def _split_categorical_feature(self,
                                   column_data: typing.List,
                                   labels: typing.List,
                                   row_indexes: typing.List):
        # finds best split on a categorical column/feature
        possible_vals = set(column_data)
        possible_splits = []
        for possible_val in possible_vals:
            left_labels, right_labels = [], []
            left_indexes, right_indexes = [], []
            for pt, label, row_idx in zip(column_data, labels, row_indexes):
                if pt == possible_val:
                    left_labels.append(label)
                    left_indexes.append(label)
                else:
                    right_labels.append(pt)
                    right_indexes.append(pt)
            # ensure both splits have at least min_samples_split
            if (len(left_labels) < self.min_samples_split or
               len(right_labels) < self.min_samples_split):
                continue
            split_detail = {'left_labels': left_labels,
                            'left_row_indexes': left_indexes,
                            'right_labels': right_labels,
                            'right_row_indexes': right_indexes,
                            'split_value': possible_val,
                            'info_gain': information_gain(column_data,
                                                          left_labels,
                                                          right_labels)}
            possible_splits.append(split_detail)
        if len(possible_splits) == 0:
            return None
        else:
            return max(possible_splits, key=lambda x: x['info_gain'])

    def _split_continuous_feature(self,
                                  column_data: typing.List,
                                  labels: typing.List,
                                  row_indexes: typing.List):
        # finds best split on a continuous feature
        raise NotImplementedError

    def predict(self, data: typing.List):
        return self.tree.predict(data)

def information_gain(parent_labels: typing.List,
                     left_split_labels: typing.List,
                     right_split_labels) -> float:
    total_length = len(parent_labels)
    left_split_entropy = ((len(left_split_labels) / total_length) *
                          entropy(left_split_labels))
    right_split_entropy = ((len(right_split_labels) / total_length) *
                           entropy(right_split_labels))
    return entropy(parent_labels) - (left_split_entropy + right_split_entropy)

