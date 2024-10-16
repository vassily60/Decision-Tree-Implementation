class Node:
    def __init__(self, df, attribute, remaining_attribute, val, category, yes_no):
        self._df = df
        self._attribute = attribute
        self._remaining_attribute = remaining_attribute
        self._child = []
        self._val = val
        self._category = category
        self._yes_no = yes_no

    def append(self, value):
        return self._child.append(value)