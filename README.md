# CS365 LabC

This Python script implements a simple decision tree classifier using the ID3 algorithm. The script takes a tab-separated input file containing training data, where each row represents an example with features and a target classification. The decision tree is built based on the training data and then evaluated using leave-one-out cross-validation.

## Requirements
- Python 3.x
- pandas
- numpy

## Usage

To run the code, you can use the `main.py` script with the following options:

- `-i`: Name of file containing dataset.

For example, to run the code with the dataset tennis.txt:

```
python main.py -i tennis.txt
```

## Input File Format
The last column should contain the target classification, and the rest of the columns represent features.

## Output
The script outputs the decision tree in a format using indents to represent the levels of the tree and the accuracy of the classifier using leave-one-out cross-validation.

Example output using dataset tennis.txt:

```
|-- outlook =  split
||-- overcast =  yes
||-- rain =  split
||   |-- strong =  no
||   |-- weak =  yes
||-- sunny =  split
||   |-- high =  no
||   |-- normal =  yes
Accuracy: 0.7857142857142857
```



