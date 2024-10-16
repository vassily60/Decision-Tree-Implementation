import pandas as pd
import math 
import numpy as np
import argparse
import os
import sys
from util import Node

# features, classification = df.iloc[:,:len(df.columns)-1], df.iloc[:, -1:]


def entropy(df, attribute, category):
    if attribute in df.columns:
        count_no = len(df[(df[attribute] == category) & (df.iloc[:, -1] == 'no')])
        count_yes = len(df[(df[attribute] == category) & (df.iloc[:, -1] == 'yes')])
        q = count_yes/(count_yes+count_no)
        if q == 0 or 1-q == 0:
            return 0 
        else:
            result = -(q * math.log2(q) + (1-q) * math.log2(1-q))
            return result
    else:
        raise ValueError("feature doesn't exist")

def entropy_parent(df):
    count_no = len(df[(df.iloc[:, -1] == 'no')])
    count_yes = len(df[(df.iloc[:, -1] == 'yes')])
    q = count_yes/(count_yes+count_no)
    if q == 0 or 1-q == 0:
        return 0 
    else:
        result = -(q * math.log2(q) + (1-q) * math.log2(1-q))
        return result
    


def remainder(df, attribute):
    categories = np.unique(df[attribute])
    tot_count = len(df[attribute])
    result = 0
    for i in categories:
        one_entropy = entropy(df, attribute, i)
        count = len(df[(df[attribute] == i)])
    
        result += (count/tot_count) * one_entropy

    return result

def gain(df, attribute):
    return entropy_parent(df) - remainder(df, attribute)

def highest_gain(df):
    min = 0
    attribute = None
    for i in df.iloc[:,:-1].columns:
        value = gain(df, i)
        if value > min:
            min = value
            attribute = i

    columns_to_keep = [col for col in df.columns if col != attribute]

    return attribute, columns_to_keep, min

def check_same_except_classification(df):
    first_n_minus_1_cols = df.iloc[:, :-1]
    all_same_except_last = first_n_minus_1_cols.apply(lambda x: x.nunique()) == 1
    last_col_unique = df.iloc[:, -1].nunique() > 1
    return all(all_same_except_last) and last_col_unique

def tree(df):
    attribute, remaining_attribute, val = highest_gain(df)
    root = Node(df, attribute, remaining_attribute, val, None, None)

    build_tree(root)

    return root

def build_tree(node):
    if len(node._df.columns) == 1 or node._attribute is None:  # If there's only one column left, it's the target variable
        return

    categories = np.unique(node._df[node._attribute])

    for category in categories:
        sub_df = node._df[node._df[node._attribute] == category]
        unique_values = sub_df.iloc[:, -1].unique()

        if check_same_except_classification(sub_df): #If all attributes are the same but have different classifications it returns no
            child = Node(sub_df, None, None, None, category, "no") 
        elif len(unique_values) == 1 and unique_values[0] == 'yes':  # If all values in the target column are the same
            child = Node(sub_df, None, None, None, category, "yes")  # Leaf node
        elif len(unique_values) == 1 and unique_values[0] == 'no':
            child = Node(sub_df, None, None, None, category, "no")
            
        else:
            attribute, remaining_attribute, val = highest_gain(sub_df)
            child = Node(sub_df, attribute, remaining_attribute, val, category, None)
            build_tree(child)  # Recursively build the subtree

        node._child.append(child)

def printTree(node, level=0):
    
    # Print the decision made at this node
    if level > 0:
        print("|", end="")
    if level == 0:
        print(f"|-- {node._attribute} =  split")
    if node._yes_no == None:
        node._yes_no = 'split'
    if node._category != None:
        print(f"{'|   ' * (level - 1)}|-- {node._category} =  {node._yes_no}")
    
    for child in node._child:
        printTree(child, level + 1)

    
def leave_one_out_cross_validation(df):
    correct_predictions = 0
    total_examples = len(df)

    for idx, example in df.iterrows():
        training_set = df.drop(idx)
        
        decision_tree = tree(training_set)
        
        predicted_classification = classify_example(decision_tree, example)
        if predicted_classification == example[-1]:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_examples
    return accuracy  

def classify_example(node, example):
    if node._child == []:  # Leaf node
        return node._yes_no

    attribute_value = example[node._attribute]
    for child in node._child:
        if child._category == attribute_value:
            return classify_example(child, example)

    return None         

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='File')
    parser.add_argument('-i', help='input file', required=True)
    args = parser.parse_args()

    #Check if the file exists
    if not (os.path.isfile(args.i)):
        print("error", args.i, "does not exist, exiting.", file = sys.stderr)
        exit(-1)
    df = pd.read_csv(args.i, delimiter= "\t")
   
  

    a = tree(df)

    printTree(a)

    accuracy = leave_one_out_cross_validation(df)
    print("Accuracy:", accuracy)

 
