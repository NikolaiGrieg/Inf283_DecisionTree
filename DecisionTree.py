from collections import Counter
from queue import Queue

import numpy as np
import pandas as pd
from anytree import Node, RenderTree, PreOrderIter
from sklearn.model_selection import train_test_split


def contains_multi_class(lst):
    """
    Checks if a list contains more than one unique value
    :return: Boolean
    """
    return lst[1:] == lst[:-1]


def information_gain(impurity_base, df, impurity_measure):
    """
    Calculates information gain from splitting on the given category
    :param impurity_base: entropy or gini of y as float
    :param df: the split dataset as pandas dataframe
    :param impurity_measure: entropy or gini as string
    :return: Information Gain as float between 0-1
    """
    impurity_dict = calculate_impurity_dict(df, impurity_measure)
    IG = impurity_base
    for key, value in impurity_dict.items():
        # Calculate weighted sum over variables
        count = df.iloc[:, 1].tolist().count(key)
        total = len(df.iloc[:, 1].tolist())
        intermediate = count / total * value
        IG -= intermediate
    return IG


def calculate_impurity_base(ser, impurity_measure):
    """
    Calculates impurity as entropy or gini of y
    :param ser: labels / classes in dataset in pandas serie
    :param impurity_measure: entropy or gini as string
    :return: Impurity as measured by impurity_measure, float between 0-1
    """
    lst = ser.tolist()
    categories = np.unique(lst)
    impurity = 0
    if impurity_measure == 'gini':
        impurity = 1

    for cat in categories:
        count = lst.count(cat)
        total = len(lst)
        if impurity_measure == 'gini':
            intermediate = intermediate_gini(count, total)
            impurity -= intermediate
        else:
            intermediate = intermediate_entropy(count, total)
            impurity -= intermediate

    return impurity


def calculate_impurity_dict(df, impurity_measure):
    """
    Intermediate calculations for Information Gain.

    Calculates impurity of each value of the given category and stores the information in a dict.
    :param df: Dataframe containing the category and corresponding y values.
    :param impurity_measure: entropy or gini as string
    :return: dictionary of impurities, with the values of the category as keys
    """
    categories = np.unique(df.iloc[:, 1])
    impurity_dict = {}
    for cat_x in categories:
        # calculate weighted sum of impurity for each column
        impurity = 0
        if impurity_measure == 'gini':
            impurity = 1
        for cat_y in np.unique(df.iloc[:, 0]):
            col_name = df.columns.values[1]
            split = df.loc[df[col_name] == cat_x]
            count = split.iloc[:, 0].tolist().count(cat_y)  # count of each category in y
            total = len(split)
            if count is not 0 and total is not 0:
                if impurity_measure == 'gini':
                    intermediate = intermediate_gini(count, total)
                else:
                    intermediate = intermediate_entropy(count, total)
                impurity -= intermediate

            impurity_dict[cat_x] = impurity

    return impurity_dict


def intermediate_gini(count, total):
    """
    Formula for calculating gini index
    :returns: float between 0 and 1
    """
    return np.power(count / total, 2)


def intermediate_entropy(count, total):
    """
    Formula for calculating entropy
    :returns: float between 0 and 1
    """
    return (count / total) * np.log2(count / total)


def recurse(X, y, parent, impurity_measure='entropy'):
    """
    Implementation of ID3 algorithm.

    Recursively splits dataset on highest information gain.
    Nodes are connected to the root given as input in the first call, so nothing is returned.
    :param X: Input data as pandas dataframe
    :param y: Input labels as pandas serie
    :param parent: The parent node used for building the tree. Initial input should be root
    :param root: Root node of the tree as type Node in library anytree
    :param impurity_measure: gini or entropy as string
    """
    if contains_multi_class(y.tolist()):
        # return leaf with label
        Node(y.tolist()[0], parent=parent, M=y.tolist(), error=0, node_type='class')
        return

    # check if all data points have same features
    features_same = True
    for col in X.columns:
        feature = X[col].tolist()
        if not contains_multi_class(feature):
            features_same = False
            break

    if features_same:
        # calculate most common label
        majority_label = y.mode()[0]
        Node(majority_label, parent=parent, node_type='class')
        return

    # calculate information gain and choose feature
    else:
        igs = {}
        for col in X.columns:
            feature = X[col]  # one feature, all data points
            pair = pd.concat([y, feature], axis=1)

            impurity_base = calculate_impurity_base(y, impurity_measure)
            IG = information_gain(impurity_base, pair, impurity_measure)

            igs[col] = IG

            # maybe fix this to be binary tree
        build_tree(X, y, igs, parent)


def build_tree(X, y, ig_dict, parent):
    """
    Continuation of recurse function.

    Builds the tree by selecting the category with highest information gain and calling recurse with the dataset
    of all unique variables in the category.
    :param X: Data as pandas dataframe
    :param y: labels as pandas sequence
    :param ig_dict: dictionary of category-information gain at current iteration
    :param parent: parent node as type Node in library anytree
    :param root: Root node as type Node in library anytree
    """
    best_selection = max(ig_dict.keys(), key=(lambda k: ig_dict[k]))

    parent.split_on = best_selection
    feature = X[best_selection]
    categories = np.unique(feature)

    c = Counter(y.tolist())
    majority_y = c.most_common()[0]
    split_category = Node(best_selection, parent=parent, M=majority_y, error=0, node_type='decision_state')

    for cat in categories:
        pair = pd.concat([y, X], axis=1)
        selected_rows = pair.loc[feature == cat]
        selected_X = selected_rows.iloc[:, 1:]
        selected_X.drop(best_selection, axis=1, inplace=True)
        selected_y = selected_rows.iloc[:, 0]

        c = Counter(selected_y.tolist())
        majority_y = c.most_common()[0]
        node = Node(cat, parent=split_category, M=majority_y, error=0, node_type='decision_edge')

        recurse(selected_X, selected_y, node)


def learn(X, y, impurity_measure='entropy', pruning=False):
    """
    Fit decision tree to data set
    :param X: Observations as pandas dataframe
    :param y: Labels as pandas sequence
    :param impurity_measure: Entropy or gini as string
    :param pruning: Boolean to determine reduced error pruning
    :return: Root node of trained tree of type Node in library anytree
    """
    root = Node("Root", M=None, error=0, node_type='root')

    if pruning:
        # Could have implemented cross validation for this part to get more accurate pruning
        X_train, X_prune, y_train, y_prune = train_test_split(X, y, test_size=0.2, random_state=1)

        c = Counter(y_train.tolist())
        root.M = c.most_common()[0]

        recurse(X_train, y_train, root, impurity_measure=impurity_measure)
        prune(root, X_prune, y_prune)
    else:
        c = Counter(y.tolist())
        root.M = c.most_common()[0]
        recurse(X, y, root, impurity_measure=impurity_measure)

    return root


def prune(root, X_prune, y_prune):
    """
    Implementation of reduced error pruning.

    Compares prediction errors in pruning set on nodes and children.
    :param root: Root node of trained tree to be pruned, as class Node in the library anytree.
    :param X_prune: Pruning set datapoints as pandas dataframe
    :param y_prune: Pruning set labels as pandas sequence
    """
    accuracy = calculate_accuracy(root, X_prune, y_prune)
    # print('Accuracy on pruning set before pruning: ', accuracy) # Enable to print before pruning accuracy
    Q = Queue()
    q_list = []  # to be able to check if items are in que

    # put all leaves in que
    for node in PreOrderIter(root):
        if node.is_leaf:
            Q.put(node)
            q_list.append(node)

    while not Q.empty():
        node = Q.get()
        q_list.remove(node)
        if not node.is_root:
            parent = node.parent
            if not q_list.__contains__(parent):
                Q.put(parent)
                q_list.append(parent)
            if not node.is_leaf:
                # find sum of errors on descendants
                descendants = node.descendants
                sum_error_descendants = 0  # R
                for des in descendants:
                    sum_error_descendants += des.error
                if sum_error_descendants >= node.error:  # if R >= E
                    if not len(node.children) == 1:
                        node.name = node.M[0]
                        node.children = []
                        node.node_type = 'class'


def calculate_accuracy(root, X_prune, y_prune):
    """
    Calculates errors on each node and returns overall accuracy.
    :param root: Root of trained tree
    :param X_prune: Pruning set datapoints as pandas dataframe
    :param y_prune: Pruning set labels as pandas sequence
    :return: Mean accuracy of predictions on prune set as float between 0 and 1
    """
    y_prune = y_prune.tolist()
    for i in range(len(y_prune)):
        # if class is wrong, give +1 error to node
        row = X_prune.iloc[i]
        node = traverse_downwards(root, row)
        # traverse upwards, and give +1 error to any node (not decision edge) with mismatching M to class
        while True:
            if not node.node_type == 'decision_edge':
                if node.M[0] != y_prune[i]:
                    node.error += 1

                if node.is_root:
                    break
            node = node.parent

    total_error = 0
    for node in PreOrderIter(root):
        if node.is_leaf:
            total_error += node.error

    return 1 - (total_error / len(y_prune))


def predict(X, root):
    """
    Predicts new label based on data point X
    :param X: pandas dataframe with rows containing observations
    :param root: Root node of the trained tree as type Node in the library anytree
    :return: list containing predicted class for each row in X
    """
    results = []
    for row in X.iterrows():
        results.append(traverse_downwards(root, row[1]).name)

    return results


def traverse_downwards(root, row):
    """
    Returns predicted class based on a one observation, as node
    :param root: Root of trained tree as type Node in the library anytree
    :param row: Row in dataset as pandas sequence
    :return: Leaf node after traversal type Node in the library anytree
    """
    node = root
    while not node.is_leaf:
        # determine path
        children = node.children
        if node.node_type == 'decision_state':
            # select child
            for c in children:
                if c.name == row.loc[node.name]:
                    node = c
                    break
        else:
            node = children[0]
    return node


def print_tree(root, show_variables=False):
    """
    Prints the tree in console

    Predicted classes are printed in brackets. Decision edges are printed as nodes between
    decision nodes and predicted classes.
    :param root: Root node of the trained tree of type Node in the library anytree
    :param show_variables: Default false, if true, majority class and error will be printed with each node
    """
    if show_variables:
        for pre, fill, node in RenderTree(root):
            if not node.is_leaf:
                print("%s%s M:%s (E: %s) --%s" % (pre, node.name, node.M[0], node.error, node.node_type))
            else:
                print("%s[%s (E: %s)] --%s" % (pre, node.name, node.error, node.node_type))
    else:
        for pre, fill, node in RenderTree(root):
            if not node.is_leaf:
                print("%s%s" % (pre, node.name))
            else:
                print("%s[%s]" % (pre, node.name))
