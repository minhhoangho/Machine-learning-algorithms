import numpy as np

training_data = [
    ['Green', 3, 'Mango'],
    ['Yellow', 3, 'Mango'],
    ['Red', 1, 'Grape'],
    ['Red', 1, 'Grape'],
    ['Yellow', 3, 'Lemon']
]

# header
header = ["color", "diameter", "label"]


def unique_vals(rows, col):
    return set([row[col] for row in rows])


x = unique_vals(training_data, 0)
print(x)


def class_count(rows):
    # count the number of each type
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # compare the feature value in an example to the feature value in current question

        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if is_numeric(self.value):
            condition = ">="

        return "Is %s %s %s?" %(header[self.column], condition, str(self.value))

###
# partition dataset
def partition(rows, question_node):
    true_rows, false_rows = [], []
    print(question_node)
    for row in rows:
        if question_node.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return  true_rows, false_rows


def gini(rows):
    # calculate the Gini Impurity for a list of row.
    counts = class_count(rows)
    impurity = 1
    for label  in counts:
        prob_of_label = counts[label]/float(len(rows))
        impurity-=prob_of_label**2

    return impurity


def info_gain(left, right, current_uncertainty):
    # Infomation gain.
    # THe uncertain of starting node, minus the weight impurity of two children

    p = float(len(left))/ (len(left) + len(right))
    return current_uncertainty - p *gini(left) - (1-p) *gini(right)


def find_best_split(rows):
    """Find best condition to split dataset"""

    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        # values = unique_vals(rows, col)
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)

            #try splitting dataset

            true_rows, false_rows = partition(rows, question_node=question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue


            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_question, best_question



class Leaf:
    # Leaf Node
    def __init__(self, rows):
        self.predictions = class_count(rows)


class Decision_Node:

    # decision node with question

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    # Build decision tree using recursion

    gain, quesion = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question_node=quesion)

    #recursive build the true branch
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Decision_Node(quesion, true_branch, false_branch)

def print_tree(node, spacing = ""):

    if isinstance(node,Leaf):
        print(spacing+ "Predict", node.predictions)
        return

    print(spacing + str(node.question))

    print(spacing+ '--->  True:')

    print_tree(node.true_branch, spacing= spacing+ "  ")

    print_tree(spacing+ "---> False:")
    print_tree(node.false_branch, spacing= spacing+ "  ")


def classify(row, node):

    if isinstance(node, Leaf):
        return node.predictions


    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    #Print predictions at a leaf.
    total= sum(counts.values()) * 1.0
    probs = {}
    for label in counts.keys():
        probs[label] = str(int(counts[label]/total *100)) + "%"
    return probs


if __name__ == '__main__':
    my_tree = build_tree(training_data)
    print_tree(my_tree)