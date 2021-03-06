""" Machine Learning TUD 2018, Ex.3
 
Decision Tree Learning:
Based on american census data you want to predict two classes of income of people:
>50K$, <=50K$.

We do not use continuous attributes for this first decision tree task.
"""
__author__ = 'Benjamin Guthier'

from math import log

def openfile(path, fname):
    """opens the file at path+fname and returns a list of examples and attribute values.
    examples are returned as a list with one entry per example. Each entry then
    is a list of attribute values, one of them being the class label. The returned list attr
    contains one entry per attribute. Each entry is a list of possible values or an empty list
    for numeric attributes.
    """
    datafile = open(path + fname, "r")
    examples = []
    for line in datafile:
        line = line.strip()
        line = line.strip('.')
        # ignore empty lines. comments are marked with a |
        if len(line) == 0 or line[0] == '|':
            continue
        ex = [x.strip() for x in line.split(",")]
        examples.append(ex)

    attr = []
    for i in range(len(examples[0])):
        values = list({x[i] for x in examples}) # set of all different attribute values
        if values[0].isdigit():  # if the first value is a digit, assume all are numeric
            attr.append([])
        else:
            attr.append(values)
        
    return examples, attr


def calc_entropy(examples, cls_index):
    """calculates the entropy over all examples. The index of the class label in the example
    is given by cls_index. Can also be the index to an attribute.
    """
    cls_dict = {}
    for example in examples:
        if example[cls_index] in set(cls_dict.keys()):#faster
            cls_dict[example[cls_index]] = cls_dict[example[cls_index]]+1
        else:
            cls_dict[example[cls_index]] = 1
	
    result = 0
    for every_cls in cls_dict.keys():
        prob_one_kind = (cls_dict[every_cls]+0.0)/len(examples)
        if prob_one_kind == 0:
            result = result + 0
        else:
            result = result + -(prob_one_kind*log(prob_one_kind,2))
			
    return result


def calc_ig(examples, attr_index, cls_index):
    """Calculates the information gain over all examples for a specific attribute. The
    class index must be specified.
    
    uses calc_entropy
    """
	
    sub_examples_dict = {}
    for example in examples:
        if example[attr_index] in set(sub_examples_dict.keys()):
            sub_examples_dict[example[attr_index]].append(example)
        else:
            sub_examples_dict[example[attr_index]] = [example]
    remainder = 0.0
    for Tv in sub_examples_dict.keys():
        remainder += len(sub_examples_dict[Tv])/len(examples)*calc_entropy(sub_examples_dict[Tv], cls_index)
	
    result = calc_entropy(examples,cls_index) - remainder
		
    return result
    


def majority(examples, attr_index):
    """Returns the value of attribute "attr_index" that occurs the most often in the examples."""
    # create a flat list of all attribute values (with duplicates, so we can count)
    attr_vals = []
    for example in examples:
        attr_vals.append(example[attr_index])
    # among all unique attribute values, find the maximum with regards to occurrence in the attr_vals list
    return max(set(attr_vals), key=attr_vals.count)


def choose_best_attr(examples, attr_avail, cls_index):
    """Iterates over all available attributes, calculates their information gain and returns the one
    that achieves the highest. attr_avail is a list of booleans corresponding to the list of attributes.
    it is true if the attribute has not been used in the tree yet (and is not numeric).
    """
    igs = [] # list of information gains for each attribute
    for i in range(cls_index):
        if attr_avail[i] == True:
            igs.append(calc_ig(examples, i, cls_index))
        else:
            igs.append(-1.0)
    #print(igs)
    return igs.index(max(igs)) # return index of the attribute with highest IG


def dtree_learning(examples, attr_avail, default, cls_index):
    """Implementation of the decition tree learning algorithm according to the pseudo code
    in the lecture. Receives the remaining examples, the remaining attributes (as boolean list),
    the default label and the index of the class label in the attribute vector.
    Returns the root node of the decision tree. Each tree node is a tuple where the first entry is
    the index of the attribute that has been used for the split. It is "None" for leaf nodes.
    The second entry is a list of subtrees of the same format. The subtrees are ordered in the
    same way as the attribute values in "attr". For leaf nodes, the second entry is the predicted class.
    
    uses choose_best_attr, majority, dtree_learning
    """
    global attr
    if len(examples) == 0 :
        return default
    elif len(set([example[cls_index] for example in examples])) == 1:
        return examples[0][cls_index]
    elif set(attr_avail) == {False}:
        return majority(examples,cls_index)#这个函数不知道用对没有
    else:
        #print(attr_avail)
        best_attr_index = choose_best_attr(examples,attr_avail,cls_index)
        #print(best_attr_index)
        attr_avail[best_attr_index] = False
        subtree_list = []
        for v in attr[best_attr_index]:
            sub_examples = []
            for example in examples:
                if example[best_attr_index] == v:
                    sub_examples.append(example)
            subtree = dtree_learning(sub_examples,attr_avail,majority(examples,cls_index),cls_index)
            subtree_list.append(subtree)
            #print(subtree_list)
        tree = (best_attr_index,subtree_list)
    
        return tree
        

def dtree_classify(dtree, x):
    """Classifies a single example x using the given decision tree. Returns the predicted class label.
    """
    
    subtree_pos = attr[dtree[0]].index(x[dtree[0]])
    if isinstance(dtree[1][subtree_pos],str):
        return dtree[1][subtree_pos]
    else:
        return dtree_classify(dtree[1][subtree_pos], x) # descend into subtree recursively
    

def dtree_test(dtree, examples, cls_index):
    """Classify all examples using the given decision tree. Prints the achieved accuracy."""
    
    correct = 0
    for example in examples:
        if dtree_classify(dtree,example)==example[cls_index]:
            correct += 1
    
    print("{} out of {} correct ({:.2f}%)".format(correct, len(examples), correct/len(examples)*100))


path = "data/"  #directory of your data
datafile = "data_set.txt"
testfile = "test_set.txt"
examples, attr = openfile(path, datafile) # load the training set
#数据特征，特征的类别
test, test_attr = openfile(path, testfile) # load the test set


cls_index = len(attr)-1 # the last attribute is assumed to be the class label
#attr_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]

attr_avail = [] # marks which attributes are available for splitting (not numeric and not the class label)
for i in range(len(attr)):
    # the list attr[i] contains all possible values of attribute i. It is empty for numeric attributes.
    attr_avail.append(len(attr[i])>0 and i != cls_index)
#print(attr_avail)#表示该特征是否会用到

dtree = dtree_learning(examples, attr_avail, majority(examples,cls_index), cls_index)

dtree_test(dtree, examples, cls_index)
dtree_test(dtree, test, cls_index)
