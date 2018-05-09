from sklearn import tree
from sklearn.feature_extraction import DictVectorizer 
from sklearn import preprocessing 
import copy

def openfile(path, fname, test_path, test_fname):
    datafile = open(path + fname, "r")
    header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
    examples_list = []
    cls_list = []
    for line in datafile:
        line = line.strip()
        line = line.strip('.')
        # ignore empty lines. comments are marked with a |
        if len(line) == 0 or line[0] == '|':
            continue
        ex = [x.strip() for x in line.split(",")]
        cls_list.append(ex[-1])
        ex_dict = {}
        for i in range(len(header)-1):
            #if header[i] == 'fnlwgt':
            #    continue
            if ex[i].isdigit():
                ex[i] = float(ex[i])
            ex_dict[header[i]] = ex[i]
        
        examples_list.append(ex_dict)
        
    test_datafile = open(test_path + test_fname, "r")
    #header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "class"]
    test_examples_list = []
    test_cls_list = []
    for line in test_datafile:
        line = line.strip()
        line = line.strip('.')
        # ignore empty lines. comments are marked with a |
        if len(line) == 0 or line[0] == '|':
            continue
        ex = [x.strip() for x in line.split(",")]
        test_cls_list.append(ex[-1])
        ex_dict = {}
        for i in range(len(header)-1):
            #if header[i] == 'fnlwgt':
            #    continue
            if ex[i].isdigit():
                ex[i] = float(ex[i])
            ex_dict[header[i]] = ex[i]
        
        test_examples_list.append(ex_dict)
        
    all_examples_list = copy.deepcopy(examples_list)
    all_examples_list.extend(test_examples_list)
    all_cls_list = copy.deepcopy(cls_list)
    all_cls_list.extend(test_cls_list)
    
    vec = DictVectorizer()  
    all_dummyX = vec.fit_transform(all_examples_list).toarray()  
    
    lb = preprocessing.LabelBinarizer()  
    all_dummyY = lb.fit_transform(all_cls_list)
    
    dummyX = all_dummyX[0:len(examples_list)]
    dummyX_test = all_dummyX[len(examples_list):]
    
    dummyY = all_dummyY[0:len(examples_list)]
    dummyY_test = all_dummyY[len(examples_list):]
    
    return dummyX,dummyY,dummyX_test,dummyY_test
        

def trainDecisionTree(dummyX, dummyY):
    clf = tree.DecisionTreeClassifier() #创建一个分类器，entropy决定了用ID3算法  
    clf = clf.fit(dummyX, dummyY)
    return clf
    
def predictByDecisionTree(clf,dummyX, dummyY):
    predict_results = clf.predict(dummyX)
    correct = 0
    for i in range(len(predict_results)):
        if predict_results[i] == dummyY[i]:
            correct += 1
    print("{} out of {} correct ({:.2f}%)".format(correct, len(predict_results), correct/len(predict_results)*100))


if __name__ == '__main__':
    path = "data/"  #directory of your data
    datafile = "data_set.txt"
    testfile = "test_set.txt"
    dummyX,dummyY,dummyX_test,dummyY_test = openfile(path, datafile, path, testfile) # load the training set
    #数据特征，特征的类别
    #test_examples_list,test_cls_list = openfile(path, testfile) # load the test set
    clf = trainDecisionTree(dummyX,dummyY)
    
    predictByDecisionTree(clf,dummyX,dummyY)
    predictByDecisionTree(clf,dummyX_test,dummyY_test)
    