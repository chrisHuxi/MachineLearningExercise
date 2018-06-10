from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree

#load data set to type of pandas.DataFrame
#in  : 1.path 2.file name
#out : 1.data set of all features 2.class set
def openfile(path, fname):
    header = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country"]
    datafile = open(path + fname, "r")
    examples = []
    cls_set = []
    for line in datafile:
        line = line.strip()
        line = line.strip('.')
        # ignore empty lines. comments are marked with a |
        if len(line) == 0 or line[0] == '|':
            continue
        ex = [x.strip() for x in line.split(",")[:-1]]
        examples.append(ex)
        cls_set.append(line.split(",")[-1].strip())
    X = pd.DataFrame(examples,columns = header)
    Y = pd.Series(cls_set)
    return X,Y

#transform the data set with categorical features to one-hot data set
#in  : 1. training data 2. test data
#out : 1. transformed training data 2. transformed test data   
# ps. we should transform both of training data and test data at the same time,
# because it is possible that the features of only training data or test data 
# are not complete. for example, the "sex" feature in training data is all "female",
# but in test data we find a data, whose "sex" feature is "male". In this case, we 
# must know all possibility of this feature. That's why I load training data and test
# data separately in previous code(DT_skt.py)

def dataTransform(X_train,X_test):
    le=LabelEncoder()
    # Iterating over all the common columns in train and test
    for col in X_test.columns.values:
        # Encoding only categorical variables
        if X_test[col].dtypes=='object':
            # Using whole data to form an exhaustive list of levels
            data=X_train[col].append(X_test[col])
            le.fit(data.values)
            X_train[col]=le.transform(X_train[col])
            X_test[col]=le.transform(X_test[col])
            
    enc=OneHotEncoder(sparse=False)
    X_train_cont = X_train[["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]]
    X_test_cont = X_test[["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]]
    
    new_X_train = X_train_cont
    new_X_test = X_test_cont
    
    categorical_header = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
    for col in categorical_header:
        # creating an exhaustive list of all possible categorical values

        data = X_train[[col]].append(X_test[[col]])

        enc.fit(data)
        # Fitting One Hot Encoding on train data
        temp = enc.transform(X_train[[col]])
        # Changing the encoded features into a data frame with new column names
        temp = pd.DataFrame(temp,columns = [(col+"_"+str(i)) for i in data[col]
             .value_counts().index])
        # In side by side concatenation index values should be same
        # Setting the index values similar to the X_train data frame
        temp = temp.set_index(X_train.index.values)

        # adding the new One Hot Encoded varibales to the train data frame
        new_X_train = pd.concat([new_X_train,temp],axis=1)
        # fitting One Hot Encoding on test data
        temp = enc.transform(X_test[[col]])
        # changing it into data frame and adding column names
        temp = pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]
             .value_counts().index])
        # Setting the index for proper concatenation
        temp = temp.set_index(X_test.index.values)
        # adding the new One Hot Encoded varibales to test data frame
        new_X_test = pd.concat([new_X_test,temp],axis=1)
    
    return new_X_train,new_X_test
    
    
def trainDecisionTree(dummyX, dummyY):
    clf = tree.DecisionTreeClassifier() #create a classifier
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

    train_X,train_Y = openfile(path, datafile) # load the training set
    test_X,test_Y = openfile(path, testfile) # load the test set
    
    train_X,test_X = dataTransform(train_X,test_X)
    
    clf = trainDecisionTree(train_X,train_Y)
    predictByDecisionTree(clf,train_X,train_Y)
    predictByDecisionTree(clf,test_X,test_Y)

    
    
    
    