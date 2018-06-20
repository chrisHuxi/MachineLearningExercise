import numpy as np
from math import log
from math import pi
from sklearn.naive_bayes import GaussianNB

def openfile():
    csvpath = './data/'             
    csvname = 'Data3.csv'
    csvfile = csvpath + csvname
    print('Reading "' + csvfile + '":')
    dat = np.loadtxt(csvfile, delimiter=';')
    examples = []
    for row in dat:
        ex = []
        for col in row:
            ex.append(col)
        examples.append(ex)
    training_exs = examples[0:80]
    test_exs = examples[80:]
    return training_exs,test_exs
    
def sktNaiveBayes(training_exs,test_exs):
    gnb = GaussianNB()
    training_arr = np.array(training_exs)
    test_arr = np.array(test_exs)
    
    y_pred = gnb.fit(training_arr[:,0:-1], training_arr[:,-1]).predict(test_arr[:,0:-1])
    print(y_pred)
    print(test_arr[:,-1])
    print("scikit-learn : Number of mislabeled points out of a total %d points : %d" % (test_arr.shape[0],(test_arr[:,-1] != y_pred).sum()))
    print("====================================")
    return y_pred
    
    
def myNaiveBayes(training_exs,test_exs): #for data with continuous feature
    training_arr = np.array(training_exs)
    test_arr = np.array(test_exs)
    distribution_dict = {}
    set_cls = {}
    for cls in set(training_arr[:,-1]):
        distribution_dict[cls] = {}
        set_cls[cls] = []
    for ex in training_arr:
        set_cls[ex[-1]].append(ex[0:-1])
   
    for every_key in set(set_cls.keys()):
        set_cls[every_key] = np.array(set_cls[every_key])
    #print(set_cls)
    for cls in distribution_dict.keys():
        for i in range(training_arr[:,0:-1].shape[1]):
                mean = np.mean(set_cls[cls][:,i])
                var = np.cov(set_cls[cls][:,i])
                distribution_dict[cls][i] = [mean,var]
                
    test_result = []
    for i in test_arr[:,0:-1]:
        test_result.append(predictNaiveBayes(i,distribution_dict))
    test_result = np.array(test_result)
    
    print(test_result)
    print(test_arr[:,-1])
    
    print("my implement : Number of mislabeled points out of a total %d points : %d" % (test_arr.shape[0],(test_arr[:,-1] != test_result).sum()))
    print("====================================")
    return test_result

def predictNaiveBayes(ex_arr,distribution_dict):
    prob_dict = {}
    for each_cls in distribution_dict.keys():
        prob_dict[each_cls] = 1
        for i in range(ex_arr.shape[0]):
            prob_dict[each_cls] *= GaussianFunction(ex_arr[i],distribution_dict[each_cls][i][0],distribution_dict[each_cls][i][1])
    
    return max(prob_dict, key = prob_dict.get)
    

def GaussianFunction(x,mean,var):            
    return 1/(np.sqrt(2 * pi * var**2)) * np.exp(-(x-mean)**2/2*(var)**2)
    
if __name__ == '__main__':
    training_exs,test_exs = openfile()
    sktNaiveBayes(training_exs,test_exs)
    myNaiveBayes(training_exs,test_exs)