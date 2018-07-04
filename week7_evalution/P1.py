import numpy as np
from math import log
from math import pi
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import random

from scipy import interp
from sklearn.metrics import roc_curve, auc

def openfile(csvname):
    csvpath = './data/'             
    csvfile = csvpath + csvname
    print('Reading "' + csvfile + '":')
    dat = np.loadtxt(csvfile, delimiter=';')
    examples = []
    for row in dat:
        ex = []
        for col in row:
            ex.append(col)
        examples.append(ex)
    return examples


def kFoldCrossValidation(data_set,k,classifier):
    all_data_set = data_set[:]
    random.shuffle(all_data_set)
    acc_list = []
    for i in range(k-1):
        train_set = all_data_set[0:int(len(data_set)/k)*i]
        test_set = all_data_set[int(len(data_set)/k)*i:int(len(data_set)/k)*(i+1)]
        train_set.extend(all_data_set[int(len(data_set)/k)*(i+1):])
        acc = sum(classifier(train_set,test_set) == np.array(test_set)[:,-1])/len(test_set)
        acc_list.append(acc)
    return np.mean(np.array(acc_list))
    
def drawROCCurve(data_set,k,classifier):
    all_data_set = data_set[:]
    random.shuffle(all_data_set)
    acc_list = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    for i in range(k):
        train_set = all_data_set[0:int(len(data_set)/k)*i]
        test_set = all_data_set[int(len(data_set)/k)*i:int(len(data_set)/k)*(i+1)]
        train_set.extend(all_data_set[int(len(data_set)/k)*(i+1):])
        y_pred_prob = classifier(train_set,test_set)
        fpr, tpr, thresholds = roc_curve(np.array(test_set)[:,-1], y_pred_prob[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,\
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
#def drawROCCurve(test_result,test_label):
    
    
def sktNaiveBayes(training_exs,test_exs):
    gnb = GaussianNB()
    training_arr = np.array(training_exs)
    test_arr = np.array(test_exs)
    
    y_pred_prob = gnb.fit(training_arr[:,0:-1], training_arr[:,-1]).predict_proba(test_arr[:,0:-1])
    print(y_pred_prob)
    #print(test_arr[:,-1])
    #print("scikit-learn : Number of mislabeled points out of a total %d points : %d" % (test_arr.shape[0],(test_arr[:,-1] != y_pred).sum()))
    #print("====================================")
    return y_pred_prob
    
def myNaiveBayes_prob(training_exs,test_exs): #for data with continuous feature
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
                var = np.var(set_cls[cls][:,i])
                distribution_dict[cls][i] = [mean,var]
    
    #print(distribution_dict)
    
    test_result = []
    for i in test_arr[:,0:-1]:
        test_result.append(predictNaiveBayes_prob(i,distribution_dict,set_cls))
    test_result = np.array(test_result)
    
    print(test_result)

    
    print("====================================")
    return test_result

def predictNaiveBayes_prob(ex_arr,distribution_dict,set_cls):
    prob_dict = {}
    for each_cls in distribution_dict.keys():
        prob_dict[each_cls] = 1
        for i in range(ex_arr.shape[0]):
            prob_dict[each_cls] *= GaussianFunction(ex_arr[i],distribution_dict[each_cls][i][0],distribution_dict[each_cls][i][1])
        #print(len(set_cls[each_cls]))
        prob_dict[each_cls] *= len(set_cls[each_cls])*0.01
    #print(prob_dict)
    
    l = []    
    for key in prob_dict.keys():
        l.append(prob_dict[key])
    l_normalized = []
    for i in l:
        l_normalized.append(i/sum(l))
    return l[::-1]


    
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
                var = np.var(set_cls[cls][:,i])
                distribution_dict[cls][i] = [mean,var]
    
    #print(distribution_dict)
    
    test_result = []
    for i in test_arr[:,0:-1]:
        test_result.append(predictNaiveBayes(i,distribution_dict,set_cls))
    test_result = np.array(test_result)
    
    print(test_result)
    print(test_arr[:,-1])
    
    print("my implement : Number of mislabeled points out of a total %d points : %d" % (test_arr.shape[0],(test_arr[:,-1] != test_result).sum()))
    print("====================================")
    return test_result

def predictNaiveBayes(ex_arr,distribution_dict,set_cls):
    prob_dict = {}
    for each_cls in distribution_dict.keys():
        prob_dict[each_cls] = 1
        for i in range(ex_arr.shape[0]):
            prob_dict[each_cls] *= GaussianFunction(ex_arr[i],distribution_dict[each_cls][i][0],distribution_dict[each_cls][i][1])
        #print(len(set_cls[each_cls]))
        prob_dict[each_cls] *= len(set_cls[each_cls])*0.01
    #print(prob_dict)
    return max(prob_dict, key = prob_dict.get)
    


def GaussianFunction(x,mean,var):            
    return 1/np.sqrt(2 * pi * var) * np.exp(-(x-mean)**2/(2*var))
    #return 1/sqrt(variance * 2 * pi) * exp(-(x - mean)**2/(2*var))
    
if __name__ == '__main__':
    training_exs = openfile('data3.csv')
    test_exs = openfile('data3_test.csv')
    
    #sktNaiveBayes(training_exs[0:100],test_exs)
    myNaiveBayes(training_exs[0:100],test_exs)
    #=============================================================#
    x = []
    y = []
    for i in range(1,101,10):
        x.append(i)
        y.append((sum(myNaiveBayes(training_exs[0:i],test_exs) == np.array(test_exs)[:,-1]))/len(test_exs))
    print(x,y)
    plt.plot(x,y)
    plt.xlabel('train set amount')
    plt.ylabel("accuracy")
    plt.title("learn curve")
    plt.show()
    
    
    data_set = training_exs[:]
    data_set.extend(test_exs[:])
    k_str = input("how many folds wanna set? ：");

    k = int(k_str)
    acc_cv = kFoldCrossValidation(data_set,k,myNaiveBayes)
    print(acc_cv)
    #==================k fold cross validation===================#
    #============================================================#
    drawROCCurve(data_set,k,myNaiveBayes_prob)
    #drawROCCurve(data_set,k,sktNaiveBayes)
    
    #============================================================#