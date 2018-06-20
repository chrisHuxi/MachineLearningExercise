# Machine Learning Exercise

exercise of ML course in TUD, [exercise description and sildes click here](http://cvl.inf.tu-dresden.de/courses/machine-learning-1/).


## week6 : Naive Bayes
#### including :
 * [P1.py](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week6_naive_bayes/P1.py)
 * [P3.py](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week6_naive_bayes/P3.py)
 
### The result as following : 

* P1 : simulation of tossing dice :

  * theory calculate : { 1 : 1/46656 , 2 : 63/46656 , 3 : 665/46656 , 4 : 3367/46656 , 5 : 11529/46656 , 6 : 31031/46656 }


![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week6_naive_bayes/P1_result.PNG)


* P3 : Naive Bayes classifier in Data3 ( my implementation & scikit-learn implementation ) :


![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week6_naive_bayes/P3_result.PNG)


## week5 : Random Forest
#### including :
 * [random_forest.py](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week5_pruning_random_forest/random_forest.py)
 * [random_forest_SKT.py](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week5_pruning_random_forest/random_forest_SKT.py)
 * `TODO : DecisionTree_pruning.py` : I think it is impossible to implement a pruning method in decision tree consisted of list. If we want to prune a decision tree, this tree must be able to be modified during the iteration, which can't be realized with a list. A better data struct is "class". Or anyone has a idea with list? if you do, pls tell me :-)
### The result as following : 

* random forest : my implementation (without continuous feature):


![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week5_pruning_random_forest/result/my_implement.PNG)


* random forest : scikit learn package's implementation(with continuous feature):


![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week5_pruning_random_forest/result/skt_implement.PNG)


## week4 : Decision Tree with continuous value
#### edit :
  * after reading the document of scikit_learn package, I implement the DT_skt.py again with a more simple style, [code click here](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week4_decision_tree_cont_variables/DT_skt_plus.py)
  * but the result seems the same as before, unfortunately :-(
  
  
  ![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week4_decision_tree_cont_variables/result/P2_plus.PNG)
  


#### including :
 * [DecisionTree_cont_var.py](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week4_decision_tree_cont_variables/DecisionTree_cont_var.py)
 * [DT_skt.py](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week4_decision_tree_cont_variables/DT_skt.py)
 * [DT_gain_ratio.py](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week4_decision_tree_cont_variables/DT_gain_ratio.py)
 * [data](https://github.com/chrisHuxi/MachineLearningExercise/tree/master/week4_decision_tree_cont_variables/data)
### The result as following : 

* P1 DecisionTree_cont_var : my implementation:


![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week4_decision_tree_cont_variables/result/P1.PNG)


* P2 DecisionTree : scikit-learn implementation:


![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week4_decision_tree_cont_variables/result/P2.PNG)


* P3 DecisionTree : with gain ratio, on the "Data3.csv":


![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week4_decision_tree_cont_variables/result/P3.PNG)




## week3 : Decision Tree
### edit:
 * I took a mistake : in my original code, every time I produce a node I set a feature as "false", so that the final decision tree is small and with lower accuracy. 
 * so we should change code in `dtree_learning(examples, attr_avail, default, cls_index)` in line 133. As follow:
  ```python
     #attr_avail[best_attr_index] = False # original code
     new_attr_avail = attr_avail[:best_attr_index]+[False]+attr_avail[best_attr_index+1:]   #edited code
  ```
 * result of new code : 
 
    ![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week3_decision_tree/result_dt_plus.PNG)
   
 * [edited code click here](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week3_decision_tree/DecisionTree_plus.py)
 * reference : [code from Professor Guthier](http://cvl.inf.tu-dresden.de/HTML/teaching/courses/ml1/ss18/Ex/3/tree.py)

### including :
 * [DecisionTree.py](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week3_decision_tree/DecisionTree.py)
 * [data set](https://github.com/chrisHuxi/MachineLearningExercise/tree/master/week3_decision_tree/data)

### The result as following : 

![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week3_decision_tree/result_dt.PNG)



## week2 : visualize
### including :
 * [2D_gaussian_distribution.py](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week2_visualize/2D_gaussian_distribution.py)
 * [visualize.py](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week2_visualize/visualize.py)
 * [data : DatAccel.txt](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week2_visualize/DatAccel.txt) , [data : DatGyr.txt](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week2_visualize/DatGyr.txt)
 
### The result as following : 

* P1 gaussian distribution:


  ![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week2_visualize/P1_result.png)


* [covariance matrix, which calculated from generated data : ](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week2_visualize/P1_result.txt)

* P2 visualize : 


  ![](https://github.com/chrisHuxi/MachineLearningExercise/blob/master/week2_visualize/P2_result_1.png)
