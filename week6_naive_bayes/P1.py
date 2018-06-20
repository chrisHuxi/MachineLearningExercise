import random

def simulate(time):
    count_dict = {1:0,2:0,3:0,4:0,5:0,6:0} 
    for i in range(time):
        count_list = []
        for i in range(5):
            count_list.append(random.randint(1,6))
        count_dict[max(count_list)] += 1
    prob_dict = {}
    for everykey in count_dict.keys():
        prob_dict[everykey] = count_dict[everykey]/(time + 0.0)
    
    return prob_dict
    
if __name__ == '__main__':
    print(simulate(10000000))