# COMP3308 AI
# Assignment 2 Classification
# @author: Ethan Shi
# @SID: 450095111

import math
import operator

def kNN(training_data, testing_data, k):
    testing_dt = testing_data.copy()
    for test_ins in testing_dt:
        neighbours = getNeighbours(training_data, test_ins, k)
        yes_count = 0
        no_count = 0
        for nb in neighbours:
            if nb[0][-1] == "yes\n":
                yes_count += 1
            else:
                no_count += 1
        if yes_count >= no_count:
            test_ins.append("yes")
        else:
            test_ins.append("no")
    return testing_dt


""" The following functions implemented based on online resuorces
 Author: Jason Brownlee
 Date: 12 Sep, 2014
 Availability: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/ """

# function to get the k nearest neighbours
def getNeighbours(training_data, testing_instance, k):
    # training_data appends its euclidean distance
    tmp = []
    for i in range(len(training_data)):
        euclidean_dis = cal_euclidean_dis(training_data[i], testing_instance)
        tmp.append((training_data[i], euclidean_dis))
    
    # sort the tmp array by its euclidean distance for later use
    tmp.sort(key=operator.itemgetter(1))
    # get k neighbours
    neighbours = []
    for j in range(k):
        neighbours.append(tmp[j])
    return neighbours

# function to calculate the euclidean distance
def cal_euclidean_dis(training_data_instance, testing_instance):
    euclidean_dis = 0
    for i in range(len(testing_instance)):
        euclidean_dis += pow((testing_instance[i] - training_data_instance[i]), 2)
    return math.sqrt(euclidean_dis)
