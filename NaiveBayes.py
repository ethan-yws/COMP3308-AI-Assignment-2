# COMP3308 AI
# Assignment 2 Classification
# @author: Ethan Shi
# @SID: 450095111

import math
import numpy as np

def NB(training_data, testing_data):
    training_data_size = len(training_data)
    # the first is to separate data by class
    class_yes = []
    class_no = []

    for ins in training_data:
        if ins[-1] == "yes\n":
            class_yes.append(ins)
        else:
            class_no.append(ins)

    size_yes = len(class_yes)
    size_no = len(class_no)
    # calculate the Probability of class "yes" and class "no"
    p_yes = size_yes / training_data_size
    p_no = size_no / training_data_size

    # list of mean and standard deviation fro each attribute in class yes
    mean_std_yes = get_mean_std(class_yes)
    mean_std_no = get_mean_std(class_no)

    # make a copy of testing_data to append results for output later
    testing_dt = testing_data.copy()
    for ins in testing_dt:
        p_yes_ins = p_yes
        p_no_ins = p_no
        for i in range(len(ins)):
            p_yes_ins *= get_probability_density(mean_std_yes[i][0], mean_std_yes[i][1], ins[i])
            p_no_ins *= get_probability_density(mean_std_no[i][0], mean_std_no[i][1], ins[i])

        if p_yes_ins >= p_no_ins:
            ins.append("yes")
        else:
            ins.append("no")

    return testing_dt


# the probability density function
def get_probability_density(mean, sd, x):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(sd,2))))
    return (1 / (math.sqrt(2*math.pi) * sd)) * exponent
    

""" The following function implemented based on online resuorces
 Author: Jason Brownlee
 Date: 12 Sep, 2014
 Availability: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/"""
# function to calculate the mean and standard deviation for each attribute for each class
# will forms sublists by the columns
def get_mean_std(data):
    # remove the last column i.e. remove "yes" and "no", stay only numeric data
    dt = [x[:-1] for x in data]

    mean_std = [(np.mean(attribute), np.std(attribute)) for attribute in zip(*dt)]
    return mean_std