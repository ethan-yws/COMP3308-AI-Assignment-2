# COMP3308 AI
# Assignment 2 Classification
# @author: Ethan Shi
# @SID: 450095111

from kNN import kNN
from NaiveBayes import NB

import sys
import csv
from collections import deque

# Read and analysis the command line inputs
training_file = sys.argv[1]

# function to prepare the data
def convert_to_instance(fname):
    # read each row's content as an array
    data = []
    with open(fname, "r") as f:
        for row in f:
            row = row.split(",")
            data.append(row)
    # convert numeric data to float type
    # check if it's training data
    instance_len = len(data[0])
    if data[0][-1] == "yes\n" or data[0][-1] == "no\n":
        convert_to_float(data, instance_len-1)
    else:
        convert_to_float(data, instance_len)
    return data

# function to convert numeric data from str type to float type
def convert_to_float(data, length):
    for row in data:
        for i in range(length):
            row[i] = float(row[i])

# archive the prepared data
training_data = convert_to_instance(training_file)

# function to seprate the data by class
def separate_class(training_data):
    class_yes = []
    class_no = []

    for ins in training_data:
        if ins[-1] == "yes\n":
            class_yes.append(ins)
        else:
            class_no.append(ins)
    return class_yes, class_no

# archive the seprated data
class_yes, class_no = separate_class(training_data)
# convert them to deque type for 10 fold later
class_yes_dq = deque(class_yes)
class_no_dq = deque(class_no)

# separate to 10 folds
# since we have 268 class yes and 500 class no
# fold 1 to fold 8 will consist of 27 yes and 50 no
# for the last two folds - fold 9 and 10, consisting of 26 yes and 50 no
folds = []
# the first 8 fold
for i in range(8):
    fold = []
    for j in range(27):
        fold.append(class_yes_dq.popleft())
    for k in range(50):
        fold.append(class_no_dq.popleft())
    folds.append(fold)
# the 9th and 10th fold
for i in range(2):
    fold = []
    for j in range(26):
        fold.append(class_yes_dq.popleft())
    for k in range(50):
        fold.append(class_no_dq.popleft())
    folds.append(fold)


# function to wirte the 10 folds into csv file named "pima-folds.csv"
def write_to_file(folds):
    # remove "\n"
    for fold in folds:
        for ins in fold:
            ins[-1] = ins[-1].replace("\n", "")

    with open("pima-folds.csv", "w") as of:
        for i in range(9):
            of.write("fold%d\n" % (i+1))
            writer = csv.writer(of)
            writer.writerows(folds[i])
            of.write("\n")
        # write the 10th fold into file
        of.write("fold10\n")
        writer = csv.writer(of)
        writer.writerows(folds[-1])

#************************************************************
# generate the "pima-folds.csv" file
# Uncomment the following function call to generate the file
#************************************************************
# write_to_file(folds)


# Evaluate the accuracy for 1NN
accuracy_list = []
k = 0
while k < 10:
    tenFolds = list(folds) # make a copy

    testing_set_raw = tenFolds[k]
    testing_set = [ins[:-1] for ins in testing_set_raw]
    #print(testing_set)

    training_set_raw = folds[:k] + folds[k+1:]
    training_set = []
    for item in training_set_raw:
        training_set += item
    
    """***********************************************************************"""
    """ uncomment the following codes to test accuracy for different algorithm """

    #results = kNN(training_set, testing_set, 1) # 1NN
    results = kNN(training_set, testing_set, 5) # 5NN
    #results = NB(training_set, testing_set)     # Naive Bayes
    """***********************************************************************"""


    #print(len(results))
    match = 0
    for i in range(len(results)):
        if results[i][-1] in testing_set_raw[i][-1]:
            match += 1

    accuracy = match / len(results)
    accuracy_list.append(accuracy)
    print("Round %d:" % (k+1), "%.2f%%" % (accuracy*100))

    k += 1
    

# Calculate the average of the accuracy for 10 folds
avg = sum(accuracy_list) / len(accuracy_list)
print("\nAverage Accuracy: %.2f%%" % (avg * 100))



