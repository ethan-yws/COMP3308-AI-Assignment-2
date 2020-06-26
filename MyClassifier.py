# COMP3308 AI
# Assignment 2 Classification
# @author: Ethan Shi
# @SID: 450095111

from kNN import kNN
from NaiveBayes import NB

import sys
import csv

# Read and analysis the command line inputs
training_file = sys.argv[1]
testing_file = sys.argv[2]
algo_select = sys.argv[3]

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
testing_data = convert_to_instance(testing_file)


# Driver code
if algo_select == "NB":
    results = NB(training_data, testing_data)
    for x in results:
        print(x[-1])
else:
    results = kNN(training_data, testing_data, int(algo_select[0]))
    for x in results:
        print(x[-1])

    
