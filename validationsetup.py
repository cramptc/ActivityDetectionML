import os
import csv
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import enum as Enum
import pandas as pd

# Create an ArgumentParser

def run(path):
    #HyperParameters
    time_dimension = 75 #refers to the amount of data points to include in a piece of data
    time_stride = 75 #refers to the amount of data points to skip between pieces of data

    DERIVATIVE_SMOOTHING = 2 

    labelstrresp = ['other']
    labelstract = []
    labelstr = [labelstrresp,labelstract]

    def getActivityEnum(activity):
        activity_enum = {
            'ascending': 0,
            'descending': 1,
            'lyingBack': 2,
            'lyingLeft': 3,
            'lyingRight': 4,
            'lyingStomach': 5,
            'miscMovement': 6,
            'normalWalking': 7,
            'running': 8,
            'shuffleWalking': 9,
            'sitStand': 10
        }
        return activity_enum.get(activity, -1)

    def getConditionEnum(condition):
        condition_enum = {
            'breathingNormal': 1,
            'coughing': 2,
            'hyperventilating': 3,
            'other': 0
        }
        return condition_enum.get(condition, -1)

    def getLabel(s):
        activity, condition = s.split('_')
        activity_enum = getActivityEnum(activity)
        condition_enum = getConditionEnum(condition)
        labelstr[1].append(activity)
        labelstr[0].append(condition)
        return [activity_enum, condition_enum]

    def extend_start_end(mat, start, end):
        first = mat[:1,:]
        last = mat[-1:,:]
        return np.concatenate([*[first] * start, mat, *[last] * end])

    def smooth(mat):
        total = None
        for i in range(-DERIVATIVE_SMOOTHING, 1+DERIVATIVE_SMOOTHING):
            shift = extend_start_end(mat, i + DERIVATIVE_SMOOTHING, DERIVATIVE_SMOOTHING - i)
            if total is None:
                total = shift
            else:
                total += shift
        
        return total[DERIVATIVE_SMOOTHING:-DERIVATIVE_SMOOTHING,:] / (DERIVATIVE_SMOOTHING * 2 + 1)


    def create_derivative(mat):
        mat = smooth(mat)
        shift = extend_start_end(mat, 1, 0)[:-1,:]
        return mat - shift

    #Store them all as TFRecords
    def numpy_matrices_to_tfrecord(data, fourier, derivatives, labelsresp, labelsact,output_file):
        with tf.io.TFRecordWriter(output_file) as writer:
            for (ind,matrix) in enumerate(data):
                feature = {
                    'data': tf.train.Feature( float_list=tf.train.FloatList(value=matrix.flatten())),
                    'fourier': tf.train.Feature( float_list=tf.train.FloatList(value=fourier[ind].flatten())),
                    'derivatives': tf.train.Feature( float_list=tf.train.FloatList(value=derivatives[ind].flatten())),
                    'labelresp': tf.train.Feature(int64_list=tf.train.Int64List(value=[labelsresp[ind]])),
                    'labelact': tf.train.Feature(int64_list=tf.train.Int64List(value=[labelsact[ind]])),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                serialized = example.SerializeToString()
                writer.write(serialized)

            #Store them all as TFRecords
    def shapes_to_tfrecord(datashape, fouriershape, derivativeshape, output_file):
        with tf.io.TFRecordWriter(output_file) as writer:
                feature = {
                    'datashape': tf.train.Feature( int64_list=tf.train.Int64List(value=np.ravel(datashape))),
                    'fouriershape': tf.train.Feature( int64_list=tf.train.Int64List(value=np.ravel(fouriershape))),
                    'derivativeshape': tf.train.Feature( int64_list=tf.train.Int64List(value=np.ravel(derivativeshape))),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                serialized = example.SerializeToString()
                writer.write(serialized)

    #Get Data
    # Initialize an empty list to store the data from the first column
    BigData = []
    LabelsResp = []
    LabelsAct = []
    TempData = []
    Fourier = []
    Derivatives = []
    files =  [path]
    for file in files:
        perfile = 0
        row_count = 0
        csvl = len(pd.read_csv(file,skiprows=1,header=None))
        while row_count+time_dimension < csvl:
            # Read the CSV file and extract data from the right columns
            df = pd.read_csv(file, skiprows=range(0, row_count), nrows=time_dimension)
            df = df.iloc[0:time_dimension,1:7]
            details = df.iloc[0:time_dimension,9]
            type = details.iloc[0,0]
            if details.iloc[time_dimension,9] == type:
                # Convert the list to a NumPy matrix
                matrix = df.to_numpy()
                # Print the resulting NumPy matrix
                TempData.append(matrix)
                labels = getLabel(file)
                LabelsResp.append(labels[0])
                LabelsAct.append(labels[1])
                perfile+=1
        #carry on to next data piece 
            row_count+=time_stride
    for mat in TempData:
        ft =[]
        for i in range(6):
            fft = np.fft.fft(mat[:,i])
            z = np.array(fft).ravel()
            amp = np.abs(z)
            ft.append(np.copy(amp))
        ft = np.stack(ft,axis=1)
        Fourier.append(np.copy(ft))
        BigData.append(np.copy(mat))
        Derivatives.append(create_derivative(mat))
    output_file = 'ValidationData.tfrecord'  # Output TFRecord file name
    numpy_matrices_to_tfrecord(BigData, Fourier, Derivatives, LabelsResp, LabelsAct, output_file)

    datashape = derivativeshape = fouriershape= [time_dimension,6,1]

    shapefile = "Shapes.tfrecord"
    shapes_to_tfrecord(datashape, fouriershape, derivativeshape, shapefile)

    for i in labelstr:
        print(i)
    print(max(LabelsResp))
    print(max(LabelsAct))

