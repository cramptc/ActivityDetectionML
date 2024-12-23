import os
import csv
import numpy as np
import tensorflow as tf
import pandas as pd
import argparse
import matplotlib.pyplot as plt

# Create an ArgumentParser

sensor = 'Respeck'

#HyperParameters
time_dimension = 75 #refers to the amount of data points to include in a piece of data
time_stride = 75 #refers to the amount of data points to skip between pieces of data

DERIVATIVE_SMOOTHING = 2 

stft_len = time_dimension
stft_stride = time_stride
stft_dimensions = 1 + ((time_dimension - stft_len) // stft_stride)


labelstrresp = ['other']
labelstract = []
labelstr = [labelstrresp,labelstract]

def getLabel(s,r):
	out = []
	s = s.split("_")
	if r == 0:
		if s[3][:-4] in ['laughing', 'singing', 'talking', 'eating']:
			out.append('other')
		else:
			out.append(s[3][:-4])
	elif r == 1:
		if s[2] in ['sitting', 'standing']:
			out.append('stationary')
		else:
			out.append(s[2])
	else:
		print("\n\n\n\n\n\n Wrong Sensor Number \n\n\n\n\n\n")
		out.append(s[2])
	out = "_".join(out)
	if not (out in labelstr[r]):
		labelstr[r].append(out)
	return labelstr[r].index(out)

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

#Read in the initial 2D matrices for data set	
direc = os.getcwd()
datapath = direc + "/Data/" + sensor
c = 0
for snum in os.listdir(datapath):
	#Get Data
	# Initialize an empty list to store the data from the first column
	print(snum)
	BigData = []
	LabelsResp = []
	LabelsAct = []
	TempData = []
	Fourier = []
	Derivatives = []
	studepath = datapath+"/"+snum
	files =  [f for f in os.listdir(studepath)]
	for file in files:
		perfile = 0
		row_count = 0
		csvl = len(pd.read_csv(studepath+"/"+file,skiprows=1,header=None))
		while row_count+time_dimension < csvl:
			# Read the CSV file and extract data from the right columns
			df = pd.read_csv(studepath+"/"+file, skiprows=range(0, row_count), nrows=time_dimension)
			df = df.iloc[0:time_dimension,2:8]
			
			# Convert the list to a NumPy matrix
			matrix = df.to_numpy()

			# Print the resulting NumPy matrix
			TempData.append(matrix)
			LabelsResp.append(getLabel(file,0))
			LabelsAct.append(getLabel(file,1))
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
	output_file = snum+'Data.tfrecord'  # Output TFRecord file name
	numpy_matrices_to_tfrecord(BigData, Fourier, Derivatives, LabelsResp, LabelsAct, output_file)

datashape = derivativeshape = fouriershape= [time_dimension,6,1]

shapefile = "Shapes.tfrecord"
shapes_to_tfrecord(datashape, fouriershape, derivativeshape, shapefile)

for i in labelstr:
	print(i)

