import sys
import tensorflow as tf
from sklearn.metrics import classification_report
import os
import numpy as np
import pandas as pd

#####################################################################################
# This file is used to evaluate a given model on new data
# The process requires use of the DataResults.tfrecord file for normalisation
# in the ml_code directory so it MUST be run with the same file structure as it has
# been submitted in or the pathToNorm variable must be changed to the correct path
#
# USE THE 3model.h5 file for the model which can evaluate any of the 4 respiratory 
# symptom classes and any of the 11 activity classes separately before merging back 
# together to give a final prediction. As such there is the potential for 44 different
# classes to be predicted.
#####################################################################################

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
condition_enum = {
    'breathingNormal': 1,
    'coughing': 2,
    'hyperventilating': 3,
    'other': 0
}
pathToNorm = "./ml_code/DataResults.tfrecord"
shapefile = "./ml_code/Shapes.tfrecord"
################################
#Setup
################################
def run(path):
    #HyperParameters
    time_dimension = 75 #refers to the amount of data points to include in a piece of data
    time_stride = 75 #refers to the amount of data points to skip between pieces of data

    DERIVATIVE_SMOOTHING = 2 


    def getLabel(s):
        activity, condition = s.split('_')
        activity_enum = getActivityEnum(activity)
        condition_enum = getConditionEnum(condition)
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
            dfm = pd.read_csv(file, skiprows=range(0, row_count), nrows=time_dimension)
            df = dfm.iloc[0:time_dimension,1:7]
            details = dfm.iloc[0:time_dimension, 8]
            type = details.iloc[0]
            if details.iloc[time_dimension-1] == type:
                # Convert the list to a NumPy matrix
                matrix = df.to_numpy()
                TempData.append(matrix)
                labels = getLabel(type)
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

def getActivityEnum(activity):

    return activity_enum.get(activity, -1)

def getConditionEnum(condition):
    return condition_enum.get(condition, -1)

def getActivityKey(value):
    for key, val in activity_enum.items():
        if val == value:
            return key
    return None

def getConditionKey(value):
    for key, val in condition_enum.items():
        if val == value:
            return key
    return None

def parse_hyper(example):
	feature_description = {
			'datashape': tf.io.FixedLenFeature([3], tf.int64),
			'fouriershape': tf.io.FixedLenFeature([3], tf.int64),
			'derivativeshape': tf.io.FixedLenFeature([3], tf.int64)
		}
	example = tf.io.parse_single_example(example, feature_description)
	data = example['datashape']
	fourier = example['fouriershape']
	derivatives = example['derivativeshape']
	return data,fourier,derivatives

#Get DataShape
datashape = tf.data.TFRecordDataset(shapefile)
for d,s,Δ in datashape.map(parse_hyper):
	data_s = tuple(np.array(d))
	fourier_s =  tuple(np.array(s))
	derivative_s = tuple(np.array(Δ))


def parse_tfrecord_fn(example):
	feature_description = {
		'data': tf.io.FixedLenFeature(data_s, tf.float32),  # Define the shape
		'fourier':tf.io.FixedLenFeature(fourier_s, tf.float32),
		'derivatives':tf.io.FixedLenFeature(derivative_s, tf.float32),
		'labelresp': tf.io.FixedLenFeature([1], tf.int64),  # Adjust label shape as needed
		'labelact': tf.io.FixedLenFeature([1], tf.int64),  # Adjust label shape as needed
	}
	#Get Data
	example = tf.io.parse_single_example(example, feature_description)
	data = example['data']
	labelresp = example['labelresp']
	labelact = example['labelact']
	fourier = example['fourier']
	derivatives = example['derivatives']
	return data, labelresp,labelact, fourier, derivatives

def parse_tfmetarecord_fn(example):
	feature_description = {
		'datamean': tf.io.FixedLenFeature(data_s, tf.float32),
		'datastd': tf.io.FixedLenFeature(data_s, tf.float32),  # Define the shape
		'fouriermean':tf.io.FixedLenFeature(fourier_s, tf.float32),
		'fourierstd':tf.io.FixedLenFeature(fourier_s, tf.float32),
		'derivativesmean':tf.io.FixedLenFeature(derivative_s, tf.float32),
		'derivativesstd':tf.io.FixedLenFeature(derivative_s, tf.float32),
		}
	#Get Data
	example = tf.io.parse_single_example(example, feature_description)
	datam = example['datamean']
	datas = example['datastd']
	fourierm = example['fouriermean']
	fouriers = example['fourierstd']
	derm = example['derivativesmean']
	ders = example['derivativesstd']
	return datam, datas, fourierm, fouriers, derm, ders

def isRecord(s):
	return (s[0] == 's' and s[-8:] == "tfrecord" )

def normdata(x,dm,ds):
	return (x-dm)/ds


def normfourier(x,fm,fs):
	return (x-fm)/fs


def normderivatives(x,Δm,Δs):	
	if Δs == 0:
		return float(0)
	a = (x-Δm)/Δs
	return a

def remove_second_half(arr):
	return arr[:, :3]

def pulldata():
	metaset = tf.data.TFRecordDataset(pathToNorm)
	parsed_meta = metaset.map(parse_tfmetarecord_fn)
	for dm,ds,fm,fs,Δm,Δs in parsed_meta:
		dm = np.array(dm)
		ds = np.array(ds)
		fm = np.array(fm)
		fs = np.array(fs)
		Δm = np.array(Δm)
		Δs = np.array(Δs)
		dm = remove_second_half(dm)
		ds = remove_second_half(ds)
		fm = remove_second_half(fm)
		fs = remove_second_half(fs)
		Δm = remove_second_half(Δm)
		Δs = remove_second_half(Δs)
	j = 'ValidationData.tfrecord'
	dataset = tf.data.TFRecordDataset(j)
	parsed_dataset = dataset.map(parse_tfrecord_fn)
	vdata = np.vectorize(normdata)
	vfour= np.vectorize(normfourier)
	vder = np.vectorize(normderivatives)
	DataSet = []
	LabelRespSet = []
	LabelActSet = []
	FourierSet = []
	DerivativeSet = []
	for data,labelresp,labelact,fourier,derivatives in parsed_dataset:
		data = remove_second_half(data)
		fourier = remove_second_half(fourier)
		derivatives = remove_second_half(derivatives)
		DataSet.append(vdata(np.array(data),dm,ds))
		LabelRespSet.append(labelresp)
		LabelActSet.append(labelact)
		FourierSet.append(vfour(np.array(fourier),fm,fs))
		DerivativeSet.append(vder(np.array(derivatives),Δm,Δs))
	return np.array(DataSet),np.array(LabelRespSet),np.array(LabelActSet), np.array(FourierSet),np.array(DerivativeSet)





#######################
#Main Script
#######################

if len(sys.argv) != 3:
    print("Usage: python model_evaluation.py <modelpath> <datapath>")
    sys.exit(1)

modelpath = sys.argv[1]
datapath = sys.argv[2]


gen = run(datapath)

# Load the model
loaded_model = tf.keras.models.load_model(modelpath)
print("Loaded model from disk")


def generate_unique_ids(list1, list2):
    unique_ids = []
    for i in range(len(list1)):
        unique_id = str(getConditionKey(int(list1[i]))) + '_' + str(getActivityKey(int(list2[i])))
        unique_ids.append(unique_id)
    return unique_ids


# Load the data
BDataSet, BLabelRespSet, BLabelActSet, BFourierSet, BDerivativeSet = pulldata()

correct = generate_unique_ids(BLabelActSet, BLabelRespSet)
# Make predictions
# Combine the input data into a single list
input_data = [BDataSet, BFourierSet, BDerivativeSet]

# Make predictions
predictions = loaded_model.predict(input_data)
resppredicted_class = tf.argmax(predictions[1], axis=-1).numpy()
actpredicted_class = tf.argmax(predictions[0], axis=-1).numpy()
predictset = generate_unique_ids( actpredicted_class,resppredicted_class)



# Evaluate the model
report = classification_report( correct, predictset)
print(report)

