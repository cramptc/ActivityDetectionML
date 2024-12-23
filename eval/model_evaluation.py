import sys
import validationsetup
import tensorflow as tf
from sklearn.metrics import classification_report
import os
import numpy as np
  





shapefile = "Shapes.tfrecord"
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


recordList = []
recordList = list(filter( isRecord,os.listdir(os.getcwd())))
numStuds = len(recordList)

BDataSet = [0]*numStuds
BLabelRespSet = [0]*numStuds
BLabelActSet = [0]*numStuds
BFourierSet = [0]*numStuds
BDerivativeSet = [0]*numStuds


def remove_second_half(arr):
	return arr[:, :3]

def pulldata(half):
	metaset = tf.data.TFRecordDataset('DataResults.tfrecord')
	parsed_meta = metaset.map(parse_tfmetarecord_fn)
	for dm,ds,fm,fs,Δm,Δs in parsed_meta:
		dm = np.array(dm)
		ds = np.array(ds)
		fm = np.array(fm)
		fs = np.array(fs)
		Δm = np.array(Δm)
		Δs = np.array(Δs)
	if half:
		dm = remove_second_half(dm)
		ds = remove_second_half(ds)
		fm = remove_second_half(fm)
		fs = remove_second_half(fs)
		Δm = remove_second_half(Δm)
		Δs = remove_second_half(Δs)
	c = 0
	for j in recordList:
		print("Loaded: " + j)
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
			if half:
				data = remove_second_half(data)
				fourier = remove_second_half(fourier)
				derivatives = remove_second_half(derivatives)
			DataSet.append(vdata(np.array(data),dm,ds))
			LabelRespSet.append(labelresp)
			LabelActSet.append(labelact)
			FourierSet.append(vfour(np.array(fourier),fm,fs))
			DerivativeSet.append(vder(np.array(derivatives),Δm,Δs))
		BDataSet[c] = np.array(DataSet)
		BLabelRespSet[c]=np.array(LabelRespSet)
		BLabelActSet[c]=np.array(LabelActSet)
		BFourierSet[c]= np.array(FourierSet)
		BDerivativeSet[c]= np.array(DerivativeSet)
		c+=1










if len(sys.argv) != 3:
    print("Usage: python model_evaluation.py <modelpath> <datapath>")
    sys.exit(1)

modelpath = sys.argv[1]
datapath = sys.argv[2]

sensors = 1
if os.path.basename(modelpath) == "4outmodel.h5":
    sensors = 2

gen = validationsetup.run(datapath,sensors)

# Load the model
loaded_model = tf.keras.models.load_model(modelpath)
print("Loaded model from disk")



# Load the data
data = validationsetup.load_data(datapath)

# Make predictions
predictions = loaded_model.predict(data)

# Evaluate the model
report = classification_report(data.labels, predictions)
print(report)

# Rest of your code goes here
