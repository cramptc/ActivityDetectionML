import tensorflow as tf
import os
import pydot
import csv
import numpy as np
from tensorflow.keras import layers, models, utils,activations
from tensorflow.keras.layers import Lambda
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import classification_report
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#Stop the Epoch printing for every single iteration
tf.get_logger().setLevel('ERROR')
sensor = 'Respeck'

epochnum =20
dpoints = 35000
numResps = 4
numActs = 11



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

def generate_unique_ids(list1, list2):
	unique_ids = []
	for i in range(len(list1)):
		unique_id = str(int(list1[i])) + '_' + str(int(list2[i]))
		unique_ids.append(unique_id)
	return unique_ids

def isRecord(s):
	return (s[0] == 's' and s[-8:] == "tfrecord" )

metaset = tf.data.TFRecordDataset('DataResults.tfrecord')
parsed_meta = metaset.map(parse_tfmetarecord_fn)
for dm,ds,fm,fs,Δm,Δs in parsed_meta:
	dm = np.array(dm)
	ds = np.array(ds)
	fm = np.array(fm)
	fs = np.array(fs)
	Δm = np.array(Δm)
	Δs = np.array(Δs)

def normdata(x,dm,ds):
	return (x-dm)/ds


def normfourier(x,fm,fs):
	return (x-fm)/fs


def normderivatives(x,Δm,Δs):	
	if Δs == 0:
		return float(0)
	a = (x-Δm)/Δs
	return a

def modelGen():
	#Model Construction
	dmodel = models.Sequential()
	dmodel.add(layers.Conv2D(32, (5,5),activation='relu', input_shape=(data_s),padding='same'))
	dmodel.add(layers.MaxPooling2D((2, 1)))
	dmodel.add(layers.Conv2D(32, (3,3),activation='relu'))
	dmodel.add(layers.MaxPooling2D((2, 1)))
	dmodel.add(layers.Flatten())
	dmodel.add(layers.Dense(32,activation='relu'))

	fmodel = models.Sequential()
	fmodel.add(layers.MaxPooling2D((5, 1),input_shape=(fourier_s)))
	fmodel.add(layers.Conv2D(32, (5,5),activation='relu',padding='same',input_shape=(fourier_s)))
	fmodel.add(layers.MaxPooling2D((2, 1)))
	fmodel.add(layers.Conv2D(32, (3,3),activation='relu'))
	fmodel.add(layers.MaxPooling2D((2, 1)))
	fmodel.add(layers.Flatten())
	fmodel.add(layers.Dense(32,activation='relu'))

	Δmodel = models.Sequential()
	Δmodel.add(layers.Conv2D(32, (5,5),activation='relu', input_shape=(derivative_s),padding='same'))
	Δmodel.add(layers.MaxPooling2D((2, 1)))
	Δmodel.add(layers.Conv2D(32, (3,3),activation='relu'))
	Δmodel.add(layers.MaxPooling2D((2, 1)))
	#Δmodel.add(layers.Conv2D(64, (2,2),activation='relu'))
	#Δmodel.add(layers.MaxPooling2D((2, 1)))
	Δmodel.add(layers.Flatten())
	Δmodel.add(layers.Dense(64,activation='relu'))


	merged = layers.concatenate([dmodel.output,fmodel.output, Δmodel.output])
	Middle = layers.Dense(32)(merged)
	penresp = layers.Dense(32) (Middle)
	penact = layers.Dense(32) (Middle)
	outputresp = layers.Dense(numResps,name='Respiratory')(penresp)
	outputact = layers.Dense(numActs,name='Activity') (penact)
	#outputfunct = Lambda(generate_unique_ids)([np.argmax(outputresp),np.argmax(outputact)])



	outmodel = models.Model(inputs=[ dmodel.input,fmodel.input,Δmodel.input],outputs=[outputresp, outputact])


	outmodel.compile(optimizer='adam',
				  
			  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			  metrics=['accuracy'])
	return outmodel
#outmodel.summary()

recordList = []
recordList = list(filter( isRecord,os.listdir(os.getcwd())))
numStuds = len(recordList)

BDataSet = [0]*numStuds
BLabelRespSet = [0]*numStuds
BLabelActSet = [0]*numStuds
BFourierSet = [0]*numStuds
BDerivativeSet = [0]*numStuds


metaset = tf.data.TFRecordDataset('DataResults.tfrecord')
parsed_meta = metaset.map(parse_tfmetarecord_fn)
for dm,ds,fm,fs,Δm,Δs in parsed_meta:
	dm = np.array(dm)
	ds = np.array(ds)
	fm = np.array(fm)
	fs = np.array(fs)
	Δm = np.array(Δm)
	Δs = np.array(Δs)
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


def runnormal(n,split=4):
	test_data = np.array(BDataSet[0])
	test_fourier = np.array(BFourierSet[0])
	test_derivatives = np.array(BDerivativeSet[0])
	test_labelsresp = np.array(BLabelRespSet[0])
	test_labelsact = np.array(BLabelActSet[0])
	for i in range(1,split):
		test_data = np.concatenate((test_data,BDataSet[i]))
		test_fourier = np.concatenate((test_fourier,BFourierSet[i]))
		test_derivatives = np.concatenate((test_derivatives,BDerivativeSet[i]))
		test_labelsresp = np.concatenate((test_labelsresp,BLabelRespSet[i]))
		test_labelsact = np.concatenate((test_labelsact,BLabelActSet[i]))
	train_data = np.array([])
	train_fourier = np.array([])
	train_derivatives = np.array([])
	train_labelsresp = np.array([])
	train_labelsact = np.array([])
	outmodel = modelGen()
	for j in range(split+1,BDataSet.__len__()):
		if train_data.size == 0:
			train_data = np.array(BDataSet[j])
			train_fourier = np.array(BFourierSet[j])
			train_derivatives = np.array(BDerivativeSet[j])
			train_labelsresp = np.array(BLabelRespSet[j])
			train_labelsact = np.array(BLabelActSet[j])
		else:
			train_data = np.concatenate((train_data,BDataSet[j]))
			train_fourier = np.concatenate((train_fourier,BFourierSet[j]))
			train_derivatives = np.concatenate((train_derivatives,BDerivativeSet[j]))
			train_labelsresp = np.concatenate((train_labelsresp,BLabelRespSet[j]))
			train_labelsact = np.concatenate((train_labelsact,BLabelActSet[j]))	
	history = outmodel.fit([train_data, train_fourier, train_derivatives],[train_labelsresp,train_labelsact],
						  epochs=n, validation_data=([test_data, test_fourier,test_derivatives],
								[test_labelsresp, test_labelsact]))
	print(history.history)
	resptrain_acc = history.history['Respiratory_accuracy']
	acttrain_acc = history.history['Activity_accuracy']
	respval_acc = history.history['val_Respiratory_accuracy']
	actval_acc = history.history['val_Activity_accuracy']


	# Get the number of epochs
	epochs = range(n)

	# Plot the graph
	plt.plot(epochs, resptrain_acc, 'b', label='Respiratory Training Accuracy')
	plt.plot(epochs, respval_acc, 'r', label='Respiratory Validation Accuracy')
	plt.plot(epochs, acttrain_acc, 'g', label='Activity Training Accuracy')
	plt.plot(epochs, actval_acc, 'y', label='Activity Validation Accuracy')
	plt.title('Training and Validation Accuracies')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()
	plt.savefig(sensor + 'Accuracies.png')


def loo(e):
	#compile outmodel using different metrics for each output
	score = 0

	for i in range(BDataSet.__len__()):
		outmodel = modelGen()
		print("This is the " + str(i) + "th iteration of LOO")
		test_data = np.array(BDataSet[i])
		test_fourier = np.array(BFourierSet[i])
		test_derivatives = np.array(BDerivativeSet[i])
		test_labelsresp = np.array(BLabelRespSet[i])
		test_labelsact = np.array(BLabelActSet[i])
		train_data = np.array([])
		train_fourier = np.array([])
		train_derivatives = np.array([])
		train_labelsresp = np.array([])
		train_labelsact = np.array([])
		for j in range(BDataSet.__len__()):
			if j != i:
				if train_data.size == 0:
					train_data = np.array(BDataSet[j])
					train_fourier = np.array(BFourierSet[j])
					train_derivatives = np.array(BDerivativeSet[j])
					train_labelsresp = np.array(BLabelRespSet[j])
					train_labelsact = np.array(BLabelActSet[j])
				else:
					train_data = np.concatenate((train_data,BDataSet[j]))
					train_fourier = np.concatenate((train_fourier,BFourierSet[j]))
					train_derivatives = np.concatenate((train_derivatives,BDerivativeSet[j]))
					train_labelsresp = np.concatenate((train_labelsresp,BLabelRespSet[j]))
					train_labelsact = np.concatenate((train_labelsact,BLabelActSet[j]))
		early_stopping = EarlyStopping(monitor='val_Respiratory_accuracy', patience=5, restore_best_weights=True)

		outmodel.fit([train_data, train_fourier, train_derivatives], [train_labelsresp, train_labelsact],
						epochs=e, validation_data=([test_data, test_fourier, test_derivatives],
														[test_labelsresp, test_labelsact]),
						callbacks=[early_stopping])

		predresp,predact = outmodel.predict([test_data,test_fourier,test_derivatives])
		rep = classification_report(generate_unique_ids(test_labelsresp,test_labelsact),
							  generate_unique_ids(list(map(np.argmax,predresp)),list(map(np.argmax,predact))),
							  output_dict=True)
		acc = rep["accuracy"]
		print(acc)
		score += acc
		tf.keras.backend.clear_session()


	print("Average Accuracy: ")
	print(float(float(score)/float(len(recordList))))

	


###### DO WHHICHEVER TEST YOU WANT HERE
#loo(20)
runnormal(30)



###### Option to save the model
#outmodel.save(sensor + "Model.h5")
