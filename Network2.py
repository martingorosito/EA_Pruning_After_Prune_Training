	
import numpy as np
import time



from mlxtend.data import loadlocal_mnist
import platform
from keras.utils.np_utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

class DeepNetwork():
	def __init__(self, sizes, mask1, mask2, maskio):

		self.sizes = sizes 
		self.parameters = self.initialization()
		self.parameters_2 = {}
		self.accuracies = []
		self.costs = []
		self.mask1 = mask1.reshape(self.sizes[1],self.sizes[0])
		self.mask2 = mask2.reshape(self.sizes[2],self.sizes[1])
		self.maskio = maskio.reshape(self.sizes[2],self.sizes[0])

	def initialization(self):	

		parameters = {}

		parameters['W1'] = np.random.randn(self.sizes[1],self.sizes[0]) * np.sqrt(1. / self.sizes[1])
		parameters['W2'] = np.random.randn(self.sizes[2],self.sizes[1]) * np.sqrt(1. / self.sizes[2])
		parameters['Wio'] = np.random.randn(self.sizes[2], self.sizes[0]) * np.sqrt(1. / self.sizes [2])

		parameters['b1'] = np.ones([self.sizes[1],1])
		parameters['b2'] = np.ones([self.sizes[2],1])
		parameters['bio'] = np.ones([self.sizes[2],1])
		#Momentum Parameters
		parameters['VdW1'] = np.zeros([self.sizes[1],self.sizes[0]]) 
		parameters['VdW2'] = np.zeros([self.sizes[2],self.sizes[1]])
		parameters['VdWio'] = np.zeros([self.sizes[2],self.sizes[0]])

		parameters['Vdb1'] = np.zeros([self.sizes[1],1])
		parameters['Vdb2'] = np.zeros([self.sizes[2],1])
		parameters['Vdbio'] = np.zeros([self.sizes[2],1])
		return parameters



def forward_prop(DeepNetwork, x_train):

	DeepNetwork.parameters['W1'] = DeepNetwork.parameters['W1']*DeepNetwork.mask1	
	DeepNetwork.parameters['W2'] = DeepNetwork.parameters['W2']*DeepNetwork.mask2
	DeepNetwork.parameters['Wio'] = DeepNetwork.parameters['Wio']*DeepNetwork.maskio		
	
	DeepNetwork.parameters_2['A0'] = x_train				
	
	DeepNetwork.parameters_2['Z1'] = np.dot(DeepNetwork.parameters['W1'], DeepNetwork.parameters_2['A0']) + DeepNetwork.parameters['b1']
	DeepNetwork.parameters_2['A1'] = np.tanh(DeepNetwork.parameters_2['Z1'])

	DeepNetwork.parameters_2['Z2'] = np.dot(DeepNetwork.parameters['W2'],DeepNetwork.parameters_2['A1']) + DeepNetwork.parameters['b2'] 
	DeepNetwork.parameters_2['Zio'] = np.dot(DeepNetwork.parameters['Wio'],DeepNetwork.parameters_2['A0']) + DeepNetwork.parameters['bio'] 
	DeepNetwork.parameters_2['A2'] = softmax(DeepNetwork.parameters_2['Z2'] + DeepNetwork.parameters_2['Zio'])

	return DeepNetwork.parameters_2['A2']

def forward_prop_2(DeepNetwork, x_train):
	
	#Pruning
	W1 = DeepNetwork.parameters['W1']*DeepNetwork.mask1
	W2 = DeepNetwork.parameters['W2']*DeepNetwork.mask2

	Wio = DeepNetwork.parameters['Wio']*DeepNetwork.maskio
	b1 = DeepNetwork.parameters['b1']
	b2 = DeepNetwork.parameters['b2']	
	bio = DeepNetwork.parameters['bio']
	
	A0 = x_train				
	
	Z1 = np.dot(W1, A0) + b1 
	A1 = np.tanh(Z1)

	Z2= np.dot(W2,A1) + b2
	Zio = np.dot(Wio,A0) + bio
	A2 = softmax(Z2 + Zio)

	return A2

def compute_cost(DeepNetwork, AL, y_train):		
	#cost = -np.mean([y_train*np.log(AL)])
	predictions = []
	output = AL
	pred = np.argmax(output, axis = 0)
	predictions.append(pred == np.argmax(y_train, axis = 0))
	cost =1- np.mean(predictions)
	return cost

def back_prop(DeepNetwork, y_train, A2):
	dW_updates = {}
	m = y_train.shape[1]

	dW_updates['dZ2'] = (A2 - y_train)*softmax_backwards(DeepNetwork.parameters_2['Z2'])
	dW_updates['dW2'] = (1/m)*np.dot(dW_updates['dZ2'],DeepNetwork.parameters_2['A1'].T)
	dW_updates['db2'] = (1/m)*np.sum(dW_updates['dZ2'], axis = 1, keepdims = True)

	dW_updates['dZio'] = (A2 - y_train)*softmax_backwards(DeepNetwork.parameters_2['Zio'])
	dW_updates['dWio'] = (1/m)*np.dot(dW_updates['dZio'],DeepNetwork.parameters_2['A0'].T)
	dW_updates['dbio'] = (1/m)*np.sum(dW_updates['dZio'], axis = 1, keepdims = True)


	t = np.tanh(DeepNetwork.parameters_2['Z1'])
	dW_updates['dZ1'] = np.dot(DeepNetwork.parameters['W2'].T,dW_updates['dZ2'])*(1 - t**2)
	dW_updates['dW1'] = (1/m)*np.dot(dW_updates['dZ1'],DeepNetwork.parameters_2['A0'].T)
	dW_updates['db1'] = (1/m)*np.sum(dW_updates['dZ1'], axis = 1, keepdims = True)

	return dW_updates

def update_parameters(DeepNetwork, dW_updates, l_rate, momentum):
	DeepNetwork.parameters['VdW2'] = momentum*DeepNetwork.parameters['VdW2'] + (1-momentum)*dW_updates['dW2']	
	DeepNetwork.parameters['Vdb2'] = momentum*DeepNetwork.parameters['Vdb2'] + (1-momentum)*dW_updates['db2']	
	DeepNetwork.parameters['W2'] = DeepNetwork.parameters['W2'] - l_rate*DeepNetwork.parameters['VdW2']
	DeepNetwork.parameters['b2'] = DeepNetwork.parameters['b2'] - l_rate*DeepNetwork.parameters['Vdb2']

	DeepNetwork.parameters['VdW1'] = momentum*DeepNetwork.parameters['VdW1'] + (1-momentum)*dW_updates['dW1']
	DeepNetwork.parameters['Vdb1'] = momentum*DeepNetwork.parameters['Vdb1'] + (1-momentum)*dW_updates['db1']	
	DeepNetwork.parameters['W1'] = DeepNetwork.parameters['W1'] - l_rate*DeepNetwork.parameters['VdW1']
	DeepNetwork.parameters['b1'] = DeepNetwork.parameters['b1'] - l_rate*DeepNetwork.parameters['Vdb1']

	DeepNetwork.parameters['VdWio'] = momentum*DeepNetwork.parameters['VdWio'] + (1-momentum)*dW_updates['dWio']
	DeepNetwork.parameters['Vdbio'] = momentum*DeepNetwork.parameters['Vdbio'] + (1-momentum)*dW_updates['dbio']	
	DeepNetwork.parameters['Wio'] = DeepNetwork.parameters['Wio'] - l_rate*DeepNetwork.parameters['VdWio']
	DeepNetwork.parameters['bio'] = DeepNetwork.parameters['bio'] - l_rate*DeepNetwork.parameters['Vdbio']

	return

def compute_accuracy(DeepNetwork, x_val_acc, y_val_acc):
	predictions = []
	output = forward_prop(DeepNetwork, x_val_acc)
	pred = np.argmax(output, axis = 0)
	predictions.append(pred == np.argmax(y_val_acc, axis = 0))
	accuracy = np.mean(predictions)

	return accuracy

def pruned_accuracy(DeepNetwork, x_val, y_val, mask1, mask2, maskio):

	DeepNetwork.mask1 = mask1.reshape(DeepNetwork.sizes[1],DeepNetwork.sizes[0])
	DeepNetwork.mask2 = mask2.reshape(DeepNetwork.sizes[2],DeepNetwork.sizes[1])
	DeepNetwork.maskio = maskio.reshape(DeepNetwork.sizes[2],DeepNetwork.sizes[0])

	predictions = []
	output = forward_prop_2(DeepNetwork, x_val)
	pred = np.argmax(output, axis = 0)
	predictions.append(pred == np.argmax(y_val, axis = 0))
	accuracy = np.mean(predictions)

	return accuracy

def predict(DeepNetwork, x):		
	output = forward_prop(DeepNetwork, x)
	pred = np.argmax(output)
	return pred
	
def train(DeepNetwork, x_train, y_train, x_val, y_val, epochs,l_rate, momentum, batch_size):
	start_time = time.time()

	n_examples = x_train.shape[1]
	iteration = 0	
	n_batches = round(n_examples/batch_size + 0.5)
	
	accuracy_epoch = 0
	cost_epoch = 0

	for iteration in range(epochs):
		costs_for_batch = []
		accuracies_for_batch = []

		for b in range(n_batches-1):
			x_batch = x_train[:,b*batch_size:(b+1)*batch_size]
			y_batch = y_train[:,b*batch_size:(b+1)*batch_size]

			A2 = forward_prop(DeepNetwork, x_batch)
			cost = compute_cost(DeepNetwork, A2, y_batch)
			costs_for_batch.append(cost)
			dW_updates = back_prop(DeepNetwork, y_batch,A2)
			
			accuracy = compute_accuracy(DeepNetwork, x_val,y_val)
			accuracies_for_batch.append(accuracy)
			update_parameters(DeepNetwork, dW_updates, l_rate, momentum)

		accuracy_epoch = np.mean(accuracies_for_batch)
		cost_epoch = np.mean(costs_for_batch)
		DeepNetwork.costs.append(cost_epoch)
		DeepNetwork.accuracies.append(accuracy_epoch)
		
		
		#print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%, Cost: {3:.2f}'.format(iteration+1,time.time()-start_time,accuracy*100,cost_epoch))

	return cost, accuracy


#----------------------------Activation Formulas
def softmax(z):
	exps = np.exp(z)
	A = exps/np.sum(exps,axis=0)	
	return A

def softmax_backwards(z):
	exps = np.exp(z)
	dZ = exps/np.sum(exps,axis=0)*(1-exps/np.sum(exps,axis=0))
	return dZ
