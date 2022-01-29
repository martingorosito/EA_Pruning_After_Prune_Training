
#-------------------------------------------------------------------------------------------------------
#--Libraries--------------------------------------------------------------------------------------------
import numpy as np
import Network2

from mlxtend.data import loadlocal_mnist
import platform
from keras.utils.np_utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import RMSprop
#-------------------------------------------------------------------------------------------------------

#--Hyperparameters/Global Variables---------------------------------------------------------------------

POP_SIZE = 20
GENERATIONS = 50
N_BEST = 5
STP_CRIT = 51

DATASET = "IRIS"     		#N° of times used:
# DATASET = "SONAR"			#N° of times used:
# DATASET = "WINE"			#N° of times used:
# DATASET = "BREAST"		#N° of times used:
# DATASET = "IONOSPHERE"	#N° of times used:
#DATASET = "DIABETES"		#N° of times used:

#These values are changed in the Choose dataset folder.
EPOCHS = 80
IN_SIZE = 4
HID_SIZE = 5
OUT_SIZE = 3
FILE_NAME = ""

L_RATE = 0.15
MOMENTUM = 0.9
BATCH_SIZE = 1

#SEED = 10 , 175, 247, 300, 465, 574, 663, 750, 891, 928
SEED = 10
#-------------------------------------------------------------------------------------------------------

#--Choose the data--------------------------------------------------------------------------------------

def Choose_dataset(dataset_name):
	if dataset_name == "IRIS":

		#Read dataset
		dataset = pd.read_csv('iris.data')
		dataset.columns = ['SepalLengthCm', 'SepalWidthCM', 'PetalLengthCm', 'PetalWidthCm', 'Species']
		dataset.head()

		#Transform classes into one hot encoded arrays
		one_hot = pd.get_dummies(dataset['Species'])
		dataset = dataset.drop('Species', axis=1)
		dataset = dataset.join(one_hot)

		#Normalizing units
		minmax = MinMaxScaler()
		for x in dataset.columns[dataset.dtypes == 'float64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))
		for x in dataset.columns[dataset.dtypes == 'int64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))

		print("Iris dataset")
		print(dataset)
		xy = dataset.to_numpy()
		x = xy[:,0:4]
		y = xy[:,4:7]

		IN_SIZE = 4
		HID_SIZE = 5
		OUT_SIZE = 3
		EPOCHS = 80
		FILE_NAME = 'NT_IRIS.csv'

		return x, y, IN_SIZE, HID_SIZE, OUT_SIZE, EPOCHS, FILE_NAME

	elif dataset_name == "WINE":

		#Read dataset
		dataset = pd.read_csv('wine.data')
		dataset.columns = ['Class','Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280/OD315_of_diluted_wines', 'Proline']
		dataset.head()

		#Transform classes into one hot encoded arrays
		one_hot = pd.get_dummies(dataset['Class'])
		dataset = dataset.drop('Class', axis=1)
		dataset = dataset.join(one_hot)
		
		#Normalizing units
		minmax = MinMaxScaler()
		for x in dataset.columns[dataset.dtypes == 'float64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))
		for x in dataset.columns[dataset.dtypes == 'int64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))

		print("Wine dataset")
		print(dataset)
		xy = dataset.to_numpy()
		x = xy[:,0:13]
		y = xy[:,13:16]

		IN_SIZE = 13
		HID_SIZE = 5
		OUT_SIZE = 3
		EPOCHS = 15
		FILE_NAME = 'NT_WINE.csv'

		return x, y, IN_SIZE, HID_SIZE, OUT_SIZE, EPOCHS, FILE_NAME

	elif dataset_name == "IONOSPHERE":

		#Read dataset
		dataset = pd.read_csv('ionosphere.data')
		dataset.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, "Class"]
		dataset.head()

		#Transform class into one-hot encoded arrays
		one_hot = pd.get_dummies(dataset['Class'])
		dataset = dataset.drop('Class', axis=1)
		dataset = dataset.join(one_hot)

		# #Normalizing units
		minmax = MinMaxScaler()
		for x in dataset.columns[dataset.dtypes == 'float64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))
		for x in dataset.columns[dataset.dtypes == 'int64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))

		print("Ionosphere dataset")
		print(dataset)
		xy = dataset.to_numpy()
		x = xy[:,0:34]
		y = xy[:,34:36]

		IN_SIZE = 34
		HID_SIZE = 10
		OUT_SIZE = 2
		EPOCHS = 40
		FILE_NAME = 'NT_IONO.csv'

		return x, y, IN_SIZE, HID_SIZE, OUT_SIZE, EPOCHS, FILE_NAME

	elif dataset_name == "SONAR":

		#Read dataset
		dataset = pd.read_csv('sonar.all-data')
		dataset.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, "Class"]
		dataset.head()

		#Transform into one hot encoder arrays
		one_hot = pd.get_dummies(dataset['Class'])
		dataset = dataset.drop('Class', axis=1)
		dataset = dataset.join(one_hot)

		#Normalizing units
		minmax = MinMaxScaler()
		for x in dataset.columns[dataset.dtypes == 'float64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))
		for x in dataset.columns[dataset.dtypes == 'int64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))

		print("Sonar dataset")
		print(dataset)
		xy = dataset.to_numpy()
		x = xy[:,0:60]
		y = xy[:,60:62]

		IN_SIZE = 60
		HID_SIZE = 10
		OUT_SIZE = 2
		EPOCHS = 60
		FILE_NAME = 'NT_SONAR.csv'

		return x,y, IN_SIZE, HID_SIZE, OUT_SIZE, EPOCHS, FILE_NAME

	elif dataset_name == "BREAST":

		#Read dataset
		dataset = pd.read_csv('breast-cancer-wisconsin.data')
		dataset.columns = ['Id', 'Clump_thickness', 'Uniformity_cell_size', 'Uniformity_cell_shape', 'Marginal_adhesion', 'Single_e_cell_size', 'Bare_nuclei', 'Bland_chromatin', 'Normal_nucleoli', 'Mitoses', 'Class']
		dataset.head()

		#Transforming objects into number datatypes
		cleanup = {"Bare_nuclei": {"?":0, "1": 1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8, "9":9, "10":10}}
		dataset = dataset.replace(cleanup)

		#Transforming class into one hot encoded arrays
		one_hot = pd.get_dummies(dataset['Class'])
		dataset = dataset.drop('Class', axis=1)
		dataset = dataset.join(one_hot)

		# #Normalizing units
		minmax = MinMaxScaler()
		for x in dataset.columns[dataset.dtypes == 'float64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))
		for x in dataset.columns[dataset.dtypes == 'int64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))

		print("Breast Cancer dataset")
		print(dataset)
		xy = dataset.to_numpy()
		x = xy[:,1:10]
		y = xy[:,10:12]

		IN_SIZE = 9
		HID_SIZE = 5
		OUT_SIZE = 2
		EPOCHS = 20
		FILE_NAME = 'NT_BREAST.csv'

		return x, y, IN_SIZE, HID_SIZE, OUT_SIZE, EPOCHS, FILE_NAME

	elif dataset_name == "DIABETES":

		#Read dataset
		dataset = pd.read_csv('diabetes.csv')
		dataset.columns = ['Pregnancies', 'Glucose', 'Blood_Pressure', 'Skin_Thickness', 'Insulin', 'BMI', 'Diabetes_Ped_Function', 'Age', 'Outcome']
		dataset.head()

		#Transforming classes into one hot encoded arrays
		one_hot = pd.get_dummies(dataset['Outcome'])
		dataset = dataset.drop('Outcome', axis=1)
		dataset = dataset.join(one_hot)

		#Normalizing units
		minmax = MinMaxScaler()
		for x in dataset.columns[dataset.dtypes == 'float64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))
		for x in dataset.columns[dataset.dtypes == 'int64']:
		    dataset[x] = minmax.fit_transform(np.array(dataset[x]).reshape(-1,1))

		print("PIMA Diabetes dataset")		
		print(dataset)
		xy = dataset.to_numpy()
		x = xy[:,0:8]
		y = xy[:,8:10]

		IN_SIZE = 8
		HID_SIZE = 5
		OUT_SIZE = 2
		EPOCHS = 30
		FILE_NAME = 'NT_DIABETES.csv'

		return x, y, IN_SIZE, HID_SIZE, OUT_SIZE, EPOCHS, FILE_NAME
#-------------------------------------------------------------------------------------------------------

#--Setting up the data----------------------------------------------------------------------------------

np.random.seed(SEED)
x, y, IN_SIZE, HID_SIZE, OUT_SIZE, EPOCHS, FILE_NAME = Choose_dataset(DATASET)

L = (HID_SIZE+OUT_SIZE)*IN_SIZE + HID_SIZE*OUT_SIZE
L1 = IN_SIZE*HID_SIZE
L2 = HID_SIZE*OUT_SIZE
L3 = IN_SIZE*OUT_SIZE

#Shuffling the data to avoid trainig bias
randomize = np.arange(len(x))
np.random.shuffle(randomize)
x = x[randomize]
y = y[randomize]

# Split the data for training and testing for the algorithm
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.50, shuffle = False)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.50, shuffle = False)

x_train	= np.transpose(x_train)
y_train	= np.transpose(y_train)
x_val	= np.transpose(x_val)
y_val	= np.transpose(y_val)
#-------------------------------------------------------------------------------------------------------

#--Individual Representation----------------------------------------------------------------------------

class Individual():
	def __init__(self):
		self.genome = np.zeros(L)
		self.genome1 = np.zeros(L1) 
		self.genome2 = np.zeros(L2)
		self.genome3 = np.zeros(L3)
		self.fitness = 0

	def get_fitness(self, net, x_fit, y_fit):
		self.genome1 = self.genome[0:L1]
		self.genome2 = self.genome[L1:(L2+L1)]
		self.genome3 = self.genome[(L1+L2):]

		mask1 = self.genome1
		mask2 = self.genome2
		maskio = self.genome3

		dnn2 = Network2.DeepNetwork(sizes=[IN_SIZE, HID_SIZE, OUT_SIZE], mask1 = self.genome1, mask2 = self.genome2, maskio = self.genome3)
		dnn2.parameters["W1"] = net.parameters["W1"]
		dnn2.parameters["W2"] = net.parameters["W2"] 
		dnn2.parameters["Wio"] = net.parameters["Wio"]
		dnn2.parameters["b1"] = net.parameters["b1"]
		dnn2.parameters['b2'] = net.parameters['b2'] 
		dnn2.parameters['bio'] = net.parameters['bio']
		dnn2.parameters['VdW1'] = net.parameters['VdW1'] 
		dnn2.parameters['VdW2'] = net.parameters['VdW2']
		dnn2.parameters['VdWio'] = net.parameters['VdWio']
		dnn2.parameters['Vdb1'] = net.parameters['Vdb1']
		dnn2.parameters['Vdb2'] = net.parameters['Vdb2']
		dnn2.parameters['Vdbio'] = net.parameters['Vdbio']
		

		fitness = Network2.compute_accuracy(dnn2, x_fit, y_fit)
		print("Copy's fitness"+str(fitness))

		fitness = Network2.pruned_accuracy(net, x_fit, y_fit, mask1, mask2, maskio)
		print("Benchmark pruned fitness'"+str(fitness))
		return fitness

#-------------------------------------------------------------------------------------------------------

#--Functions--------------------------------------------------------------------------------------------

def Pop_Generation(POP_SIZE, net):
	population = []
	for i in range(POP_SIZE):
		population.append(Individual())
		population[i].genome = np.random.randint(0, 2, L)
		population[i].genome1 = population[i].genome[0:L1]
		population[i].genome2 = population[i].genome[L1:(L2+L1)]
		population[i].genome3 = population[i].genome[(L1+L2):]
		population[i].fitness = population[i].get_fitness(net, x_val, y_val)
	return population

def search_best_individual(candidates, best_individual, prev_fitness, count_fit, print_choice = True):
	for i in range(len(candidates)):
		if best_individual.fitness <= candidates[i].fitness:
			best_individual = candidates[i]
	
	if prev_fitness == best_individual.fitness:
		count_fit = count_fit + 1
	else:
		count_fit = 0

	if print_choice == True:
		print ("\nBest individual so far")
		print("Genome 1: " + str(best_individual.genome1))
		print("Genome 2: " + str(best_individual.genome2))
		print("Genome 3: " + str(best_individual.genome3))
		print("Fitness: " +str(best_individual.fitness))
		print("Times this individual was here: " + str(count_fit))

	return best_individual, count_fit

def tournament(candidates):
	L = len(candidates) - 1
	maximum = np.random.randint(0,L)
	select = np.random.randint(0,L)
	if candidates[maximum].fitness < candidates[select].fitness:
		minimum = maximum
		maximum = select
	else:
		minimum = select
	select = np.random.randint(0,L)
	if candidates[select].fitness > candidates[minimum].fitness:
		minimum = select
	return candidates[maximum], candidates[minimum]

def keep_best(candidates, offspring):
	new_population = candidates + offspring
	new_population.sort(key = lambda x: x.fitness, reverse = True)
	survivors = new_population[:POP_SIZE]
	return survivors

def keep_offspring(candidates, offspring):
	survivors = offspring
	candidates = []
	return survivors

def bitflip_mutation(offspring):
	for i in range(len(offspring.genome)):
		if np.random.uniform(0,1) < 1/len(offspring.genome):
			offspring.genome[i] = 0**offspring.genome[i]
	return offspring

def uniform_crossover(parent_A, parent_B, offspring):
	for i in range(len(parent_A.genome)):
		if np.random.uniform(0,1) > 0.5:
			offspring.genome[i] = parent_A.genome[i]
		else:
			offspring.genome[i] = parent_B.genome[i]
	return offspring

def print_Individual(individual, title):
	print ("\n" + str(title))
	print("Genome: " + str(individual.genome))
	print("Genome 1: " + str(individual.genome1))
	print("Genome 2: " + str(individual.genome2))
	print("Genome 3: " + str(individual.genome3))
	print("Fitness: " +str(individual.fitness))

#-------------------------------------------------------------------------------------------------------

#--Main Algorithm---------------------------------------------------------------------------------------

benchmark_individual = Individual()
benchmark_individual.genome = np.ones(L)
benchmark_individual.genome1 = np.ones(L1)
benchmark_individual.genome2 = np.ones(L2)
benchmark_individual.genome3 = np.ones(L3)

dnn = Network2.DeepNetwork(sizes=[IN_SIZE, HID_SIZE, OUT_SIZE], mask1 = benchmark_individual.genome1, mask2 = benchmark_individual.genome2, maskio = benchmark_individual.genome3)
benchmark_cost, benchmark_accuracy = Network2.train(dnn, x_train, y_train, x_val, y_val, epochs = EPOCHS , l_rate = L_RATE, momentum = MOMENTUM, batch_size = BATCH_SIZE)
benchmark_individual.fitness = benchmark_individual.get_fitness(dnn, x_val, y_val)

print_Individual(benchmark_individual, title = "Benchmark individual")

#Pop generation
history = [] #List to keep all the births so far
candidates = Pop_Generation(POP_SIZE, dnn)
evolution_fitness = [] #Variable to keep track of the best fitness found and plot them in the end.
count_fit = 0 #counter for stopping criterion
total_births = POP_SIZE
history.extend(candidates)

best_individual = candidates[0]
prev_fitness = best_individual.fitness
best_individual, count_fit = search_best_individual(candidates, best_individual, prev_fitness, count_fit)
prev_fitness = best_individual.fitness
evolution_fitness.append(best_individual.fitness*100)


g = 0
while g<GENERATIONS and count_fit<STP_CRIT:
	print ("\nGENERATION "+str(g))
	#Choosing Parents
	parent_A, parent_B = tournament(candidates)

	#Reproduction
	offspring = []
	for p in range(POP_SIZE):
		offspring.append(Individual())
		offspring[p] = uniform_crossover(parent_A, parent_B, offspring[p])
		offspring[p] = bitflip_mutation(offspring[p])
		offspring[p].fitness = offspring[p].get_fitness(dnn, x_val, y_val)

	total_births = total_births + POP_SIZE
	
	#Choosing Survivors
	#candidates = keep_best(candidates,offspring)
	candidates = keep_offspring(candidates,offspring)
	history.extend(candidates)

	#Searching Best individual
	best_individual, count_fit = search_best_individual(candidates, best_individual, prev_fitness, count_fit)
	prev_fitness = best_individual.fitness
	evolution_fitness.append(best_individual.fitness*100)

	g=g+1

#-------------------------------------------------------------------------------------------------------

#--5x2cv------------------------------------------------------------------------------------------------

#Getting the best 20 individuals found
history.sort(key = lambda x: x.fitness, reverse = True)
new_history = []
aux_list = []
historical_best = []

for obj in history:
	aux_val = obj.genome.tobytes()
	if aux_val not in aux_list:
		new_history.append(obj)
		aux_list.append(aux_val)

new_history.sort(key = lambda x: x.fitness, reverse = True)
for element in new_history:
	if element.fitness == best_individual.fitness:
		historical_best.append(element)
	else:
		break

with open(FILE_NAME, 'a', newline='') as csvfile:

	#Heading
	header = ['Results from 5x2 cross validation combined F test with seed: '+str(SEED)]
	writer = csv.DictWriter(csvfile, fieldnames = header)
	writer.writeheader()
	fieldnames = ['Genome', 'Genome1','Genome2', 'Genome3', 'EA Fitness', 'Benchmark mean accuracy', 'Best Individual mean accuracy', 'F value']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator = '\n')
	writer.writeheader()

for j in range(len(historical_best)):

	best_individual = historical_best[j]
	bench_ind_accuracies = []
	best_ind_accuracies = []
	acc_differences = []

	for fold in range(5):
		kfold2 = KFold(2,True)
		for train, test in kfold2.split(x):
			x_train_cv = x[train]
			y_train_cv = y[train]
			x_test_cv = x[test]
			y_test_cv = y[test]

			x_train_cv = np.transpose(x_train_cv)
			y_train_cv = np.transpose(y_train_cv)
			x_test_cv = np.transpose(x_test_cv)
			y_test_cv = np.transpose(y_test_cv)

			dnn_benchmark = Network2.DeepNetwork(sizes=[IN_SIZE, HID_SIZE, OUT_SIZE], mask1 = benchmark_individual.genome1, mask2 = benchmark_individual.genome2, maskio = benchmark_individual.genome3)
			bench_cost, bench_accuracy = Network2.train(dnn_benchmark, x_train_cv,y_train_cv,x_test_cv,y_test_cv, epochs = EPOCHS , l_rate = L_RATE, momentum = MOMENTUM, batch_size = BATCH_SIZE)

			best_accuracy = best_individual.get_fitness(dnn_benchmark, x_test_cv, y_test_cv)

			bench_ind_accuracies.append(bench_accuracy)
			best_ind_accuracies.append(best_accuracy)
			acc_differences.append(bench_accuracy-best_accuracy)

	variances = []
	numerator = 0
	denominator = 0
	f = 0

	for i in range(5):
		mean = (acc_differences[2*i] + acc_differences[2*i+1])/2
		variance = (acc_differences[2*i]-mean)**2 + (acc_differences[2*i+1]-mean)**2 

		denominator = denominator + 2*variance
		numerator = numerator + acc_differences[2*i]**2 + acc_differences[2*i+1]**2
		
	f = numerator/denominator

	with open(FILE_NAME, 'a', newline='') as csvfile:

	    #Field names
		fieldnames = ['Genome', 'Genome1','Genome2', 'Genome3', 'EA Fitness', 'Benchmark mean accuracy', 'Best Individual mean accuracy', 'F value']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator = '\n')
		#writer.writeheader()

	    #Individuals
		writer.writerow({'Genome':best_individual.genome, 'Genome1': best_individual.genome1, 'Genome2': best_individual.genome2, 'Genome3': best_individual.genome3, 
			'EA Fitness': best_individual.fitness, 'Benchmark mean accuracy': np.mean(bench_ind_accuracies)*100, 'Best Individual mean accuracy': np.mean(best_ind_accuracies)*100, 'F value': f})
	    
	print("\n---------------------Results---------------------")
	print_Individual(benchmark_individual,"Benchmark Individual")
	print_Individual(best_individual,"Best Individual Found")
	print("F obtained " + str(f))
	print("Mean Accuracy of Fully connected network during 5x2cv: {0:.2f}%".format(np.mean(bench_ind_accuracies)*100))
	print("Mean Accuracy of Pruned network during 5x2cv: {0:.2f}%".format(np.mean(best_ind_accuracies)*100))
	print(bench_ind_accuracies)
	print(best_ind_accuracies)

space_covered = total_births/(2**L)
with open(FILE_NAME, 'a', newline='') as csvfile:

	#Heading
	header = ['Total births: '+str(total_births)]
	writer = csv.DictWriter(csvfile, fieldnames = header)
	writer.writeheader()
	header = ['Percentage of space covered: '+str(space_covered*100)+'%']
	writer = csv.DictWriter(csvfile, fieldnames = header)
	writer.writeheader()
	fieldnames = ['Genome', 'Genome1','Genome2', 'Genome3', 'EA Fitness', 'Benchmark mean accuracy', 'Best Individual mean accuracy', 'F value']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator = '\n')
	writer.writerow({'Genome': '' })

print("Total births: "+ str(total_births))
print("Percentage of space covered: {0}%".format(space_covered*100))
