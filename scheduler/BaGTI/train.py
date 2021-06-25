from src.constants import *
from src.utils import *
from src.models import *
from src.ga import *
from src.opt import *
from src.optW import *

from sys import argv, maxsize
from time import time

import warnings
warnings.filterwarnings("ignore")

def custom_loss(y_pred, y_true, model_name):
	if 'stochastic' in model_name:
		return KL_loss(y_pred, Coeff_Energy*y_true[0] + Coeff_Latency*y_true[1])
	return torch.sum((y_pred - y_true) ** 2)

def backprop(dataset, model, optimizer):
	total = 0
	for feat in dataset:
		feature = feat[0]
		if not 'W' in model.name:
			feature = torch.tensor(feature,dtype=torch.float)
			y_pred = model(feature)
		else:
			apps, graph = feat[1], feat[2]
			y_pred = model(feature, apps, graph)
		y_true = feat[-1]
		# print(y_pred, y_true)
		optimizer.zero_grad()
		loss = custom_loss(y_pred, y_true, model.name)
		loss.backward()
		optimizer.step()
		total += loss
	return total/len(dataset)

def accuracy(dataset, model):
	total = 0
	for feat in dataset:
		feature = feat[0]
		if not 'W' in model.name:
			feature = torch.tensor(feature,dtype=torch.float)
			y_pred = model(feature)
		else:
			apps, graph = feat[1], feat[2]
			y_pred = model(feature, apps, graph)
		y_true = feat[-1]
		loss = custom_loss(y_pred, y_true, model.name)
		total += loss
	return total/len(dataset)

def save_model(model, optimizer, epoch, accuracy_list):
	file_path = MODEL_SAVE_PATH + "/" + model.name + "_" + str(epoch) + ".ckpt"
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(filename, model, data_type):
	optimizer = torch.optim.Adam(model.parameters() , lr=0.0001, weight_decay=1e-5) if 'stochastic' not in data_type else torch.optim.AdamW(model.parameters() , lr=0.0001)
	file_path1 = MODEL_SAVE_PATH + "/" + filename + "_Trained.ckpt"
	file_path2 = 'scheduler/BaGTI/' + file_path1
	file_path = file_path1 if os.path.exists(file_path1) else file_path2
	if os.path.exists(file_path):
		print(color.GREEN+"Loading pre-trained model: "+filename+color.ENDC)
		checkpoint = torch.load(file_path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		epoch = -1; accuracy_list = []
		print(color.GREEN+"Creating new model: "+model.name+color.ENDC)
	return model, optimizer, epoch, accuracy_list

if __name__ == '__main__':
	data_type = argv[1] # can be 'energy', 'energy_latency', 'energy_latency2', energy_latencyW', 
	# 'stochastic_energy_latency', 'stochastic_energy_latency2' + '_' + str(HOSTS)
	exec_type = argv[2] # can be 'train', ga', 'opt'

	model = eval(data_type+"()")
	model, optimizer, start_epoch, accuracy_list = load_model(data_type, model, data_type)
	dtl = data_type.split('_')
	dataset, dataset_size, _ = eval("load_"+'_'.join(dtl[:-1])+"_data("+dtl[-1]+")")

	if exec_type == "train":
		split = int(0.8 * dataset_size)

		for epoch in range(start_epoch+1, start_epoch+EPOCHS+1):
			print('EPOCH', epoch)
			random.shuffle(dataset)
			trainset = dataset[:split]
			validation = dataset[split:]
			loss = backprop(trainset, model, optimizer)
			trainAcc, testAcc = float(loss.data), float(accuracy(validation, model).data)
			accuracy_list.append((testAcc, trainAcc))
			print("Loss on train, test =", trainAcc, testAcc)
			if epoch % 10 == 0:
				save_model(model, optimizer, epoch, accuracy_list)
		print ("The minimum loss on test set is ", str(min(accuracy_list)), " at epoch ", accuracy_list.index(min(accuracy_list)))

		plot_accuracies(accuracy_list, data_type)
	else:
		print(model.find); start = time()
		for param in model.parameters(): param.requires_grad = False
		init = torch.tensor(random.choice(dataset)[0], dtype=torch.float, requires_grad=True)

		if exec_type == "ga":
			result, iteration, fitness = ga(dataset, model, [], data_type)
		elif exec_type == "opt":
			result, iteration, fitness = opt(init, model, [], data_type)
		print("Time", time()-start)
		print("Iteration: {}\nResult: {}\nFitness: {}".format(iteration, result, fitness)) 
