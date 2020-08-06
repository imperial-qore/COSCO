from src.constants import *
from src.utils import *
from src.models import *
from src.ga import *
from src.opt import *

from sys import argv
from time import time

def backprop(dataset, model, optimizer):
	total = 0
	for feat in dataset:
		feature = feat[0]
		feature = torch.tensor(feature,dtype=torch.float)
		y_pred = model(feature)
		y_true = feat[1]
		# print(y_pred, y_true)
		optimizer.zero_grad()
		loss = torch.sum((y_pred - y_true) ** 2)
		loss.backward()
		optimizer.step()
		total += loss
	return total/len(dataset)

def accuracy(dataset, model):
	total = 0
	for feat in dataset:
		feature = feat[0]
		feature = torch.tensor(feature,dtype=torch.float)
		y_pred = model(feature)
		y_true = feat[1]
		loss = torch.sum((y_pred - y_true) ** 2)
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
	optimizer = torch.optim.Adam(model.parameters() , lr=0.0001, weight_decay=1e-5)
	file_path = MODEL_SAVE_PATH + "/" + filename + "_Trained.ckpt"
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
	data_type = argv[1]
	exec_type = argv[2]

	model = eval(data_type+"()")
	model, optimizer, start_epoch, accuracy_list = load_model(data_type, model, data_type)
	dataset, dataset_size = eval("load_"+data_type+"_data()")

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

		plot_accuracies(accuracy_list)
	else:
		print(model.find); start = time()
		for param in model.parameters(): param.requires_grad = False
		bounds = np.array([[0,9.5], [0,90], [1,40], [-16,16]])

		if exec_type == "ga":
			ga(dataset, model, bounds, data_type)
		elif exec_type == "opt":
			opt(dataset, model, bounds, data_type)
		elif exec_type == "data":
			least_dataset(dataset, data_type)
		print("Time", time()-start)
