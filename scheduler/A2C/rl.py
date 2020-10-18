from .constants import *
from .models import *
from .utils import *

import os
from sys import argv
from time import time
from torch.distributions import Categorical

def backprop(model, schedule_t, value_t, schedule_next, optimizer):
	value_pred, action_pred = model(schedule_t)
	value_pred_next, action_next = model(schedule_next)
	optimizer.zero_grad()
	value_loss = torch.sum((value_pred - value_t) ** 2)
	policy_loss = -1 * (value_pred_next - value_pred) # -1 for minimizing value, () has advantage
	for probs in action_pred:
		m = Categorical(probs)
		a = m.sample()
		policy_loss += m.log_prob(a)
	loss = value_loss + policy_loss
	loss.backward()
	optimizer.step()
	return value_loss.item(), policy_loss.item(), action_next

def save_model(model, optimizer, epoch, accuracy_list):
	file_path = MODEL_SAVE_PATH + "/" + model.name + "_" + str(epoch) + ".ckpt"
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, file_path)

def load_model(filename, model, data_type):
	optimizer = torch.optim.AdamW(model.parameters() , lr=0.0001, weight_decay=1e-5)
	file_path = MODEL_SAVE_PATH + "/" + model.name + "_Trained.ckpt"
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
