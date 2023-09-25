"""
"""
import torch
import numpy as np
import os
from tqdm import tqdm

from .src.ppo_trainer import PPOTRainer

def ppo_train(workload, scheduler, train_step):
    #env = scheduler.env
    trainer = PPOTRainer(scheduler.model, scheduler.env)
    batch_size = 64
    
    best_reward = -1e4
    reward_history=[]; avgresponsetime_history=[]; energytotalinterval_history=[]
    n_steps = 0
    for i in tqdm(range(train_step)):
        newcontainerinfos = workload.generateNewContainers(scheduler.env.interval) 
        deployed, destroyed = scheduler.env.addContainers(newcontainerinfos) 
        decisions, actions, log_probs, mainInfo, encoder_inputs, steps, decoder_inputs = \
            scheduler.run_transformer()
        filter_decisions = scheduler.filter_placement(decisions)
        trainer.save_mid_step (encoder_inputs, decisions, filter_decisions, actions, 
                               log_probs, steps, decoder_inputs)
        
        migrations, rewards = scheduler.env.simulationStep(filter_decisions)
        step_reward = sum(rewards.values())
        reward_history.append(step_reward)
        avgresponsetime = np.average([c.totalExecTime + c.totalMigrationTime for c in destroyed]) if len(destroyed) > 0 else 0
        if avgresponsetime != 0: avgresponsetime_history.append(avgresponsetime)
        energytotalinterval = np.sum([host.getPower()*scheduler.env.intervaltime for host in scheduler.env.hostlist])
        energytotalinterval_history.append(energytotalinterval)
        trainer.save_final_step(rewards)
        workload.updateDeployedContainers(scheduler.env.getCreationIDs(migrations, deployed)) 
        n_steps += len(rewards)
        if n_steps >= batch_size:
            n_steps = trainer.train(batch_size)
            
        print('interval', scheduler.env.interval, 'step_reward %.3f' % step_reward, 
              'avgresponsetime %.2f' % avgresponsetime, 'energytotalinterval %.2f' % energytotalinterval, 
              'reward_history_50avg %.3f'% np.mean(reward_history[-50:]),
              'responsetime_history_50avg %.3f'% np.mean(avgresponsetime_history[-50:]), 
              'energytotal_history_50avg %.3f'% np.mean(energytotalinterval_history[-50:]))
    


def save_model(save_path, model, optimizer):
	file_path = save_path + "/" + model.name + "_" + "TRL" + ".ckpt"
	torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, file_path)
    
def load_model(save_path, model):
	file_path = save_path + "/" + model.name + "_" + "TRL" + ".ckpt"
	if os.path.exists(file_path):
		print("Loading pre-trained model: ")
		checkpoint = torch.load(file_path)
		model.load_state_dict(checkpoint['model_state_dict'])
	else:
		print("Creating new model: "+model.name)
	return model