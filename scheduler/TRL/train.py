"""
"""
import torch
import os
from tqdm import tqdm


def ppo_train(env, workload, scheduler, train_step):
    for i in tqdm(range(train_step)):
        newcontainerinfos = workload.generateNewContainers(env.interval) 
        deployed, destroyed = env.addContainers(newcontainerinfos) 
        decisions, actions, log_probs, mainInfo, decoder_inputs = scheduler.run_transformer()
        decisions = scheduler.filter_placement(decisions)
        
        migrations = env.simulationStep(decisions)
        workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) 
        

    best_reward = -1e4
    reward_history = []; score_history = []; remain_cap_history = []
    n_steps = 0
    n_steps1 = 0
    addition_steps = ppoTrainer.config.generat_link_number if flags[0] else 1
    for i in tqdm(range(N_TRAIN_STEPS)):
        for batch in range(len(statePrepares)):
            env.setStatePrepare(statePrepares[0])

            externalObservation, _ = env.reset()
            done = False
            episodeNormalReward = torch.tensor (0.0)
            while not done:
                internalObservations, actions, accepted_action, probs, values, \
                    rewards, steps = ppoTrainer.make_steps(externalObservation, 
                                                           env.statePrepares, flags[0], flags[1])
                print(torch.cat(rewards,0).sum())
                episodeNormalReward += torch.cat(rewards,0).sum()
                externalObservation_, extraReward, done, info = env.step(accepted_action)
                
                if episodeNormalReward < -20: done = True
                ppoTrainer.memory.save_normal_step(externalObservation, actions, \
                                                   probs, values, rewards, done, \
                                                   steps, internalObservations)
                
                n_steps += addition_steps
                if n_steps % ppoTrainer.config.normal_batch == 0:
                    ppoTrainer.train('normal')
                if ~(accepted_action == [-1,-1]).all():
                    #print(accepted_action)
                    n_steps1 += addition_steps
                    ppoTrainer.memory.save_extra_step(externalObservation, actions, \
                                                      probs, values, extraReward, \
                                                      done, steps, internalObservations)   
                    if n_steps1 % ppoTrainer.config.extra_batch == 0:
                        ppoTrainer.train('extra')
                externalObservation = externalObservation_
                
            scores, remain_cap_ratios = env.final_score()
            batch_score_per_grredy = np.mean([s/gs for s,gs in zip(scores, greedyScores[batch])])
            
            reward_history.append(float(episodeNormalReward))
            score_history.append(batch_score_per_grredy)
            remain_cap_history.append(np.mean(remain_cap_ratios))
            avg_reward = np.mean(reward_history[-50:])
            avg_score = np.mean(score_history[-50:])
            
            if avg_reward > best_reward:
                best_reward  = avg_reward
                ppoTrainer.save_models()
            print('episode', i, 'score %.3f' % batch_score_per_grredy, 'avg score %.2f' % avg_score,
                  'time_steps', n_steps, 'remain_cap_ratio %.3f'% np.mean(remain_cap_ratios),
                  'interanl_reward %.3f'%float(episodeNormalReward), 'avg reward %.3f' %avg_reward)
    


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