from environment.Datacenter.SimpleFog import *
from environment.Datacenter.BitbrainFog import *
from scheduler.IQR_MMT_Random import IQRMMTRScheduler
from scheduler.MAD_MMT_Random import MADMMTRScheduler
from scheduler.Random_Random_FirstFit import RFScheduler
from scheduler.Random_Random_LeastFull import RLScheduler
from scheduler.RLR_MMT_Random import RLRMMTRScheduler
from scheduler.Threshold_MC_Random import TMCRScheduler
from scheduler.Random_Random_Random import RandomScheduler
from scheduler.GOBI import GOBIScheduler
from scheduler.GA import GAScheduler
from scheduler.DRL import DRLScheduler
from workload.StaticWorkload_StaticDistribution import *
from workload.BitbrainWorkload_GaussianDistribution import *
from environment.Environment import *
from stats.Stats import *
from utils.Utils import *
import os
import pickle
import shutil

# Global constants
NUM_SIM_STEPS = 100
HOSTS = 50
CONTAINERS = 50
TOTAL_POWER = 1000
ROUTER_BW = 10000
INTERVAL_TIME = 300 # seconds
NEW_CONTAINERS = 7

def initalizeEnvironment():
	# Initialize simple fog datacenter
	''' Can be SimpleFog, BitbrainFog '''
	datacenter = BitbrainFog(HOSTS)

	# Initialize workload
	''' Can be SWSD, BWGD '''
	workload = BWGD(NEW_CONTAINERS, 3)
	
	# Initialize scheduler
	''' Can be LRMMTR, RF, RL, RM, Random, RLRMMTR, TMMR, TMMTR, GA, GOBI, DRL '''
	scheduler = DRLScheduler('energy')

	# Initialize Environment
	hostlist = datacenter.generateHosts()
	env = Environment(TOTAL_POWER, ROUTER_BW, scheduler, CONTAINERS, HOSTS, INTERVAL_TIME, hostlist)
	
	# Initialize stats
	stats = Stats(env, workload, datacenter, scheduler)

	# Execute first step
	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
	deployed = env.addContainersInit(newcontainerinfos) # Deploy new containers and get container IDs
	decision = scheduler.placement(deployed) # Decide placement using container ids
	migrations = env.allocateInit(decision) # Schedule containers
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # Update workload allocated using creation IDs
	print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
	print("Containers in host:", env.getContainersInHosts())
	print("Schedule:", env.getActiveContainerList())
	printDecisionAndMigrations(decision, migrations)

	stats.saveStats(deployed, migrations, [], deployed, decision)
	return datacenter, workload, scheduler, env, stats

def stepSimulation(workload, scheduler, env, stats):
	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
	deployed, destroyed = env.addContainers(newcontainerinfos) # Deploy new containers and get container IDs
	selected = scheduler.selection() # Select container IDs for migration
	decision = scheduler.placement(selected+deployed) # Decide placement for selected container ids
	migrations = env.simulationStep(decision) # Schedule containers
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # Update workload deployed using creation IDs
	print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
	print("Deployed:", len(env.getCreationIDs(migrations, deployed)), "of", len(newcontainerinfos), [i[0] for i in newcontainerinfos])
	print("Destroyed:", len(destroyed), "of", env.getNumActiveContainers())
	print("Containers in host:", env.getContainersInHosts())
	print("Num active containers:", env.getNumActiveContainers())
	print("Host allocation:", [(c.getHostID() if c else -1)for c in env.containerlist])
	printDecisionAndMigrations(decision, migrations)

	stats.saveStats(deployed, migrations, destroyed, selected, decision)

def saveStats(stats, datacenter, workload):
	dirname = "logs/" + datacenter.__class__.__name__
	dirname += "_" + workload.__class__.__name__
	dirname += "_" + str(NUM_SIM_STEPS) 
	dirname += "_" + str(HOSTS)
	dirname += "_" + str(CONTAINERS)
	dirname += "_" + str(TOTAL_POWER)
	dirname += "_" + str(ROUTER_BW)
	dirname += "_" + str(INTERVAL_TIME)
	dirname += "_" + str(NEW_CONTAINERS)
	if not os.path.exists("logs"): os.mkdir("logs")
	if os.path.exists(dirname): shutil.rmtree(dirname, ignore_errors=True)
	os.mkdir(dirname)
	stats.generateGraphs(dirname)
	stats.generateDatasets(dirname)
	with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
	    pickle.dump(stats, handle)

if __name__ == '__main__':

	datacenter, workload, scheduler, env, stats = initalizeEnvironment()

	for step in range(NUM_SIM_STEPS):
		print(color.BOLD+"Simulation Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, env, stats)

	saveStats(stats, datacenter, workload)

