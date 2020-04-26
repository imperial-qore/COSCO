from environment.Datacenter.SimpleFog import *
from workload.StaticWorkload_StaticDistribution import *
from environment.Environment import *
from scheduler.Random import *
from stats.Stats import *
import os
import pickle
import shutil

# Global constants
NUM_SIM_STEPS = 20
HOSTS = 10
CONTAINERS = 10
TOTAL_POWER = 1000
ROUTER_BW = 10000
INTERVAL_TIME = 300 # seconds
NEW_CONTAINERS = 2

def initalizeEnvironment():
	# Initialize simple fog datacenter
	datacenter = SimpleFog(HOSTS)

	# Initialize static workload
	workload = SWSD(NEW_CONTAINERS)
	
	# Initialize random scheduler
	scheduler = RandomScheduler()

	# Initialize Environment
	hostlist = datacenter.generateHosts()
	env = Environment(TOTAL_POWER, ROUTER_BW, scheduler, CONTAINERS, HOSTS, INTERVAL_TIME, hostlist)

	# Execute first step
	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
	deployed = env.addContainersInit(newcontainerinfos) # Deploy new containers and get container IDs
	workload.updateDeployedContainers(env.getCreationIDs(deployed)) # Update workload deployed using creation IDs
	decision = scheduler.placement(deployed) # Decide placement using container ids
	migrations = env.allocateInit(decision) # Schedule containers
	print("Deployed:", len(deployed), "of", len(newcontainerinfos))
	print("Containers in host:", env.getContainersInHosts())
	print("Schedule:", env.getActiveContainerList())

	# Initialize stats
	stats = Stats(env, workload, datacenter, scheduler)
	stats.saveStats(deployed, migrations, [], deployed, decision)
	return datacenter, workload, scheduler, env, stats

def stepSimulation(workload, scheduler, env, stats):
	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
	deployed, destroyed = env.addContainers(newcontainerinfos) # Deploy new containers and get container IDs
	workload.updateDeployedContainers(env.getCreationIDs(deployed)) # Update workload deployed using creation IDs
	selected = scheduler.selection() # Select container IDs for migration
	print("To place: ", deployed)
	decision = scheduler.placement(selected+deployed) # Decide placement for selected container ids
	migrations = env.simulationStep(decision) # Schedule containers
	print("Deployed containers' creation IDs:", env.getCreationIDs(deployed))
	print("Deployed:", len(deployed), "of", len(newcontainerinfos), [i[0] for i in newcontainerinfos])
	print("Destroyed:", len(destroyed), "of", env.getNumActiveContainers())
	print("Containers in host:", env.getContainersInHosts())
	print("Schedule:", env.getActiveContainerList())
	print(decision)
	print()

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
	if not os.path.exists("logs"): os.mkdir(logs)
	if os.path.exists(dirname): shutil.rmtree(dirname, ignore_errors=True)
	os.mkdir(dirname)
	stats.generateGraphs(dirname)
	with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
	    pickle.dump(stats, handle)

if __name__ == '__main__':

	datacenter, workload, scheduler, env, stats = initalizeEnvironment()

	for step in range(NUM_SIM_STEPS):
		stepSimulation(workload, scheduler, env, stats)

	saveStats(stats, datacenter, workload)

