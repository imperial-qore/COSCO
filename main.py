from environment.Datacenter.SimpleFog import *
from workload.StaticWorkload_StaticDistribution import *
from environment.Environment import *
from scheduler.Random import *
from stats.Stats import *

# Global constants
NUM_SIM_STEPS = 100
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
	migrated = env.allocateInit(decision) # Schedule containers
	env.allocateInit()

	# Initialize stats
	stats = Stats(env, workload, datacenter, scheduler)
	stats.saveStats(deployed, migrated, [], containerlist, containerlist, decision)
	return workload, scheduler, env, stats

def stepSimulation(workload, scheduler, env, stats):
	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
	deployed = env.addContainers(newcontainerinfos) # Deploy new containers and get container IDs
	workload.updateDeployedContainers(env.getCreationIDs(deployed)) # Update workload deployed using creation IDs
	selected = scheduler.selected() # Select container IDs for migration
	decision = scheduler.placement(selected) # Decide placement for selected container ids
	migrated = env.allocateInit(decision) # Schedule containers
	env.allocateInit()

	stats.saveStats(deployed, migrated, destroyed, newcontainers, selectedcontainers, decision)

def saveStats(stats):
	filename = "logs/" + datacenter.__class__.__name__
	filename += "_" + workload.__class__.__name__
	filename += "_" + str(NUM_SIM_STEPS) 
	filename += "_" + str(HOSTS)
	filename += "_" + str(CONTAINERS)
	filename += "_" + str(TOTAL_POWER)
	filename += "_" + str(ROUTER_BW)
	filename += "_" + str(INTERVAL_TIME)
	filename += "_" + str(NEW_CONTAINERS)
	with open(filename+'.pk', 'wb') as handle:
	    pickle.dump(stats, handle)

if __name__ == '__main__':

	workload, scheduler, env, stats = initalizeEnvironment()

	for step in range(NUM_SIM_STEPS):
		stepSimulation(workload, scheduler, env, stats)

	saveStats(stats)

