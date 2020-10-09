import os, sys, stat
import sys
import optparse
import logging as logger
import configparser
import pickle
import shutil
import sqlite3
import platform
from subprocess import call
from os import startfile, system, rename

# Framework imports
from framework.Framework import *
from framework.database.Database import *
from framework.datacenter.Datacenter_Setup import *
from framework.datacenter.Datacenter import *
from framework.workload.DeFogWorkload import *

# Simulator imports
from simulator.Simulator import *
from simulator.environment.BitbrainFog import *
from simulator.workload.BitbrainWorkload_GaussianDistribution import *

# Scheduler imports
from scheduler.IQR_MMT_Random import IQRMMTRScheduler
from scheduler.MAD_MMT_Random import MADMMTRScheduler
from scheduler.Random_Random_FirstFit import RFScheduler
from scheduler.Random_Random_LeastFull import RLScheduler
from scheduler.RLR_MMT_Random import RLRMMTRScheduler
from scheduler.Threshold_MC_Random import TMCRScheduler
from scheduler.Random_Random_Random import RandomScheduler
from scheduler.GOBI import GOBIScheduler

# Auxilliary imports
from stats.Stats import *
from utils.Utils import *
from pdb import set_trace as bp

# Global constants
NUM_SIM_STEPS = 5
HOSTS = 50
CONTAINERS = 50
TOTAL_POWER = 1000
ROUTER_BW = 10000
INTERVAL_TIME = 10 # seconds
NEW_CONTAINERS = 2
DB_NAME = ''
DB_HOST = ''
DB_PORT = 0
HOSTS_IP = []
logFile = 'COSCO.log'

def initalizeEnvironment(environment, logger):
	if environment != '':
		# Initialize the db
		db = Database(DB_NAME, DB_HOST, DB_PORT)

	# Initialize simple fog datacenter
	''' Can be SimpleFog, BitbrainFog // Datacenter '''
	if environment != '':
		datacenter = Datacenter(HOSTS_IP, environment)
	else:
		datacenter = BitbrainFog(HOSTS)

	# Initialize workload
	''' Can be SWSD, BWGD // DFW '''
	if environment != '':
		workload = DFW(NEW_CONTAINERS, db)
	else: 
		workload = BWGD(NEW_CONTAINERS, 3)
	
	# Initialize scheduler
	''' Can be LRMMTR, RF, RL, RM, Random, RLRMMTR, TMMR, TMMTR, GA, GOBI '''
	scheduler = RandomScheduler()

	# Initialize Environment
	hostlist = datacenter.generateHosts()
	if environment != '':
		env = Framework(scheduler, CONTAINERS, INTERVAL_TIME, hostlist, db, environment, logger)
	else:
		env = Simulator(TOTAL_POWER, ROUTER_BW, scheduler, CONTAINERS, INTERVAL_TIME, hostlist)

	#######
	# a = env.controller.create({"fields": {'name': str(4)+'_2', 'image': 'shreshthtuli/yolo'}}, "192.168.0.2")
	# a = env.controller.getContainerStat("192.168.0.2")
	# for	ccid in range(4, 9, 1):
	# 	a = env.controller.create({"fields": {'name': str(ccid)+'_2', 'image': 'shreshthtuli/yolo'}}, "192.168.0.3")
	# 	a = env.controller.checkpoint(ccid, 2, "192.168.0.3")
	# 	a = env.controller.migrate(ccid, 2, "192.168.0.3", "192.168.0.2")
	# 	a = env.controller.restore(ccid, 2, "shreshthtuli/yolo", "192.168.0.2")
	# 	print(a)
	# print(a)
	# exit()
	#######

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

	# Initialize stats
	stats = Stats(env, workload, datacenter, scheduler)
	stats.saveStats(deployed, migrations, [], deployed, decision)
	return datacenter, workload, scheduler, env, stats

def stepSimulation(workload, scheduler, env, stats):
	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
	bp()
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
	# stats.generateGraphs(dirname)
	stats.generateDatasets(dirname)
	with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
	    pickle.dump(stats, handle)
	if 'Datacenter' in datacenter.__class__.__name__:
		rename(logFile, dirname+'/'+logFile)

if __name__ == '__main__':
	usage = "usage: python main.py -e <environment> -m <mode> # empty environment run simulator"

	parser = optparse.OptionParser(usage=usage)
	parser.add_option("-e", "--environment", action="store", dest="env", default="", 
						help="Environment is AWS, Openstack, Azure, VLAN, Vagrant")
	parser.add_option("-m", "--mode", action="store", dest="mode", default="0", 
						help="Mode is 0 (Create and destroy), 1 (Create), 2 (No op), 3 (Destroy)")
	opts, args = parser.parse_args()
	env, mode = opts.env, int(opts.mode)

	if env != '':
		# Convert all agent files to unix format
		unixify(['framework/agent/', 'framework/agent/scripts/'])

		# Start InfluxDB service
		print(color.HEADER+'InfluxDB service runs as a separate front-end window. Please minimize this window.'+color.ENDC)
		if 'Windows' in platform.system():
			startfile('C:/Program Files/InfluxDB/influxdb-1.8.3-1/influxd.exe')

		configFile = 'framework/config/' + opts.env + '_config.json'
	    
		logger.basicConfig(filename=logFile, level=logging.DEBUG,
	                        format='%(asctime)s - %(levelname)s - %(message)s')
		logger.debug("Creating enviornment in :{}".format(env))
		cfg = {}
		with open(configFile, "r") as f:
			cfg = json.load(f)
		DB_HOST = cfg['database']['ip']
		DB_PORT = cfg['database']['port']
		DB_NAME = 'COSCO'

		if env == 'AWS':
			cfg["AccessKey"] = config.get(env, 'AccessKey')
			cfg["SecretAcessKey"] = config.get(env, 'SecretAcessKey')
		elif env == 'Openstack':
			cfg["image"] = config.get(env, 'image')
	        # cfg["key_name"] = config.get(env, 'key_name')
	        # cfg["network"] = config.get(env, 'network')
	        # cfg["flavours"] = list(json.loads(config.get(env, 'flavours')))
	        # logging.debug("Creating enviornment with configuration file as  :{}".format(cfg))
	        # setup(cfg)
		elif env == 'Azure':
			cfg["AccessKey"] = config.get(env, 'AccessKey')
			# cfg["SecretAcessKey"] = config.get(env, 'SecretAcessKey')
		elif env == 'Vagrant':
			print("Setting up VirtualBox environment using Vagrant")
			HOSTS_IP = setupVagrantEnvironment(configFile, mode)
			print(HOSTS_IP)
		elif env == 'VLAN':
			pass

	datacenter, workload, scheduler, env, stats = initalizeEnvironment(env, logger)

	for step in range(NUM_SIM_STEPS):
		print(color.BOLD+"Simulation Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, env, stats)
	saveStats(stats, datacenter, workload)

	if env == '':
		# Destroy environment if required
		eval('destroy'+env+'Enviornment(configFile, mode)')

		# Quit InfluxDB
		os.system('taskkill /f /im influxd.exe')

	saveStats(stats, datacenter, workload)

