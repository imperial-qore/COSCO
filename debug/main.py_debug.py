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
	    
		logger.basicConfig(filename=logFile, level=logger.DEBUG,
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

	if env != '':
		# Destroy environment if required
		eval('destroy'+env+'Environment(configFile, mode)')

		# Quit InfluxDB
		if 'Windows' in platform.system():
			os.system('taskkill /f /im influxd.exe')

	saveStats(stats, datacenter, workload, env)

