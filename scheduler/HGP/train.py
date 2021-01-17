import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, ConstantKernel as C
from sklearn.cluster import KMeans
from src.custom_kernels import HeteroscedasticKernel
from src.utils import *
from src.HGPopt import *
import pickle
from time import time
import random

def save_model(filename, gp_heteroscedastic):
    file_path = "checkpoints/" + filename + ".pt"
    with open(file_path, 'wb') as f:
        pickle.dump(gp_heteroscedastic, f)

def load_model(filename):
    dtl = filename.split('_')
    al = 1 if '10' in dtl[-1] else 0
    dataset, dataset_size, max_container_ips = eval("load_"+'_'.join(dtl[:-1])+"_data("+dtl[-1]+")")
    X = np.array([np.array(i[0]).reshape(-1) for i in dataset])
    y = np.array([Coeff_Energy*i[1][0] + Coeff_Latency*i[1][1] for i in dataset])
    prototypes = KMeans(n_clusters=10).fit(X).cluster_centers_
    kernel_hetero = C(1.0, (1e-10, 1000)) * RBF(1, (0.01, 100.0)) \
        + HeteroscedasticKernel.construct(prototypes, 1e-3, (1e-10, 50.0),
                                          gamma=5.0, gamma_bounds="fixed")
    gp_heteroscedastic = GaussianProcessRegressor(kernel=kernel_hetero, alpha=al)
    file_path1 = "checkpoints/" + filename + ".pt"
    file_path2 = 'scheduler/HGP/' + file_path1
    file_path = file_path1 if os.path.exists(file_path1) else file_path2
    if os.path.exists(file_path):
        print(color.GREEN+"Loading pre-trained model: "+filename+color.ENDC)
        with open(file_path, 'rb') as f:
            gp_heteroscedastic = pickle.load(f)
    else:
        print(color.GREEN+"Creating new model: "+filename+color.ENDC)
    return gp_heteroscedastic, X, y, max_container_ips

if __name__ == '__main__':
    data_type = argv[1] # can be 'energy_latency' + '_' + str(HOSTS)
    exec_type = argv[2] # can be 'train', 'opt'

    gp_heteroscedastic, X, y, max_container_ips = load_model(data_type)

    if exec_type == "train":
        gp_heteroscedastic.fit(X, y)
        print("Heteroscedastic kernel: %s" % gp_heteroscedastic.kernel_)
        print("Heteroscedastic LML: %.3f" \
            % gp_heteroscedastic.log_marginal_likelihood(gp_heteroscedastic.kernel_.theta))
        save_model(data_type, gp_heteroscedastic)

    else:
        init = random.choice(X)
        x, s = gp_heteroscedastic.predict(init.reshape(1, -1), return_std=True)
        print((x + UCB_K * s)[0])
        start = time()
        result, fitness = HGPopt(init, gp_heteroscedastic, data_type)
        print("Time", time()-start)
        print("Iteration: {}\nResult: {}\nFitness: {}".format(0, result, fitness)) 
