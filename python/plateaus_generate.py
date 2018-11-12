import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
from pygfl.utils import create_plateaus, grid_graph_edges
from utils import make_directory
import sys

def create_rectangular_plateaus(data, plateau_size, plateau_vals, plateaus=None):
    '''Creates rectangular plateaus of constant value in the data.'''
    if plateaus is None:
        plateaus = []
        width, height = plateau_size
        for val in plateau_vals:
            center_x, center_y = np.random.randint(data.shape[0]), np.random.randint(data.shape[1])
            plateau = []
            for i in xrange(max(0,int(np.round(center_x-width/2))), min(data.shape[1],int(np.round(center_x+width/2)))):
                for j in xrange(max(0,int(np.round(center_y-height/2))), min(data.shape[0],int(np.round(center_y+height/2)))):
                    plateau.append((i,j))
                    data[i,j] = val
            plateaus.append(np.array(plateau, dtype=int))
    return plateaus

if __name__ == '__main__':
    plateau_type = sys.argv[1]
    data_dir = make_directory('data/', plateau_type)
    train_dir = make_directory(data_dir, 'train')
    truth_dir = make_directory(data_dir, 'truth')
    q_dir = make_directory(data_dir, 'q')
    q_gfl = make_directory(q_dir, 'gfl')
    q_gamlasso = make_directory(q_dir, 'gamlasso')
    prediction_dir = make_directory(data_dir, 'predictions')
    prediction_cart = make_directory(prediction_dir, 'cart')
    prediction_gfl = make_directory(prediction_dir, 'gfl')
    prediction_gamlasso = make_directory(prediction_dir, 'gamlasso')
    prediction_crisp = make_directory(prediction_dir, 'crisp')
    prediction_gapcrisp = make_directory(prediction_dir, 'gapcrisp')
    results_dir = make_directory(data_dir, 'results')
    results_cart = make_directory(results_dir, 'cart')
    results_gfl = make_directory(results_dir, 'gfl')
    results_gamlasso = make_directory(results_dir, 'gamlasso')
    results_crisp = make_directory(results_dir, 'crisp')
    results_gapcrisp = make_directory(results_dir, 'gapcrisp')

    plots_dir = make_directory('plots/', plateau_type)
    for trial in xrange(100):
        print trial
        k = 0
        shape = (100,100)
        data = np.zeros(shape).flatten()
        plateau_size = 1000
        plateau_vals = np.array([-2, -3, -5, 5, 3, 2])
        edges = grid_graph_edges(shape[0], shape[1])
        if plateau_type == 'rectangular':
            plateaus = create_rectangular_plateaus(data.reshape(shape), (32,32), plateau_vals)
        else:
            plateaus = create_plateaus(data, edges, plateau_size, plateau_vals)
        data = data.reshape(shape)

        for N in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
            train_n_dir = make_directory('data/{0}/train/'.format(plateau_type), str(N))
            make_directory(results_cart, str(N))
            make_directory(results_gfl, str(N))
            make_directory(results_gamlasso, str(N))
            make_directory(results_crisp, str(N))
            make_directory(results_gapcrisp, str(N))

            make_directory(prediction_cart, str(N))
            make_directory(prediction_gfl, str(N))
            make_directory(prediction_gamlasso, str(N))
            make_directory(prediction_crisp, str(N))
            make_directory(prediction_gapcrisp, str(N))

            make_directory(q_gfl, str(N))
            make_directory(q_gamlasso, str(N))

            dataset = np.zeros((N,3))
            for i in xrange(N):
                x1, x2 = np.random.randint(shape[0], size=2)
                dataset[i] = [x1, x2, data[x1, x2] + np.random.normal()]
            np.savetxt('data/{0}/train/{1}/{2}.csv'.format(plateau_type, N,trial), dataset, delimiter=',')

            if trial < 10:
                plt.figure()
                plt.scatter(dataset[:,0], dataset[:,1], c=dataset[:,2], vmin=-6, vmax=6, cmap='gray_r')
                plt.xlim([0,shape[0]-1])
                plt.ylim([0,shape[0]-1])
                plt.savefig('plots/{0}/{1}-observed-{2}.pdf'.format(plateau_type, trial, N), bbox_inches='tight')
                plt.clf()
                plt.close()

        truth = []
        for i in xrange(data.shape[0]):
            for j in xrange(data.shape[1]):
                truth.append([i, j, data[i,j]])
        np.savetxt('data/{0}/truth/{1}.csv'.format(plateau_type, trial), truth, delimiter=',')

        if trial < 10:
            plt.figure()
            plt.imshow(data, vmin=-6, vmax=6, cmap='gray_r', interpolation='none')
            plt.gca().invert_yaxis()
            plt.savefig('plots/{0}/{1}-truth.pdf'.format(plateau_type, trial), bbox_inches='tight')
            plt.clf()
            plt.close()

        

