import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from utils import make_directory
from crisp_gtv import create_folds
import seaborn as sns

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
    uci_dir = make_directory('data/', 'uci')
    plots_dir = make_directory('plots/', 'uci')

    datasets = ['violence', 'concrete', 'bikeshares', 'airfoil']
    columns = [['medIncome','PopDens','ViolentCrimesPerPop'],
                ['Water','Coarse Aggr.','Compressive Strength (28-day)(Mpa)'],
                ['temp', 'windspeed', 'cnt'],
                ['freq', 'attackangle', 'soundpressure']]
    xlims = [(0,1), (150,250), (0,1), (0,25000)]
    ylims = [(0,1), (650,1100), (0,0.6), (-1,25)]

    x_min, y_min, x_max, y_max = None, None, None, None

    for dataset, cols, xlim, ylim in zip(datasets, columns, xlims, ylims):
        print dataset
        data_dir = make_directory(uci_dir, dataset)
        train_dir = make_directory(data_dir, 'train')
        test_dir = make_directory(data_dir, 'test')
        prediction_dir = make_directory(data_dir, 'predictions')
        prediction_cart = make_directory(prediction_dir, 'cart')
        prediction_gfl = make_directory(prediction_dir, 'gfl')
        prediction_gamlasso = make_directory(prediction_dir, 'gamlasso')
        prediction_crisp = make_directory(prediction_dir, 'crisp')
        results_dir = make_directory(data_dir, 'results')
        results_cart = make_directory(results_dir, 'cart')
        results_gfl = make_directory(results_dir, 'gfl')
        results_gamlasso = make_directory(results_dir, 'gamlasso')
        results_crisp = make_directory(results_dir, 'crisp')
        sweeps_dir = make_directory(data_dir, 'sweeps')
        sweeps_cart = make_directory(sweeps_dir, 'cart')
        sweeps_gfl = make_directory(sweeps_dir, 'gfl')
        sweeps_gamlasso = make_directory(sweeps_dir, 'gamlasso')
        sweeps_crisp = make_directory(sweeps_dir, 'crisp')
        
        dataframe = pd.read_csv('data/{0}.csv'.format(dataset))
        data = dataframe.as_matrix(columns=cols)
        folds = create_folds(data, 10)

        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-6) + 1e-12

        for trial in xrange(10):
            train = []
            for i in xrange(10):
                if i != trial:
                    train.extend(data[folds[i]])
            train = np.array(train)
            test = np.array(data[folds[i]])                

            np.savetxt(train_dir + '{0}.csv'.format(trial), train, delimiter=',')
            np.savetxt(test_dir + '{0}.csv'.format(trial), test, delimiter=',')

        with sns.axes_style('white'):
            plt.rc('font', weight='bold')
            plt.rc('grid', lw=2)
            plt.rc('lines', lw=3)
            plt.figure()
            plt.scatter(data[:,0], data[:,1], c=data[:,2], cmap='plasma', s=75)
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.xlabel(cols[0], weight='bold', fontsize=24)
            plt.ylabel(cols[1], weight='bold', fontsize=24)
            plt.savefig('plots/uci/{0}-observed.pdf'.format(dataset), bbox_inches='tight')
            plt.clf()
            plt.close()

    sweep_vals = []
    for x in np.linspace(0, 1, 1000):
        for y in np.linspace(0, 1, 1000):
            sweep_vals.append((x,y))
    np.savetxt(uci_dir + 'sweep.csv', sweep_vals, delimiter=',')
    

