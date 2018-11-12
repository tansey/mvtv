import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
from utils import make_directory
from crisp_gtv import create_folds

def load_austin_locations(crimes):
    latlons = []
    for location in crimes['Location_1']:
        lines = location.split('\n')
        lat, lon = [float(x.group(0)) for x in re.finditer(r"[0-9\.\-]+", lines[-1])]
        latlons.append((lat, lon))
    return np.array(latlons)

def load_chicago_locations(crimes):
    latlons = crimes[['Latitude','Longitude']]
    latlons = latlons.dropna()
    return latlons.as_matrix()

if __name__ == '__main__':
    cities = [('austin', 'austin2014', 100), ('chicago', 'chicago2015', 200)]
    for name, filename, q in cities:
        crimes = pd.read_csv('data/{0}.csv'.format(filename))
        ntrials = 20

        if name == 'austin':
            latlons = load_austin_locations(crimes)
        elif name == 'chicago':
            latlons = load_chicago_locations(crimes)

        heatmap, xedges, yedges = np.histogram2d(latlons[:,0], latlons[:,1], bins=q)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        heatmap[heatmap == 0] = np.nan
        heatmap[~np.isnan(heatmap)] = np.log(heatmap[~np.isnan(heatmap)])
        
        # plt.hist(heatmap[~np.isnan(heatmap)], bins=30)
        # plt.show()

        data_dir = make_directory('data/', 'crime')
        data_dir = make_directory(data_dir, name)
        plots_dir = make_directory('plots/', name)
        train_dir = make_directory(data_dir, 'train')
        test_dir = make_directory(data_dir, 'test')
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
        sweeps_dir = make_directory(data_dir, 'sweeps')
        sweeps_cart = make_directory(sweeps_dir, 'cart')
        sweeps_gfl = make_directory(sweeps_dir, 'gfl')
        sweeps_gamlasso = make_directory(sweeps_dir, 'gamlasso')
        sweeps_crisp = make_directory(sweeps_dir, 'crisp')
        sweeps_gapcrisp = make_directory(sweeps_dir, 'gapcrisp')

        data = np.array([(i, j, heatmap[i,j]) for i,j in np.ndindex((q,q)) if not np.isnan(heatmap[i,j])])
        np.savetxt(data_dir + 'all.csv', data, delimiter=',')

        folds = create_folds(data, ntrials)
        for trial in xrange(ntrials):
            train = []
            for i in xrange(ntrials):
                if i != trial:
                    train.extend(data[folds[i]])
            train = np.array(train)
            test = np.array(data[folds[trial]])

            np.savetxt(train_dir + '{0}.csv'.format(trial), train, delimiter=',')
            np.savetxt(test_dir + '{0}.csv'.format(trial), test, delimiter=',')

        np.savetxt(data_dir + 'sweep.csv', list(np.ndindex((q,q))), delimiter=',')

        with sns.axes_style('white'):
            plt.rc('font', weight='bold')
            plt.rc('grid', lw=2)
            plt.rc('lines', lw=3)
            
            plt.figure(1)
            plt.imshow(heatmap.T, origin='lower', cmap='plasma', vmin=0, vmax=7, interpolation='none')
            plt.gca().set_xticks([])
            plt.gca().set_xticklabels([])
            plt.gca().set_yticks([])
            plt.gca().set_yticklabels([])
            plt.savefig(plots_dir + 'raw.pdf', bbox_inches='tight')



