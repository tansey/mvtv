import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
from pygfl.utils import calc_plateaus, hypercube_edges, edge_map_from_edge_list

if __name__ == '__main__':
    #models = ['cart', 'crisp', 'gfl', 'gamlasso']
    #skiprows = [1, 1, 0, 0]
    models = ['cart', 'gfl', 'gamlasso']
    skiprows = [1, 0, 0]
    datasets = ['violence', 'concrete', 'bikeshares', 'airfoil']
    N = [1994, 103, 731, 1503]
    numtrials = 10
    
    results = np.zeros((len(models), len(datasets), 4, numtrials))
    for j, (dataset, n) in enumerate(zip(datasets, N)):
        print dataset
        for i, (model, skips) in enumerate(zip(models, skiprows)):
            print '\t{0}'.format(model)
            for trial in xrange(numtrials):
                print '\t\t{0}'.format(trial)
                results[i,j,:2,trial] = np.loadtxt('data/uci/{0}/results/{1}/{2}.csv'.format(dataset, model, trial), delimiter=',', skiprows=skips)[:2]
                sweep = np.loadtxt('data/uci/{0}/sweeps/{1}/{2}.csv'.format(dataset, model, trial), delimiter=',', skiprows=skips)
                shape = (1000,1000)
                edges = edge_map_from_edge_list(hypercube_edges(shape))
                results[i,j,2,trial] = len(calc_plateaus(sweep, edges))
                results[i,j,3,trial] = -0.5 * n * results[i,j,0,trial]**2  + results[i,j,2,trial] * (np.log(n) - np.log(2 * np.pi))
                
    agg_results = results.mean(axis=3)
    agg_std = results.std(axis=3)

    dargs = {}
    for j, (dataset, n) in enumerate(zip(datasets, N)):
        dargs['dataset'] = dataset
        dargs['N'] = n
        print '''\multicolumn{1}{l}{}{dataset} & \multicolumn{3}{c}{''' + '{dataset} (N = {N})'.format(**dargs) + '''} \\'''
        print ''' & RMSE & Max Error & Plateau Count \\'''
        print '''\midrule'''
        for i, model in enumerate(models):
            dargs['model'] = model
            dargs['RMSE'] = agg_results[i,j,0]
            dargs['MaxError'] = agg_results[i,j,1]
            dargs['Plateaus'] = agg_results[i,j,2]
            dargs['BIC'] = agg_results[i,j,3]
            dargs['RMSEstd'] = agg_std[i,j,0]
            dargs['MaxErrorstd'] = agg_std[i,j,1]
            dargs['Plateausstd'] = agg_std[i,j,2]
            dargs['BICstd'] = agg_std[i,j,3]
            
            print '''{model} & {RMSE:.4f} ({RMSEstd:.4f}) & {MaxError:.4f} ({MaxErrorstd:.4f}) & {Plateaus:.4f} ({MaxErrorstd:.4f}) \\'''.format(**dargs)
    print '''\midrule'''

