import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import networkx as nx
from pygfl.utils import calc_plateaus, hypercube_edges, edge_map_from_edge_list

if __name__ == '__main__':
    cities = [('austin', 'austin2014', 100), ('chicago', 'chicago2015', 200)]
    for city, _, q in cities:
        models = ['cart', 'crisp', 'gapcrisp', 'gfl']
        names = ['CART', 'CRISP', 'GapCRISP', 'GapTV']
        skiprows = [1, 1, 1, 0]
        numtrials = 20
        data = np.loadtxt('data/crime/{0}/all.csv'.format(city), delimiter=',')
        shape = (q,q)
        N = shape[0] * shape[1]

        # Get all the nodes that exist in the grid
        nodeset = set([i*shape[0]+j for i,j,_ in data])
        edges = hypercube_edges(shape)
        edges = edge_map_from_edge_list(edges)

        results = np.zeros((len(models), 6, numtrials))
        for i, (model, skips) in enumerate(zip(models, skiprows)):
            print '\t{0}'.format(model)
            for trial in xrange(numtrials):
                results[i,:2,trial] = np.loadtxt('data/crime/{0}/results/{1}/{2}.csv'.format(city, model, trial), delimiter=',', skiprows=skips)[:2]
                sweep = np.loadtxt('data/crime/{0}/sweeps/{1}/{2}.csv'.format(city, model, trial), delimiter=',', skiprows=skips)
                results[i,2,trial] = len(calc_plateaus(sweep, edges)) #/ float(components)
                results[i,3,trial] = 0.5 * N * results[i,0,trial]**2  + results[i,2,trial] * (np.log(N) - np.log(2 * np.pi))
                results[i,4,trial] = N * results[i,0,trial]**2  + 2 * results[i,2,trial]
                results[i,5,trial] = results[i,4,trial] + 2 * results[i,2,trial] * (results[i,2,trial]+1) / (N - results[i,2,trial] - 1.)
                if trial == 0:
                    with sns.axes_style('white'):
                        plt.rc('font', weight='bold')
                        plt.rc('grid', lw=2)
                        plt.rc('lines', lw=3)
                        
                        heatmap = np.zeros(shape)
                        heatmap[:,:] = np.nan
                        sweep = sweep.reshape(shape)
                        for x,y,_ in data:
                            x,y = int(x), int(y)
                            heatmap[x,y] = sweep[x,y]

                        plt.figure(1)
                        plt.imshow(heatmap.T, origin='lower', cmap='plasma', vmin=0, vmax=7 , interpolation='none')
                        plt.gca().set_xticks([])
                        plt.gca().set_xticklabels([])
                        plt.gca().set_yticks([])
                        plt.gca().set_yticklabels([])
                        plt.savefig('plots/{0}/example-{1}.pdf'.format(city, model), bbox_inches='tight')
                        plt.clf()
                        plt.close()

                    # Create patches
                    if i == 0:
                        grid = np.zeros(shape)
                        for x,y,_ in data:
                            x,y = int(x), int(y)
                            grid[x,y] = 1
                        
                        true_vals = np.loadtxt('data/crime/{}/train/0.csv'.format(city), delimiter=',')
                        true_grid = np.zeros(shape)
                        for x,y,val in true_vals:
                            true_grid[x,y] = val

                        all_patchsizes = [4,5,6,7]
                        all_patches = []
                        for patchsize in all_patchsizes:
                            patches = []
                            for idx1 in xrange(grid.shape[0]-patchsize+1):
                                for idx2 in xrange(grid.shape[1]-patchsize+1):
                                    if grid[idx1:idx1+patchsize,idx2:idx2+patchsize].sum() == patchsize**2:
                                        patches.append((idx1, idx2))
                            all_patches.append(patches)
                            print 'Found {} patches of size {}'.format(len(patches), patchsize)

                        # Save patches for each model
                        for patchsize, patches in zip(all_patchsizes, all_patches):
                            arr_patches = np.zeros((len(patches), patchsize, patchsize))
                            for patch_num, (x,y) in enumerate(patches):
                                arr_patches[patch_num] = true_grid[x:x+patchsize,y:y+patchsize]
                            np.save('data/crime/{}/truth_{}patches'.format(city, patchsize), arr_patches)
                        print 'Saved true patches!'

                    # Save patches for each model
                    for patchsize, patches in zip(all_patchsizes, all_patches):
                        arr_patches = np.zeros((len(patches), patchsize, patchsize))
                        for patch_num, (x,y) in enumerate(patches):
                            arr_patches[patch_num] = sweep[x:x+patchsize,y:y+patchsize]
                        np.save('data/crime/{}/{}_{}patches'.format(city, model, patchsize), arr_patches)
                    print 'Saved patches!'
                
        agg_results = results.mean(axis=2)
        agg_std = results.std(axis=2)

        dargs = {}
        dargs['dataset'] = '{0} Crime Data'.format(city)
        print '''\multicolumn{1}{l}{}{dataset} & \multicolumn{3}{c}{''' + '{dataset}'.format(**dargs) + '''} \\'''
        print ''' & RMSE & Max error & Plateaus & AIC & BIC \\'''
        print '''\midrule'''
        for i, (model, name) in enumerate(zip(models, names)):
            dargs['name'] = name
            dargs['model'] = model
            dargs['RMSE'] = agg_results[i,0]
            dargs['MaxError'] = agg_results[i,1]
            dargs['Plateaus'] = agg_results[i,2]
            dargs['BIC'] = agg_results[i,3]
            dargs['AIC'] = agg_results[i,4]
            dargs['AICc'] = agg_results[i,5]
            dargs['RMSEstd'] = agg_std[i,0]
            dargs['MaxErrorstd'] = agg_std[i,1]
            dargs['Plateausstd'] = agg_std[i,2]
            dargs['BICstd'] = agg_std[i,3]
            dargs['AICstd'] = agg_std[i,4]
            dargs['AICcstd'] = agg_std[i,5]
            print '''{name} & {RMSE:.4f} ({RMSEstd:.4f}) & {MaxError:.4f} ({MaxErrorstd:.4f}) & {Plateaus:.4f} ({MaxErrorstd:.4f}) & {AIC:.4f} ({AICstd:.4f}) & {AICc:.4f} ({AICcstd:.4f}) & {BIC:.4f} ({BICstd:.4f}) \\'''.format(**dargs)
        print '''\midrule'''

