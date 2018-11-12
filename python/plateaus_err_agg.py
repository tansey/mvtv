import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from pygfl.utils import calc_plateaus, hypercube_edges, edge_map_from_edge_list

if __name__ == '__main__':
    models = ['cart', 'crisp', 'gapcrisp', 'gfl']
    names = ['CART', 'CRISP', 'GapCRISP', 'GapTV']
    skiprows = [1, 1, 1, 0]
    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    shape = (100,100)
    edges = edge_map_from_edge_list(hypercube_edges(shape))

    results = np.zeros((len(models), len(sample_sizes), 6, 100))
    for j, N in enumerate(sample_sizes):
        print 'N={0}'.format(N)
        for i, (model, skips) in enumerate(zip(models, skiprows)):
            print '\t{0}'.format(model)
            for trial in xrange(100):
                results[i,j,:2,trial] = np.loadtxt('data/plateaus/results/{0}/{1}/{2}.csv'.format(model, N, trial), delimiter=',', skiprows=skips)[:2]
                results[i,j,2,trial] = len(calc_plateaus(np.loadtxt('data/plateaus/predictions/{0}/{1}/{2}.csv'.format(model, N, trial), delimiter=',', skiprows=skips), edges))
                results[i,j,3,trial] = 0.5 * shape[0]*shape[1] * results[i,j,0,trial]**2  + results[i,j,2,trial] * (np.log(shape[0]*shape[1]) - np.log(2 * np.pi))
                results[i,j,4,trial] = shape[0]*shape[1] * results[i,j,0,trial]**2  + 2 * results[i,j,2,trial]
                results[i,j,5,trial] = results[i,j,4,trial] + 2 * results[i,j,2,trial] * (results[i,j,2,trial]+1) / (shape[0]*shape[1] - results[i,j,2,trial] - 1.)

    agg_results = results.mean(axis=3)
    agg_stderr = results.std(axis=3) / 10. # standard error

    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=2)
        plt.rc('lines', lw=3)
        colors = ['gray', 'blue', 'orange', 'green']
        styles = ['solid', 'dashed', 'dashdot', 'solid']

        plt.figure(1)
        for i, (model, c, ls, name) in enumerate(zip(models, colors, styles, names)):
            plt.plot(np.log(sample_sizes), agg_results[i,:,0], c=c, ls=ls,  lw=4, label=name)
            plt.ylabel('RMSE', weight='bold', fontsize=24)
        plt.xlabel('Log(Sample Size)', weight='bold', fontsize=24)
        plt.legend(loc='upper right')
        plt.savefig('plots/plateaus-rmse.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.figure(1)
        for i, (model, c, ls, name) in enumerate(zip(models, colors, styles, names)):
            plt.plot(np.log(sample_sizes), agg_results[i,:,1], c=c, ls=ls,  lw=4, label=name)
            plt.ylabel('Max Error', weight='bold', fontsize=24)
        plt.xlabel('Log(Sample Size)', weight='bold', fontsize=24)
        plt.savefig('plots/plateaus-maxerr.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.figure(1)
        for i, (model, c, ls, name) in enumerate(zip(models, colors, styles, names)):
            plt.plot(np.log(sample_sizes), np.log(agg_results[i,:,2]), c=c, ls=ls,  lw=4, label=name)
            plt.ylabel('Log(Plateaus)', weight='bold', fontsize=24)
        plt.axhline(np.log(shape[0]*shape[1]), lw=4, ls='--', c='r')
        plt.xlabel('Log(Sample Size)', weight='bold', fontsize=24)
        plt.savefig('plots/plateaus-change-points.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.figure(1)
        for i, (model, c, ls, name) in enumerate(zip(models, colors, styles, names)):
            plt.plot(np.log(sample_sizes), agg_results[i,:,3], c=c, ls=ls,  lw=4, label=name)
            plt.ylabel('BIC', weight='bold', fontsize=24)
        plt.xlabel('Log(Sample Size)', weight='bold', fontsize=24)
        plt.savefig('plots/plateaus-bic.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.figure(1)
        for i, (model, c, ls, name) in enumerate(zip(models, colors, styles, names)):
            plt.plot(np.log(sample_sizes), agg_results[i,:,4], c=c, ls=ls,  lw=4, label=name)
            plt.ylabel('AIC', weight='bold', fontsize=24)
        plt.xlabel('Log(Sample Size)', weight='bold', fontsize=24)
        plt.savefig('plots/plateaus-aic.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.figure(1)
        for i, (model, c, ls, name) in enumerate(zip(models, colors, styles, names)):
            plt.plot(np.log(sample_sizes), agg_results[i,:,5], c=c, ls=ls,  lw=4, label=name)
            plt.ylabel('AICc', weight='bold', fontsize=24)
        plt.xlabel('Log(Sample Size)', weight='bold', fontsize=24)
        plt.savefig('plots/plateaus-aicc.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.figure(1)
        for i, (model, c, ls, name) in enumerate(zip(models, colors, styles, names)):
            plt.plot(np.log(sample_sizes), agg_results[i,:,0] / agg_results[i,:,1], c=c, ls=ls,  lw=4, label=name)
            plt.ylabel('RMSE / Max Error', weight='bold', fontsize=24)
        plt.xlabel('Log(Sample Size)', weight='bold', fontsize=24)
        plt.savefig('plots/plateaus-ratio-rmse-maxerr.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.figure(1)
        for i, (model, c, ls, name) in enumerate(zip(models, colors, styles, names)):
            plt.plot(np.log(sample_sizes), agg_results[i,:,1] / agg_results[i,:,2], c=c, ls=ls,  lw=4, label=name)
            plt.ylabel('Max Error / Plateaus', weight='bold', fontsize=24)
        plt.xlabel('Log(Sample Size)', weight='bold', fontsize=24)
        plt.savefig('plots/plateaus-maxerr-per-plateau.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()



