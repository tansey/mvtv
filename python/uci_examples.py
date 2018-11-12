import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    models = ['cart', 'gfl', 'gamlasso'] # TEMP
    skiprows = [1, 0, 0]
    datasets = ['violence', 'concrete', 'bikeshares', 'airfoil']
    shape = (1000,1000)

    for j, dataset in enumerate(datasets):
        print dataset
        for i, (model, skips) in enumerate(zip(models, skiprows)):
            trial = 0
            yhat = np.loadtxt('data/uci/{0}/sweeps/{1}/0.csv'.format(dataset, model), delimiter=',', skiprows=skips)
            with sns.axes_style('white'):
                plt.rc('font', weight='bold')
                plt.rc('grid', lw=2)
                plt.rc('lines', lw=3)
                
                plt.figure(1)
                plt.imshow(yhat.reshape(shape), vmin=0, vmax=1, cmap='gray_r')
                ax1 = plt.gca()
                ax1.axes.xaxis.set_ticklabels([])
                ax1.axes.yaxis.set_ticklabels([])
                plt.savefig('plots/uci-example-{0}-{1}.pdf'.format(dataset, model), bbox_inches='tight')
                plt.clf()
                plt.close()
            



