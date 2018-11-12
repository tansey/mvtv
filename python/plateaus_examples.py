import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    models = ['cart', 'crisp', 'gapcrisp', 'gfl']
    skiprows = [1, 1, 1, 0]
    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    shape = (100,100)
    trial = 1

    for j, N in enumerate(sample_sizes):
        for i, (model, skips) in enumerate(zip(models, skiprows)):
            yhat = np.loadtxt('data/plateaus/predictions/{0}/{1}/{2}.csv'.format(model, N, trial), delimiter=',', skiprows=skips)
            with sns.axes_style('white'):
                plt.rc('font', weight='bold')
                plt.rc('grid', lw=2)
                plt.rc('lines', lw=3)
                
                plt.figure(1)
                plt.imshow(yhat.reshape(shape), vmin=-7, vmax=7, cmap='gray_r', interpolation='none')
                plt.savefig('plots/plateaus-example-{0}-{1}.pdf'.format(model, N), bbox_inches='tight')
                plt.clf()
                plt.close()


    with sns.axes_style('white'):
        plt.rc('font', weight='bold')
        plt.rc('grid', lw=2)
        plt.rc('lines', lw=3)

        truth = np.loadtxt('data/plateaus/truth/{0}.csv'.format(trial), delimiter=',')[:,2]
        
        plt.figure(1)
        plt.imshow(truth.reshape(shape), vmin=-7, vmax=7, cmap='gray_r', interpolation='none')
        plt.savefig('plots/plateaus-example-truth.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
            



