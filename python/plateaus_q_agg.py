import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns

if __name__ == '__main__':
    shape = (100,100)

    for N in [50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        print 'N: {0}'.format(N)
        q_gfl = np.zeros((49,6))
        q_gamlasso = np.zeros((49,6))
        chosen_q = 0
        for trial in xrange(100):
            cur_gfl = np.loadtxt('data/plateaus/q/gfl/{0}/{1}.csv'.format(N, trial), delimiter=',')
            cur_gamlasso = np.loadtxt('data/plateaus/q/gamlasso/{0}/{1}.csv'.format(N, trial), delimiter=',')
            q_gfl[:,:4] += cur_gfl
            q_gamlasso[:,:4] += cur_gamlasso
            chosen_q += np.loadtxt('data/plateaus/results/gfl/{0}/{1}.csv'.format(N, trial), delimiter=',')[2]
            q_gfl[:,4] += 0.5 * shape[0]*shape[1] * cur_gfl[:,1]**2  + cur_gfl[:,3] * (np.log(shape[0]*shape[1]) - np.log(2 * np.pi))
            q_gfl[:,5] += shape[0]*shape[1] * cur_gfl[:,1]**2  + 2 * cur_gfl[:,3]
            
        q_gfl /= 100.
        q_gamlasso /= 100.
        chosen_q /= 100.

        with sns.axes_style('white'):
            plt.rc('font', weight='bold')
            plt.rc('grid', lw=2)
            plt.rc('lines', lw=3)

            plt.figure(1)
            ax1 = plt.gca()
            plt.plot(q_gfl[:,0], q_gfl[:,1], c='blue', lw=4, label='RMSE')
            ax1.set_ylabel('RMSE', weight='bold', fontsize=24)
            ax2 = ax1.twinx()
            ax2.plot(q_gfl[:,0], q_gfl[:,2], c='orange', lw=4, label='Max Error')
            ax2.set_ylabel('Max Error', weight='bold', fontsize=24)
            plt.axvline(chosen_q, lw=3, ls='--', color='red')
            plt.xlabel('q', weight='bold', fontsize=24)
            plt.savefig('plots/q-gfl-rmse-maxerr-{0}.pdf'.format(N), bbox_inches='tight')
            plt.clf()
            plt.close()

            plt.figure(1)
            plt.plot(q_gfl[:,0], q_gfl[:,4], c='blue', lw=4, label='BIC')
            ax1 = plt.gca()
            ax1.set_ylabel('BIC', weight='bold', fontsize=24)
            ax2 = ax1.twinx()
            ax2.plot(q_gfl[:,0], q_gfl[:,2], c='orange', lw=4, label='Max Error')
            ax2.set_ylabel('Max Error', weight='bold', fontsize=24)
            plt.axvline(chosen_q, lw=3, ls='--', color='red')
            plt.xlabel('q', weight='bold', fontsize=24)
            plt.savefig('plots/q-gfl-bic-maxerr-{0}.pdf'.format(N), bbox_inches='tight')
            plt.clf()
            plt.close()

            plt.figure(1)
            plt.plot(q_gfl[:,0], q_gfl[:,1], c='blue', lw=4, label='BIC')
            ax1 = plt.gca()
            ax1.set_ylabel('RMSE', weight='bold', fontsize=24)
            ax2 = ax1.twinx()
            ax2.plot(q_gfl[:,0], q_gfl[:,3], c='orange', lw=4, label='Max Error')
            ax2.set_ylabel('Plateaus', weight='bold', fontsize=24)
            plt.axvline(chosen_q, lw=3, ls='--', color='red')
            plt.xlabel('q', weight='bold', fontsize=24)
            plt.savefig('plots/q-gfl-rmse-plateaus-{0}.pdf'.format(N), bbox_inches='tight')
            plt.clf()
            plt.close()

            plt.figure(1)
            ax1 = plt.gca()
            plt.plot(q_gfl[:,0], q_gfl[:,1], c='blue', lw=4, label='RMSE')
            ax1.set_ylabel('AIC', weight='bold', fontsize=24)
            plt.axvline(chosen_q, lw=3, ls='--', color='red')
            plt.xlabel('q', weight='bold', fontsize=24)
            plt.savefig('plots/q-gfl-aic-{0}.pdf'.format(N), bbox_inches='tight')
            plt.clf()
            plt.close()



