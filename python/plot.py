import matplotlib
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
import numpy as np

def plot_empirical_means(filename, X, y, grid, xlim=None, ylim=None, cmap=None):
    if xlim is None:
        xlim = X[:,0].min(), X[:,0].max()
    if ylim is None:
        ylim = X[:,1].min(), X[:,1].max()
    if cmap is None:
        cmap = matplotlib.cm.get_cmap('gray')
    plt.figure()
    plt.xlim(xlim)
    plt.ylim(ylim)
    norm = matplotlib.colors.Normalize(vmin=y.min(), vmax=y.max())
    grid = np.copy(grid)
    grid[0] = ylim[0]-1e-12
    grid[-1] = ylim[1]+1e-12
    grid[:,0] = xlim[0]-1e-12
    grid[:,-1] = xlim[1]+1e-12
    i = 0
    for x1_left, x1_right in zip(grid[0][:-1], grid[0][1:]):
        for x2_left, x2_right in zip(grid[1][:-1], grid[1][1:]):
            vals = np.where((X[:,0] >= x1_left) * (X[:,0] < x1_right) * (X[:,1] >= x2_left) * (X[:,1] < x2_right))[0]
            color = cmap(norm(y[vals].mean() if len(vals) > 0 else 0))
            plt.gca().add_patch(Rectangle((x1_left, x2_left), x1_right-x1_left+1e-12, x2_right-x2_left+1e-12, facecolor=color, edgecolor=color))
            i += 1
    plt.title('Empirical Means')
    plt.savefig(filename, bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    for N in [100, 500]:
        # The corresponding scenario from the CRISP paper
        for s, scenario in enumerate([scenario1, scenario2, scenario3, scenario4]):
            print 'Scenario {0} (N={1})'.format(s+1, N)
            # Draw the results
            fig = plt.figure(figsize=(16,3))
            gs = matplotlib.gridspec.GridSpec(1, 6,
                                   width_ratios=[10,10,10,10,10,1]
                                   )
            ax0 = plt.subplot(gs[0])
            ax1 = plt.subplot(gs[1])
            ax2 = plt.subplot(gs[2])
            ax3 = plt.subplot(gs[3])
            ax4 = plt.subplot(gs[4])
            ax5 = plt.subplot(gs[5])


            cmap = matplotlib.cm.get_cmap('gray')

            truth_grid = []
            for x1 in np.linspace(2.5, -2.5, 50):
                for x2 in np.linspace(2.5, -2.5, 50):
                    truth_grid.append([x1,x2])
            truth_grid = np.array(truth_grid)
            Z = scenario(truth_grid)
            norm = matplotlib.colors.Normalize(vmin=Z.min(), vmax=Z.max())
            ax0.set_xlim([-2.5,2.5])
            ax0.set_ylim([-2.5,2.5])
            for (x1, x2), z in zip(truth_grid, Z):
                ax0.add_patch(Rectangle((x1, x2), 5./50., 5./50., facecolor=cmap(norm(z)), edgecolor=cmap(norm(z))))
            ax0.set_title('Truth')

            ax1.set_xlim([-2.5,2.5])
            ax1.set_ylim([-2.5,2.5])
            ax1.scatter(x[:,0], x[:,1], color=[cmap(norm(yi)) for yi in y])
            ax1.set_title('Observations (N={0})'.format(N))

            ax2.set_xlim([-2.5,2.5])
            ax2.set_ylim([-2.5,2.5])
            i = 0
            for x1_left, x1_right in zip(grid[0][:-1], grid[0][1:]):
                for x2_left, x2_right in zip(grid[1][:-1], grid[1][1:]):
                    vals = np.where((x[:,0] >= x1_left) * (x[:,0] < x1_right) * (x[:,1] >= x2_left) * (x[:,1] < x2_right))[0]
                    weights[i] = len(vals)
                    color = cmap(norm(y[vals].mean() if len(vals) > 0 else 0))
                    ax2.add_patch(Rectangle((x1_left, x2_left), x1_right-x1_left+1e-12, x2_right-x2_left+1e-12, facecolor=color, edgecolor=color))
                    i += 1
            ax2.set_title('Empirical Means')

            ax3.set_xlim([-2.5,2.5])
            ax3.set_ylim([-2.5,2.5])
            i = 0
            for x1_left, x1_right in zip(grid[0][:-1], grid[0][1:]):
                for x2_left, x2_right in zip(grid[1][:-1], grid[1][1:]):
                    color = cmap(norm(beta[i]))
                    ax3.add_patch(Rectangle((x1_left, x2_left), x1_right-x1_left+1e-12, x2_right-x2_left+1e-12, facecolor=color, edgecolor=color))
                    i += 1
            ax3.set_title('GFL')

            ax4.plot(q_vals, q_scores, lw=3)
            ax4.axvline(q, color='r', ls='--', lw=3)
            ax4.set_xlabel('q')
            ax4.set_ylabel('gap score')
            ax4.set_title('Gap Score vs. q')

            matplotlib.colorbar.ColorbarBase(ax5, cmap=cmap,
                                            norm=norm,
                                            orientation='vertical')

            plt.tight_layout()
            plt.savefig('plots/scenario{0}_n{1}.pdf'.format(s+1, N))
            plt.clf()