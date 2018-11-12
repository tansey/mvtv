import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from pygfl.easy import solve_gfl
from utils import create_folds, bucket_vals, bin_indices, mse

def var_sums(x, y, q):
    breakpoints = np.linspace(0, 1, q+1)
    s = 0
    for b1, b2 in zip(breakpoints[:-1], breakpoints[1:]):
        y_i = y[((x[:,0] >= b1) & (x[:,0] < b2))]
        if len(y_i) > 1:
            s += y_i.var()
    return s

def predict(X, beta, grid):
    indices = np.indices([len(g)-1 for g in grid])
    if len(grid) == 1:
        indices = indices[0]
    bins = bin_indices(X, grid)
    predictions = beta[bins]
    return predictions

def gtv_cvlam(X, y, q, num_folds=5, num_lams=20):
    n = len(X)
    folds = create_folds(n, num_folds)
    scores = np.zeros(num_lams)
    lams = None
    for i,fold in enumerate(folds):
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        x_train, y_train = X[mask], y[mask]
        x_test, y_test = X[~mask], y[~mask]
        data, weights, grid = bucket_vals(x_train, y_train, q)
        results = solve_gfl(data, None, weights=weights, full_path=True, minlam=0.1, maxlam=20., numlam=num_lams)
        fold_score = np.array([mse(y_test, predict(x_test, beta, grid)) for beta in results['beta']])
        scores += fold_score
        if i == 0:
            lams = results['lambda']
    scores /= float(num_folds)
    lam_best = lams[np.argmin(scores)]
    data, weights, grid = bucket_vals(X, y, q)
    beta = solve_gfl(data, None, weights=weights, lam=lam_best)
    return beta.reshape(q), grid

def plot_example(x, y, q_vals):
    q = q_vals[np.argmax(np.array([var_sums(x,y,q) for q in q_vals]))]
    beta, grid = gtv_cvlam(x, y, (q,))
    bins = np.linspace(0, 1, 100)[:,np.newaxis]
    predictions = predict(bins, beta, grid)
    plt.scatter(x[:,0], y, color='blue')
    plt.plot(bins, predictions, color='orange')
    for b in breakpoints[1:-1]:
        plt.axvline(b, ls='--', lw=3, color='green')
    plt.savefig('plots/1d-example.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    breakpoints = np.array([0,0.15, 0.5, 0.75, 1.])
    means = np.array([0, 1., 4, -2])
    nchangepoints = len(means) - 1
    N = [10, 20, 30, 50, 100]
    ntrials = 100
    qmax = 50
    q_vals = np.arange(qmax-1)+2
    avg_q = np.zeros(len(N))
    avg_changepoints = np.zeros((len(N), len(q_vals)))
    for nidx, n in enumerate(N):
        print 'N={}'.format(n)
        avg_scores = np.zeros(len(q_vals))
        avg_errors = np.zeros(len(q_vals))
        for trial in xrange(ntrials):
            x = np.random.random(size=(n,1))
            y = np.random.normal(size=n)
            for m, b1, b2 in zip(means, breakpoints[:-1], breakpoints[1:]):
                y[((x[:,0] >= b1) & (x[:,0] < b2))] += m

            scores = np.array([(-np.inf if q > n else var_sums(x,y,q)) for q in q_vals])
            avg_scores += scores / ntrials

            # Run MVTV
            # plot_example(x, y, q_vals)
            avg_q[nidx] += q_vals[np.argmax(scores)] / float(ntrials)

            # Enumerate q, show how the error relates to MVTV choice
            neval = 20
            errors = np.zeros(len(q_vals))
            changepoints = np.zeros(len(q_vals))
            bins = np.linspace(0, 1, neval)[:,np.newaxis]
            truth = np.zeros(neval)
            for m, b1, b2 in zip(means, breakpoints[:-1], breakpoints[1:]):
                truth[((bins[:,0] >= b1) & (bins[:,0] < b2))] = m
            for i,q in enumerate(q_vals):
                if q > n:
                    errors[i] = np.nan
                    changepoints[i] = np.nan
                else:
                    beta, grid = gtv_cvlam(x, y, (q,))
                    predictions = predict(bins, beta, grid)[:,0]
                    errors[i] = mse(truth, predictions)
                    changepoints[i] = (np.abs(predictions[:-1] - predictions[1:]) > 0.01).sum()
            avg_errors += errors / float(ntrials)
            avg_changepoints[nidx] += changepoints / float(ntrials)
            #     if i == 0:
            #         plt.plot(bins, truth - predictions, color='gray')
            #     if q == q_vals[np.argmax(scores)]:
            #         plt.plot(bins, truth - predictions, color='orange')
            # plt.savefig('plots/test.pdf')
            # plt.close()
        # plt.scatter(q_vals, avg_changepoints)
        # plt.axvline(avg_q, ls='--', lw=3, color='orange')
        # plt.axhline(nchangepoints, ls='--', lw=3, color='green')
        # plt.savefig('plots/1d-changepoints.pdf', bbox_inches='tight')
        # plt.close()
    colors = ['blue', 'orange', 'green', 'purple', 'goldenrod', 'red', 'brown']
    styles = ['-', '-', '-', '-', '-', '-', '-']
    for n, q, changes, color, ls in zip(N, avg_q, avg_changepoints, colors, styles):
        plt.plot(q_vals, changes, color=color, ls=ls, label='{}'.format(n), lw=3)
        plt.axvline(q, ls='--', lw=3, color=color)
    plt.axhline(nchangepoints, ls='--', lw=3, color='gray', label='True change points')
    plt.legend(loc='upper left')
    plt.savefig('plots/1d-q-growth.pdf', bbox_inches='tight')
    # with sns.axes_style('white'):
    #     plt.rc('font', weight='bold')
    #     plt.rc('grid', lw=2)
    #     plt.rc('lines', lw=3)

    #     plt.scatter(x, y)
    

