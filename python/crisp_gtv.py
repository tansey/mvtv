import matplotlib
import matplotlib.pylab as plt
import numpy as np
from pygfl.utils import hypercube_edges
from matplotlib.patches import Rectangle
from networkx import Graph
from pygfl.trails import decompose_graph
from pygfl.solver import TrailSolver
from pygfl.trendfiltering import TrendFilteringSolver
from pygfl.utils import *
from scenarios import scenario1, scenario2, scenario3, scenario4, sample_scenario
from gap import gap_statistic
from evaluate import mse

def predict(beta, grid, X):
    i = 0
    predictions = np.zeros(len(X))
    for x1_left, x1_right in zip(grid[0][:-1], grid[0][1:]):
        for x2_left, x2_right in zip(grid[1][:-1], grid[1][1:]):
            vals = np.where((X[:,0] >= x1_left) * (X[:,0] < x1_right) * (X[:,1] >= x2_left) * (X[:,1] < x2_right))[0]
            if len(vals) > 0:
                predictions[vals] = beta[i]
            i += 1
    return predictions

def generate_grid(X, q):
    # Generate percentile bins along each dimension
    grid = []
    for i,qi in enumerate(q):
        percentiles = np.linspace(0, 100, qi+1)
        grid.append(np.array([np.percentile(X[:,i], p) for p in percentiles]))
        grid[-1][0] = -np.inf
        grid[-1][-1] = np.inf
    return grid

def train_gtv(X, y, q, minlam=0.2, maxlam=10., numlam=30, verbose=1, tf_k=0, penalty='gfl', **kwargs):
    if isinstance(q, int):
        q = (q, q)

    grid = generate_grid(X, q)

    # Divide the space into q^2 bins
    data = np.zeros(q[0]*q[1])
    weights = np.zeros(q[0]*q[1])
    i = 0
    for x1_left, x1_right in zip(grid[0][:-1], grid[0][1:]):
        for x2_left, x2_right in zip(grid[1][:-1], grid[1][1:]):
            vals = np.where((X[:,0] >= x1_left) * (X[:,0] < x1_right) * (X[:,1] >= x2_left) * (X[:,1] < x2_right))[0]
            weights[i] = len(vals)
            data[i] = y[vals].mean() if len(vals) > 0 else 0
            i += 1

    # Get the edges for a 2d grid
    edges = hypercube_edges(q)

    ########### Setup the graph
    if penalty == 'gfl':
        g = Graph()
        g.add_edges_from(edges)
        chains = decompose_graph(g, heuristic='greedy')
        ntrails, trails, breakpoints, edges = chains_to_trails(chains)
    elif penalty == 'dp' or penalty == 'gamlasso':
        trails = np.array(edges, dtype='int32').flatten()
        breakpoints = np.array(range(2, len(trails)+1, 2), dtype='int32')
        ntrails = len(breakpoints)

    print '\tSetting up trail solver'
    solver = TrailSolver(maxsteps=30000, penalty=penalty)

    # Set the data and pre-cache any necessary structures
    solver.set_data(data, edges, ntrails, trails, breakpoints, weights=weights)

    print '\tSolving'
    # Grid search to find the best lambda
    results = solver.solution_path(minlam, maxlam, numlam, verbose=verbose)
    results['grid'] = grid
    return results

def create_folds(X, k):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in xrange(k):
        start = end
        end = start + len(indices) / k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[start:end])
    return folds

def fit_crisp_gtv_cv(X, y, q, folds, **kwargs):
    # Use cross-validation to select lambda
    cv_scores = None
    for i,fold in enumerate(folds):
        print '\tFold #{0}'.format(i)
        mask = np.ones(len(X), dtype=bool)
        mask[fold] = False
        x_fold, y_fold = X[mask], y[mask]
        results = train_gtv(x_fold, y_fold, q, verbose=0, **kwargs)
        if cv_scores is None:
            cv_scores = np.zeros(len(results['lambda']))
        fold_score = np.array([mse(y[fold], predict(b, results['grid'], X[fold])) for b in results['beta']])
        cv_scores += fold_score
    cv_scores /= float(len(folds))
    return cv_scores

def fit_crisp_gtv(X, y, num_q=50, q_vals=None, num_cv_folds=5, q_cv=False, **kwargs):
    if q_vals is None:
        q_vals = np.arange(num_q) + 2

    y_mean, y_stdev = y.mean(), y.std()
    y = (y - y_mean) / y_stdev

    folds = create_folds(X, num_cv_folds)

    if q_cv:
        # Use cross-validation to get both q and lambda
        q_scores = []
        for q in q_vals:
            print 'q={0}'.format(q)
            cv_scores = fit_crisp_gtv_cv(X, y, q, folds, **kwargs)
            q_scores.append(cv_scores)
        q_scores = np.array(q_scores)
        q_idx, lam_idx = np.unravel_index(np.argmin(q_scores), q_scores.shape)
        q = q_vals[q_idx]
        results = train_gtv(X, y, q)
        beta = results['beta'][lam_idx]
        grid = results['grid']
    else:
        # Use the regression gap statistic to select q
        print 'Calculating gap statistic'
        q_arr, null_scores, empirical_scores, null_sds = gap_statistic(X, y, q_vals, **kwargs)
        q_scores = null_scores - empirical_scores
        # gap_proportion = (null_scores - empirical_scores) / (null_scores - null_sds)
        gap = null_scores - empirical_scores
        q = q_arr[np.argmax(empirical_scores)]
        # q = q_arr[np.argmax(gap_proportion)]
        # q_viable = q_scores[:-1] >= (q_scores[1:] - 0.*null_sds[1:])
        # q_viable = np.array(list(q_viable) + [1])
        # q = q_arr[np.where(q_viable)[0][0]]
        for q_i, null_score, empirical_score, null_sd in zip(q_vals, null_scores, empirical_scores, null_sds):
            print 'q={} null={} empirical={} null std={} gap={} gap-sd= {} gap/null={}'.format(q_i, null_score, empirical_score, null_sd, null_score - empirical_score, null_score - empirical_score - 0.1*null_sd, (null_score - empirical_score) / (null_score - null_sd))
        print 'Q: {}'.format(q)
        print 'Finding penalty parameter via {0}-fold cross-validation'.format(num_cv_folds)
        cv_scores = fit_crisp_gtv_cv(X, y, q, folds, **kwargs)
        lam_idx = np.argmin(cv_scores)
        results = train_gtv(X, y, q, **kwargs)
        results['best'] = results['beta'][lam_idx]
    results['best'] *= y_stdev
    results['best'] += y_mean
    results['best_q'] = q
    results['best_lambda'] = results['lambda'][lam_idx]
    return results

if __name__ == '__main__':
    for N in [100, 500]:
        # The corresponding scenario from the CRISP paper
        for s, scenario in enumerate([scenario1, scenario2, scenario3, scenario4]):
            print 'Scenario {0} (N={1})'.format(s+1, N)

            X = np.loadtxt('data/x_scenario{0}_n{1}.csv'.format(s+1, N), delimiter=',')
            y = np.loadtxt('data/y_scenario{0}_n{1}.csv'.format(s+1, N), delimiter=',')
            X_test = np.loadtxt('data/test_x_scenario{0}_n{1}.csv'.format(s+1, N), delimiter=',')
            y_test = np.loadtxt('data/test_y_scenario{0}_n{1}.csv'.format(s+1, N), delimiter=',')

            # run the crisp GTV
            #results = fit_crisp_gtv(X, y)
            results = fit_crisp_gtv(X, y, q_cv=False, maxlam=100, numlam=50, num_q=100)
            beta = results['best']
            grid = results['grid']

            print '\tSaving beta to file'
            np.savetxt('data/gtv_grid_scenario{0}_n{1}.csv'.format(s+1, N), grid, delimiter=',')
            np.savetxt('data/gtv_beta_scenario{0}_n{1}.csv'.format(s+1, N), beta, delimiter=',')
            np.savetxt('data/gtv_predictions_scenario{0}_n{1}.csv'.format(s+1, N), predict(beta, grid, X_test), delimiter=',')

            
            # plt.figure()
            # scores = [mse(y_test, predict(b, grid, x_test)) for b in results['beta']]
            # plt.plot(results['lambda'], cv_scores, color='orange', label='CV MSE')
            # plt.plot(results['lambda'], scores, color='blue', label='Test MSE')
            # plt.axvline(results['lambda'][results['best_idx']], color='gray', ls='--', lw=3, label='BIC choice')
            # plt.axvline(results['lambda'][np.argmin(cv_scores)], color='orange', ls='--', lw=3, label='CV choice')
            # plt.xlabel('Lambda value')
            # plt.ylabel('mean squared error')
            # plt.title('Comparison of BIC vs. CV performance (Scenario {0}, n={1})'.format(s+1, N))
            # plt.legend(loc='lower right')
            # plt.savefig('plots/gtv_cv_bic_scenario{0}_n{1}.pdf'.format(s+1, N), bbox_inches='tight')



