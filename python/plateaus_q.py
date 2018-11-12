import sys
import os
import numpy as np
from crisp_gtv import fit_crisp_gtv_cv, predict, create_folds, train_gtv
from plot import plot_empirical_means
from utils import mse, max_error
from pygfl.utils import calc_plateaus, hypercube_edges, edge_map_from_edge_list

if __name__ == '__main__':
    trial = int(sys.argv[1])
    N = int(sys.argv[2])

    shape = (100,100)
    train = np.loadtxt('data/plateaus/train/{0}/{1}.csv'.format(N, trial), delimiter=',')
    truth = np.loadtxt('data/plateaus/truth/{0}.csv'.format(trial), delimiter=',')
    
    x_columns = (0,1)
    X = train[:,x_columns]
    y = train[:,2]
    
    X_test = truth[:,x_columns]
    y_test = truth[:,2]

    q_gfl = np.zeros((49, 4))
    q_gamlasso = np.zeros((49, 4))
    folds = create_folds(X, 5)
    edges = edge_map_from_edge_list(hypercube_edges(shape))
    for i,q in enumerate(xrange(2, 51)):
        print 'q={0}'.format(q)
        gfl_cv_scores = fit_crisp_gtv_cv(X, y, q, folds)
        lam_idx = np.argmin(gfl_cv_scores)
        gfl_results = train_gtv(X, y, q, minlam=0.1, maxlam=100., numlam=50, penalty='gfl')
        gfl_y_hat = predict(gfl_results['beta'][lam_idx], gfl_results['grid'], X_test)
        gfl_rmse = np.sqrt(mse(y_test, gfl_y_hat))
        gfl_maxerr = max_error(y_test, gfl_y_hat)
        gfl_dof = len(calc_plateaus(gfl_y_hat, edges))
        q_gfl[i] = (q, gfl_rmse, gfl_maxerr, gfl_dof)
        
        gamlasso_cv_scores = fit_crisp_gtv_cv(X, y, q, folds)
        lam_idx = np.argmin(gamlasso_cv_scores)
        gamlasso_results = train_gtv(X, y, q, minlam=0.1, maxlam=100., numlam=50, penalty='gamlasso')
        gamlasso_y_hat = predict(gamlasso_results['beta'][lam_idx], gamlasso_results['grid'], X_test)
        gamlasso_rmse = np.sqrt(mse(y_test, gamlasso_y_hat))
        gamlasso_maxerr = max_error(y_test, gamlasso_y_hat)
        gamlasso_dof = len(calc_plateaus(gamlasso_y_hat, edges))
        q_gamlasso[i] = (q, gamlasso_rmse, gamlasso_maxerr, gamlasso_dof)

    np.savetxt('data/plateaus/q/gfl/{0}/{1}.csv'.format(N, trial), q_gfl, delimiter=',')
    np.savetxt('data/plateaus/q/gamlasso/{0}/{1}.csv'.format(N, trial), q_gamlasso, delimiter=',')

