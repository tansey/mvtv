import sys
import os
import numpy as np
from crisp_gtv import fit_crisp_gtv, predict
from plot import plot_empirical_means
from utils import mse, max_error

if __name__ == '__main__':
    city = sys.argv[1]
    trial = int(sys.argv[2])

    shape = (100,100)
    train = np.loadtxt('data/crime/{0}/train/{1}.csv'.format(city,trial), delimiter=',')
    test = np.loadtxt('data/crime/{0}/test/{1}.csv'.format(city,trial), delimiter=',')
    sweep = np.loadtxt('data/crime/{0}/sweep.csv'.format(city), delimiter=',')
    
    x_columns = (0,1)
    X = train[:,x_columns]
    y = train[:,2]
    
    X_test = test[:,x_columns]
    y_test = test[:,2]

    gfl_results = fit_crisp_gtv(X, y, q_cv=False, minlam=0.1, maxlam=100., numlam=50, num_q=100, penalty='gfl')
    gfl_y_hat = predict(gfl_results['best'], gfl_results['grid'], X_test)
    gfl_rmse = np.sqrt(mse(y_test, gfl_y_hat))
    gfl_maxerr = max_error(y_test, gfl_y_hat)
    gfl_sweep = predict(gfl_results['best'], gfl_results['grid'], sweep)
    
    gamlasso_results = fit_crisp_gtv(X, y, q_cv=False, minlam=0.1, maxlam=100., numlam=50, num_q=100, penalty='gamlasso')
    gamlasso_y_hat = predict(gamlasso_results['best'], gamlasso_results['grid'], X_test)
    gamlasso_rmse = np.sqrt(mse(y_test, gamlasso_y_hat))
    gamlasso_maxerr = max_error(y_test, gamlasso_y_hat)
    gamlasso_sweep = predict(gamlasso_results['best'], gamlasso_results['grid'], sweep)

    print ''
    print 'GFL -->      Best q: {0} Best lambda: {1} RMSE: {2:.4f} Max Error: {3:.4f}'.format(gfl_results['best_q'], gfl_results['best_lambda'], gfl_rmse, gfl_maxerr)
    print 'GamLasso --> Best q: {0} Best lambda: {1} RMSE: {2:.4f} Max Error: {3:.4f}'.format(gamlasso_results['best_q'], gamlasso_results['best_lambda'], gamlasso_rmse, gamlasso_maxerr)

    np.savetxt('data/crime/{0}/results/gfl/{1}.csv'.format(city, trial), [gfl_rmse, gfl_maxerr, gfl_results['best_q'], gfl_results['best_lambda']], delimiter=',')
    np.savetxt('data/crime/{0}/results/gamlasso/{1}.csv'.format(city, trial), [gamlasso_rmse, gamlasso_maxerr, gamlasso_results['best_q'], gamlasso_results['best_lambda']], delimiter=',')

    np.savetxt('data/crime/{0}/predictions/gfl/{1}.csv'.format(city, trial), gfl_y_hat, delimiter=',')
    np.savetxt('data/crime/{0}/predictions/gamlasso/{1}.csv'.format(city, trial), gamlasso_y_hat, delimiter=',')

    np.savetxt('data/crime/{0}/sweeps/gfl/{1}.csv'.format(city, trial), gfl_sweep, delimiter=',')
    np.savetxt('data/crime/{0}/sweeps/gamlasso/{1}.csv'.format(city, trial), gamlasso_sweep, delimiter=',')

