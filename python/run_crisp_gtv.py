import sys
import os
import numpy as np
from crisp_gtv import fit_crisp_gtv, predict
from plot import plot_empirical_means

def create_train_test(dargs, max_size=2000):
    data = np.genfromtxt('data/{dataset_name}.csv'.format(**dargs), delimiter=',', names=True)
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    max_size = min(max_size, len(indices))
    train_indices, test_indices = indices[:int(np.round(max_size * 0.8))], indices[int(np.round(max_size * 0.8)):max_size]
    train_data, test_data = data[train_indices], data[test_indices]
    np.savetxt('data/{dataset_name}_train.csv'.format(**dargs), train_data, delimiter=',')
    np.savetxt('data/{dataset_name}_test.csv'.format(**dargs), test_data, delimiter=',')

def set_column_indices(dargs):
    data = np.genfromtxt('data/{dataset_name}.csv'.format(**dargs), delimiter=',', names=True)
    colname1, colname2 = dargs['x_columns']
    dargs['x_columns'] = [data.dtype.names.index(colname1),data.dtype.names.index(colname2)]
    print '{0}={1} {2}={3}'.format(colname1, dargs['x_columns'][0], colname2, dargs['x_columns'][1])

def int_or_string(x):
    try:
        result = int(x)
    except:
        result = x
    return result

if __name__ == '__main__':
    dargs = {}
    dargs['dataset_name'] = sys.argv[1]
    #dargs['x_columns'] = ['PopDens','medIncome']
    dargs['x_columns'] = [int_or_string(sys.argv[2]),int_or_string(sys.argv[3])]
    tf_k = int(sys.argv[4])
    penalty = sys.argv[5]
    if penalty != 'gfl' and penalty != 'dp' and penalty != 'gamlasso':
        raise Exception('penalty must be gfl or dp or gamlasso')

    if not os.path.exists('data/{dataset_name}_train.csv'.format(**dargs)):
        create_train_test(dargs)

    train_data = np.loadtxt('data/{dataset_name}_train.csv'.format(**dargs), delimiter=',')
    test_data = np.loadtxt('data/{dataset_name}_test.csv'.format(**dargs), delimiter=',')
    if isinstance(dargs['x_columns'][0], basestring):
        set_column_indices(dargs)

    X, y = train_data[:,dargs['x_columns']], train_data[:,-1]
    X_test, y_test = test_data[:,dargs['x_columns']], test_data[:,-1]

    results = fit_crisp_gtv(X, y, q_cv=False, maxlam=100., numlam=30, num_q=50, tf_k=tf_k, penalty=penalty)
    print 'Best q: {0} Best lambda: {1}'.format(results['best_q'], results['best_lambda'])

    if penalty == 'gamlasso':
        dargs['model_name'] = 'gamlasso'
    elif penalty == 'dp':
        dargs['model_name'] = 'gdp'
    elif penalty == 'gfl' and tf_k == 0:
        dargs['model_name'] = 'gtv'
    else:
        dargs['model_name'] = 'gtf{0}'.format(tf_k)
    np.savetxt('data/{model_name}_predictions_{dataset_name}.csv'.format(**dargs), predict(results['best'], results['grid'], X_test), delimiter=',')

    plot_empirical_means('plots/{dataset_name}_empirical_means.pdf'.format(**dargs), X, y, results['grid'])
