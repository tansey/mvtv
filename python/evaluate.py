import numpy as np
from utils import mse

if __name__ == '__main__':
    # for s in xrange(4):
    #     for N in [100, 500]:
    #         y = np.loadtxt('data/test_y_scenario{0}_n{1}.csv'.format(s+1, N), delimiter=',')
    #         y_crisp = np.loadtxt('data/crisp_predictions_scenario{0}_n{1}.csv'.format(s+1, N), delimiter=',', skiprows=1)
    #         y_gtv = np.loadtxt('data/gtv_predictions_scenario{0}_n{1}.csv'.format(s+1, N), delimiter=',')

    #         print ''
    #         print 'Scenario {0} (N={1})'.format(s+1, N)
    #         print 'CRISP: {0}'.format(mse(y, y_crisp))
    #         print 'GTV: {0}'.format(mse(y, y_gtv))
    #         print ''

    dargs = {}
    datasets = ['violence', 'concrete', 'bikeshares', 'plateaus', 'accidents', 'airfoil']
    indices = [(119,17), (0,6), (9, 10), (0,1), (0,1), (0, 4)]
    urls = ['https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime',
            'https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test',
            'https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset',
            '',
            'https://data.gov.uk/dataset/road-accidents-safety-data',
            'https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise']
    for dataset_name, x_columns in zip(datasets, indices):
        dargs['dataset_name'] = dataset_name
        train_data = np.loadtxt('data/{dataset_name}_train.csv'.format(**dargs), delimiter=',')
        test_data = np.loadtxt('data/{dataset_name}_test.csv'.format(**dargs), delimiter=',')
        y_cart = np.loadtxt('data/cart_predictions_{dataset_name}.csv'.format(**dargs), delimiter=',', skiprows=1)
        y_tps = np.loadtxt('data/tps_predictions_{dataset_name}.csv'.format(**dargs), delimiter=',', skiprows=1)
        y_crisp = np.loadtxt('data/crisp_predictions_{dataset_name}.csv'.format(**dargs), delimiter=',', skiprows=1)
        y_gtv = np.loadtxt('data/gtv_predictions_{dataset_name}.csv'.format(**dargs), delimiter=',')
        #y_gtf_2 = np.loadtxt('data/gtf2_predictions_{dataset_name}.csv'.format(**dargs), delimiter=',', skiprows=1)
        # y_gdp = np.loadtxt('data/gdp_predictions_{dataset_name}.csv'.format(**dargs), delimiter=',')
        y_gamlasso = np.loadtxt('data/gamlasso_predictions_{dataset_name}.csv'.format(**dargs), delimiter=',')
        X_test, y_test = test_data[:,x_columns], test_data[:,-1]
        dargs['N'] = len(train_data)
        print ''
        print '{dataset_name} data (N={N})'.format(**dargs)
        print 'CART: {0}'.format(mse(y_test, y_cart))
        print 'TPS: {0}'.format(mse(y_test, y_tps))
        print 'CRISP: {0}'.format(mse(y_test, y_crisp))
        print 'GTV: {0}'.format(mse(y_test, y_gtv))
        #print 'GTF (k=2): {0}'.format(mse(y_test, y_gtf_2))
        # print 'GDP: {0}'.format(mse(y_test, y_gdp))
        print 'GamLasso: {0}'.format(mse(y_test, y_gamlasso))
        print ''

