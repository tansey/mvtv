import matplotlib
import matplotlib.pylab as plt
import numpy as np
from pygfl.utils import hypercube_edges
from matplotlib.patches import Rectangle
from scenarios import sample_scenario
from gap import regression_gap

for N in [100, 500]:
    # The corresponding scenario from the CRISP paper
    for s, scenario in enumerate([scenario1, scenario2, scenario3, scenario4]):
        print 'Scenario {0} (N={1})'.format(s+1, N)
        x, y = sample_scenario(s, N)
        x_test, y_test = sample_scenario(s, N)

        np.savetxt('data/x_scenario{0}_n{1}.csv'.format(s+1, N), x, delimiter=',')
        np.savetxt('data/y_scenario{0}_n{1}.csv'.format(s+1, N), y, delimiter=',')
        np.savetxt('data/test_x_scenario{0}_n{1}.csv'.format(s+1, N), x_test, delimiter=',')
        np.savetxt('data/test_y_scenario{0}_n{1}.csv'.format(s+1, N), y_test, delimiter=',')