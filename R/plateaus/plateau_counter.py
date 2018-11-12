import sys
import numpy as np
from pygfl.utils import calc_plateaus, hypercube_edges, edge_map_from_edge_list

def count_plateaus(data):
    shape = (100,100)
    edges = edge_map_from_edge_list(hypercube_edges(shape))
    return len(calc_plateaus(data, edges))

filename = sys.argv[1]
data = np.loadtxt(filename, delimiter=',', skiprows=1)
count = count_plateaus(data)
with open(filename, 'wb') as f:
    f.write(str(count) + '\n')