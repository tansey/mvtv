import os
import sys
import csv
import numpy as np


def make_directory(base, subdir):
    if not base.endswith('/'):
        base += '/'
    directory = base + subdir
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not directory.endswith('/'):
        directory = directory + '/'
    return directory

def pretty_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print a matrix or vector.'''
    if len(p.shape) == 1:
        return vector_str(p, decimal_places, ignore)
    if len(p.shape) == 2:
        return matrix_str(p, decimal_places, ignore, label_columns)
    raise Exception('Invalid array with shape {0}'.format(p.shape))

def matrix_str(p, decimal_places=2, ignore=None, label_columns=False):
    '''Pretty-print the matrix.'''
    return '[{0}]'.format("\n  ".join([(str(i) if label_columns else '') + vector_str(a, decimal_places, ignore) for i, a in enumerate(p)]))

def vector_str(p, decimal_places=2, ignore=None):
    '''Pretty-print the vector values.'''
    style = '{0:.' + str(decimal_places) + 'f}'
    return '[{0}]'.format(", ".join([' ' if ((hasattr(ignore, "__len__") and a in ignore) or a == ignore) else style.format(a) for a in p]))

def mse(y, yhat):
    return ((y-yhat)**2).mean()


def max_error(y, yhat):
    return max(np.abs(y-yhat))

def error_variance(y, yhat):
    return np.var(y-yhat)

def create_folds(n, k):
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = []
    start = 0
    end = 0
    for f in xrange(k):
        start = end
        end = start + len(indices) / k + (1 if (len(indices) % k) > f else 0)
        folds.append(indices[start:end])
    return folds

def percentile_grid(X, q):
    '''Generate percentile bins along each dimension'''
    grid = []
    for i,qi in enumerate(q):
        percentiles = np.linspace(0, 100, qi+1)
        grid.append(np.array([np.percentile(X[:,i], p) for p in percentiles]))
        grid[-1][0] = -np.inf
        grid[-1][-1] = np.inf
    return grid

def linear_grid(X, q):
    '''Generate linear bins along each dimension'''
    grid = []
    for i,qi in enumerate(q):
        grid.append(np.linspace(X[:,i].min(), X[:,i].max(), qi+1))
        grid[-1][0] = -np.inf
        grid[-1][-1] = np.inf
    return grid

def bin_indices(X, grid):
    '''Maps each X row to a point in the percentile grid'''
    return np.array([np.digitize(X[:,d], grid[d]) for d in xrange(X.shape[1])]).T - 1

def bucket_vals(X, y, q, grid_type='percentile'):
    # Divide the space into q-space bins
    if not hasattr(q, "__len__"):
        q = [q for _ in xrange(X.shape[1])]
    if len(q) != X.shape[1]:
        raise Exception("q must be of same shape as the number of columns in X.")
    grid = percentile_grid(X, q) if grid_type == 'percentile' else linear_grid(X, q)
    data = np.zeros(q)
    counts = np.zeros(q)
    bins = bin_indices(X, grid)
    for y_i, b_i in zip(y, bins):
        data[tuple(b_i)] += y_i
        counts[tuple(b_i)] += 1.
    data[counts > 0] /= counts[counts > 0]
    data[counts == 0] = np.nan
    return data, counts, grid




