import numpy as np
import itertools
from scipy.special import digamma

def gaussian_reference_dist(y, n_k):
    sigma2 = y.var()
    v = np.sum(n_k ** 2 - n_k) / 2.0
    # return np.log(2) + digamma(v / 2.0) + np.log(2*sigma2)
    return digamma(v / 2.0) #+ len(n_k) * np.log(2*sigma2)

def sampled_gaussian_reference_dist(y, n_k, nsamples):
    samples = np.zeros(nsamples)
    mu, sigma = y.mean(), y.std()
    for trial in xrange(nsamples):
        s = [np.random.choice(y, size=n_i) for n_i in n_k]
        # s = [np.random.normal(mu, sigma, size=n_i) for n_i in n_k]
        samples[trial] = np.array([pairwise_dists(v) for v in s]).sum()
    return np.log(samples).mean(), np.log(samples).std()
    
def binomial_reference_dist(y, n_k):
    p_i = y.mean()
    n = np.sum(n_k ** 2 - n_k) / 2.0
    p = 2 * p_i * (1-p_i)
    return np.log(p*n) - (1-p) / (2*p*n) # 2nd-order taylor expansion approximation

def pairwise_dists(members):
    sum_distances = 0
    num_distances = 0.
    for m1 in xrange(len(members)):
        for m2 in xrange(m1+1, len(members)):
            sum_distances += (members[m1] - members[m2])**2
            num_distances += 1.
    return sum_distances / num_distances

def gap_statistic(X, y, q_vals, loss='gaussian', nsamples=100, **kwargs):
    # if loss == 'gaussian':
    #     log_null_dist = gaussian_reference_dist(y)
    # elif loss == 'binomial':
    #     log_null_dist = binomial_reference_dist(y)
    #q_arr = np.array(list(itertools.product(q_vals, repeat=2)))
    scores = np.zeros(len(q_vals))
    log_null_dist = np.zeros(len(q_vals))
    log_null_dist_sd = np.zeros(len(q_vals))
    for q_idx,q in enumerate(q_vals):
        from crisp_gtv import generate_grid
        grid = generate_grid(X, (q,q))
        n_k = []
        
        # Calculate the empirical distances
        for x1_left, x1_right in zip(grid[0][:-1], grid[0][1:]):
            for x2_left, x2_right in zip(grid[1][:-1], grid[1][1:]):
                vals = np.where((X[:,0] >= x1_left) * (X[:,0] < x1_right) * (X[:,1] >= x2_left) * (X[:,1] < x2_right))[0]
                if len(vals) > 1:
                    members = y[vals]
                    scores[q_idx] += pairwise_dists(members)
                    n_k.append(len(vals))

        n_k = np.array(n_k)
        if loss == 'gaussian':
            # log_null_dist[q_idx], log_null_dist_sd[q_idx] = sampled_gaussian_reference_dist(y, n_k, nsamples)
            log_null_dist[q_idx] = gaussian_reference_dist(y, n_k)
        elif loss == 'binomial':
            log_null_dist[q_idx] = binomial_reference_dist(y, n_k)
    print 'Vals:'
    for q, null, score in zip(q_vals, log_null_dist, np.log(scores)):
        print '{}: {}\t{}\t{}'.format(q, null, score, null - score)
    return q_vals, log_null_dist, np.log(scores), log_null_dist_sd * np.sqrt(1 + 1./nsamples)




