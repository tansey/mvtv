import numpy as np

def scenario1(x):
    return np.sign(x[:,0]) * ((x[:,0] * x[:,1]) >= 0)

def scenario2(x):
    return -np.sign(x[:,0] * x[:,1])

def scenario3(x):
    return (- 3 * (x[:,0] >= -2.5) * (x[:,0] < -0.83) * (x[:,1] >= -2.5) * (x[:,1] < -1.25)
            +     (x[:,0] >= -2.5) * (x[:,0] < -0.83) * (x[:,1] >= -1.25) * (x[:,1] <= 2.5)
            - 2 * (x[:,0] >= -0.83) * (x[:,0] <= 0.83) * (x[:,1] >= -2.5) * (x[:,1] < 0)
            + 2 * (x[:,0] >= -0.83) * (x[:,0] <= 0.83) * (x[:,1] >= 0) * (x[:,1] <= 2.5)
            -     (x[:,0] > 0.83) * (x[:,0] <= 2.5) * (x[:,1] >= -2.5) * (x[:,1] < 1.25)
            + 3 * (x[:,0] > 0.83) * (x[:,0] <= 2.5) * (x[:,1] >= 1.25) * (x[:,1] <= 2.5))

def scenario4(x):
    return (10. / (((x[:,0] - 2.5) / 3.)**2 + ((x[:,1] - 2.5) / 3.)**2 + 1)
            + 10. / (((x[:,0] + 2.5) / 3.)**2 + ((x[:,1] + 2.5) / 3.)**2 + 1))

def sample_scenario(s, n):
    # Sample x1, x2 from Uniform[-2.5, 2.5]
    X = np.random.random(size=(n,2)) * 5 - 2.5

    # Sample values from noisy versions of the observations
    scenarios = [scenario1, scenario2, scenario3, scenario4]
    scenario = scenarios[s]
    y = scenario(X) + np.random.normal(size=len(X))

    return X, y
