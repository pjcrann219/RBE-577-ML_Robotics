import numpy as np
import time
from matplotlib import pyplot as plt

def getMiniBatch(l = 10**6):

    # Define ranges for data
    F1_range = [-10000, 10000]
    F2_range = [-5000, 5000]
    F3_range = [-5000, 5000]
    a2_range = [-180, 180]
    a3_range = [-180, 180]

    # Define step sizes for random walk
    F_step = 1000
    a_step = 10

    # Initiate arrays
    F1s = np.zeros(shape=(l,))
    F2s = np.zeros(shape=(l,))
    F3s = np.zeros(shape=(l,))
    a2s= np.zeros(shape=(l,))
    a3s = np.zeros(shape=(l,))

    # Pick a random startin point
    F1s[0] = np.random.uniform(low=F1_range[0], high=F1_range[1])
    F2s[0] = np.random.uniform(low=F2_range[0], high=F2_range[1])
    F3s[0] = np.random.uniform(low=F3_range[0], high=F3_range[1])
    a2s[0] = np.random.uniform(low=a2_range[0], high=a2_range[1])
    a3s[0] = np.random.uniform(low=a3_range[0], high=a3_range[1])

    # Perform random walk
    for i in range(1, l):
        F1s[i] = np.clip(F1s[i-1] + F_step * np.random.uniform(low=-1.0, high=1.0), a_min=F1_range[0], a_max=F1_range[1])
        F2s[i] = np.clip(F2s[i-1] + F_step * np.random.uniform(low=-1.0, high=1.0), a_min=F2_range[0], a_max=F2_range[1])
        F3s[i] = np.clip(F3s[i-1] + F_step * np.random.uniform(low=-1.0, high=1.0), a_min=F3_range[0], a_max=F3_range[1])
        a2s[i] = np.clip(a2s[i-1] + a_step * np.random.uniform(low=-1.0, high=1.0), a_min=a2_range[0], a_max=a2_range[1])
        a3s[i] = np.clip(a3s[i-1] + a_step * np.random.uniform(low=-1.0, high=1.0), a_min=a3_range[0], a_max=a3_range[1])

    # Initiate generalized forces
    tau1s = np.zeros(shape=(l,))
    tau2s = np.zeros(shape=(l,))
    tau3s = np.zeros(shape=(l,))

    # Ship parameters
    l1, l2, l3, l4 = -14, 14.5, -2.7, 2.7

    # Calculate generalized forces at each time
    for i, (f1, f2, f3, a2, a3) in enumerate(zip(F1s, F2s, F3s, a2s, a3s)):
        tau1s[i] = 0 + np.cos(np.deg2rad(a2))*f2 + np.cos(np.deg2rad(a3))*f3
        tau2s[i] = f1 + np.sin(np.deg2rad(a2))*f2 + np.sin(np.deg2rad(a3))*f3
        tau3s[i] = l2*f1 + (l1*np.sin(np.deg2rad(a2)) - l3*np.cos(np.deg2rad(a2)))*f2 + \
        (l1*np.sin(np.deg2rad(a3)) - l4*np.cos(np.deg2rad(a3)))*f3

    # Return all data as 1 array
    return np.vstack([F1s, F2s, F3s, a2s, a3s, tau1s, tau2s, tau3s])
    

if __name__ == "__main__":

    n_batch = 10

    # Compute and save off mini batches
    for i in range(2,n_batch):
        st = time.time()
        print(f"Processing batch {i}... ")
        data = getMiniBatch()
        np.save('data/data_' + str(i) + '.npy', data)
        print(f"\tDone in {time.time() - st}s")