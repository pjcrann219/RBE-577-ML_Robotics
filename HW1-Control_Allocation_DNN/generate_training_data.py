import numpy as np

forces = np.random.uniform(low=-10, high=10, size=(5, 10))


F1, F2, F3, a1, a2 = forces


def transform(forces):
    F1, F2, F3, a1, a2 = forces
    
    tau = np.zeros(shape=(3, len(F1)))

    for i in range(len(tau)):
        mat = np.array([0,  np.cos(a1), np.cos(a2); ...
                        1,  np.sin(a1), np.sin(a2); ...
                        ])
        tau[:,i] = 


    return 0