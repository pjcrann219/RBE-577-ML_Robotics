import torch
import random
import numpy as np

class DataClass:
    def __init__(self, data, split=.80):
        data.iloc[:,1] += 1000
        data['isNew'] = data['X'] == 0
        data['group_id'] = data['isNew'].cumsum() - 1
        self.data = data
        self.current_idx = 0
        self.n_runs = max(data['group_id'])
        self.max_seq_length = self.data.groupby('group_id').size().max()
        self.percent_split = split
        
        group_ids = self.data['group_id'].unique()
        self.training_group_ids = group_ids[np.random.choice(len(group_ids), int(len(group_ids) * self.percent_split), replace=False)]
        self.testing_group_ids = np.setdiff1d(group_ids, self.training_group_ids)

        print()
    def __len__(self):
        return len(self.data)
    
    def next(self):
        input, traj, length = self.get_run_torch(self.current_idx)
        done = self.iterate_idx()
        return input, traj, length, done

    def iterate_idx(self):
        self.current_idx += 1
        if self.current_idx >= self.n_runs:
            done = True
        else:
            done = False
        return done
    
    def get_run_torch(self, idx):
        run_data = self.data.loc[self.data['group_id'] == idx]
        input = torch.tensor(run_data[['XDelta', 'YDelta', 'HeadingDelta', 'GammaDelta']].iloc[0].values, dtype=torch.float)
        
        traj = torch.tensor(run_data[['X', 'Y', 'Z']].values, dtype=torch.float)
        seq_length = len(traj)
        
        padded_traj = torch.zeros((self.max_seq_length, 3), dtype=torch.float)
        padded_traj[:seq_length] = traj
        
        return input, padded_traj, seq_length
    
    def reset(self):
        self.current_idx = 0