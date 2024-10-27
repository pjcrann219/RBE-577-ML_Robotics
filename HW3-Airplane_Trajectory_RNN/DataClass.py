import torch

class DataClass:
    def __init__(self, data):
        #data.iloc[:,1] += 1000
        #data['isNew'] = data['X'] == 0
        #data['group_id'] = data['isNew'].cumsum() - 1
        self.data = data
        self.current_idx = 0
        self.valid_groups = sorted(data['group_id'].unique())
        self.n_runs = len(self.valid_groups)
        print(self.n_runs)
        self.max_seq_length = self.data.groupby('group_id').size().max()
        self.idx_to_group = dict(enumerate(self.valid_groups))
    
    def __len__(self):
        return len(self.data)
    
    def next(self):
        if self.current_idx >= self.n_runs:    ##This needs to be fixed the index should match
            return None, None, None, True
            
        input, traj, length = self.get_run_torch(self.current_idx)
        self.current_idx += 1
        done = (self.current_idx >= self.n_runs)
        
        return input, traj, length, done
    def get_run_torch(self, idx):
        group_id = self.idx_to_group[idx]
        run_data = self.data.loc[self.data['group_id'] == idx]
        input = torch.tensor(run_data[['XDelta', 'YDelta', 'HeadingDelta', 'GammaDelta']].iloc[0].values, dtype=torch.float)
        
        traj = torch.tensor(run_data[['X', 'Y', 'Z']].values, dtype=torch.float)
        seq_length = len(traj)
        
        padded_traj = torch.zeros((self.max_seq_length, 3), dtype=torch.float)
        padded_traj[:seq_length] = traj
        
        return input, padded_traj, seq_length
    
    def reset(self):
        self.current_idx = 0