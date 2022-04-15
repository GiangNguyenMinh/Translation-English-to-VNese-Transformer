from torch.utils.data import Dataset

class MyDatasets(Dataset):
    def __init__(self, en_data, vi_data):
        self.vi_data = vi_data
        self.en_data = en_data

    def __len__(self):
        return len(self.vi_data)

    def __getitem__(self, idx):
        return self.en_data[idx], self.vi_data[idx]