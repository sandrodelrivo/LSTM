import torch
import torch.utils.data as data

class Data(data.Dataset):

    def __init__(self, bit_len=50, sample_size=100000,):
        self.bit_len = bit_len
        self.sample_size = sample_size
        self.features, self.labels = self.generate_fixed_length(sample_size, bit_len)

    def __getitem__(self, index):
        return self.features[index, :], self.labels[index]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def generate_fixed_length(sample_size=100000, sequence_length=50):

        strings = torch.randint(2, size=(sample_size, sequence_length, 1)).float()

        sums = strings.cumsum(axis=1)

        parity = (sums % 2 == 0).float() # even parity

        return strings, parity


