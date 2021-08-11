from imageio import imread
import torch
from torch.utils.data import Dataset, DataLoader


class OmniglotDataset(Dataset):
    def __init__(self, dataset_info):
        self.dataset_info = dataset_info

        self.img_array, self.label_array = [], []

        with open(self.dataset_info, 'r') as f:
            for line in f.readlines():
                first_img_path, second_img_path, label = line.strip().split("\t")
                self.img_array.append((torch.Tensor(imread(first_img_path, as_gray=True)),
                                       torch.Tensor(imread(second_img_path, as_gray=True))))
                self.label_array.append(int(label))

    def __len__(self):
        return len(self.img_array)

    def __getitem__(self, idx):
        return self.img_array[idx], self.label_array[idx]
