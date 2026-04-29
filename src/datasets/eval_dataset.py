import os
from PIL import Image
from torch.utils.data import Dataset

class EvalDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        files = sorted(os.listdir(root))

        self.images = []
        self.labels = []

        for idx, file in enumerate(files):

            img_path = os.path.join(root, file)

            label = idx // 2

            self.images.append(img_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label