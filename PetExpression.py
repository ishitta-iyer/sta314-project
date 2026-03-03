class PetExpressionTrain(Dataset):
    def __init__(self, paths, ctoi, transform=None):
        self.paths = paths
        self.ctoi = ctoi
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        y = self.ctoi[path.parent.name]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img, y
class PetExpressionTest(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img, path.name 