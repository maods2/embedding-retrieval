from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Callable, Optional, Tuple, Any


class ImageDataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.dataset = CustomImageFolder(
            self.data_dir, transform=self.transform, target_transform=self._get_class_name)

        self.dataloader = DataLoader(self.dataset, shuffle=False)

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return iter(self.dataloader)

    def _get_class_name(self, index):
        return index


class CustomImageFolder(ImageFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):

        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
        )
        # self.paths = [s[0] for s in self.samples]
        # self.labels = [self.classes[s[1]]
        #                for s in self.samples]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path, self.classes[target]
