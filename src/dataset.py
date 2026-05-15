from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.config import TRAIN_DIR,TEST_DIR,IMAGE_SIZE,BATCH_SIZE


def get_transforms(image_size=IMAGE_SIZE):
    train_transforms=transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.RandomResizedCrop((image_size,image_size), scale=(0.8, 1.0)),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    test_transforms=transforms.Compose(
        [
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ]
    )

    return train_transforms, test_transforms

def get_dataloaders(image_size=IMAGE_SIZE):
    train_transforms, test_transforms= get_transforms(image_size)

    train_dataset= datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transforms
    )

    test_dataset=datasets.ImageFolder(

        root=TEST_DIR,
        transform=test_transforms
    )

    train_loader= DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader= DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader,train_dataset.classes