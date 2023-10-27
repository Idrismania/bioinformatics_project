from torchvision import transforms

resize_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
