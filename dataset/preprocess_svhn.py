def prepare_data():
    # Transformations: SVHN images are 32x32 RGB, so just convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # optional normalization
    ])

    # Load SVHN dataset (using the 'train' split)
    dataset = datasets.SVHN(
        root='./data',
        split='train',   # 'train' or 'test'
        download=True,
        transform=transform
    )

    # Split into training and validation sets (80/20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)

    # Reset the random seed if necessary elsewhere in your code
    torch.manual_seed(torch.initial_seed())

    return train_loader, val_loader
