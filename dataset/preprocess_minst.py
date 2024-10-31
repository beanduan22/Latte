def prepare_data():

    # Transformations: Resize to 28x28 (as MNIST images are 28x28), convert to tensor, and normalize
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Load MNIST dataset
    dataset = datasets.MNIST(
        root='./data',  # Path where MNIST data will be downloaded
        train=True,  # Use the training data
        download=True,  # Download data if not already available
        transform=transform
    )

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)

    # Reset the random seed if necessary elsewhere in your code
    torch.manual_seed(torch.initial_seed())

    return train_loader, val_loader