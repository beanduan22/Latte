def prepare_data():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])


    dataset = datasets.FashionMNIST(
        root='./data',      
        train=True,         
        download=True,      
        transform=transform
    )


    train_size = int(0.8 * len(dataset))  
    val_size = len(dataset) - train_size  
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)


    torch.manual_seed(torch.initial_seed())

    return train_loader, val_loader
