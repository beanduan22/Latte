def prepare_data():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Load ImageNet dataset from the specified path, using only the first 50 classes
    #full_dataset = datasets.ImageFolder(root='./data/ILSVRC2012_img_train/', transform=transform)
    full_dataset = datasets.ImageFolder(root='./data/ILSVRC2012_img_train/', transform=transform)

    # Filter out only the first 50 classes
    class_indices = {class_name: idx for idx, (class_name, _) in enumerate(full_dataset.class_to_idx.items()) if
                     idx < 50}
    filtered_indices = [i for i, (_, target) in enumerate(full_dataset.samples) if target in class_indices.values()]
    filtered_dataset = torch.utils.data.Subset(full_dataset, filtered_indices)

    # Split into train and test datasets
    train_size = int(0.8 * len(filtered_dataset))
    test_size = len(filtered_dataset) - train_size
    train_dataset, test_dataset = random_split(filtered_dataset, [train_size, test_size])

    # Create DataLoader for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader