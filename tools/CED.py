import torch

def calculate_euclidean_distance(vectors1, vectors2):
    """
    Calculate the Euclidean distance between two batches of vectors.
    vectors1, vectors2: Tensors of shape (N, D)
    """
    # Calculate Euclidean distance
    distances = torch.norm(vectors1 - vectors2, dim=1)

    # Return the average distance
    return torch.mean(distances).item()

# Example usage:
# vectors1 = torch.randn(64, 512)
# vectors2 = torch.randn(64, 512)
# average_distance = calculate_euclidean_distance(vectors1, vectors2)
