import torch
import torch.nn.functional as F


def calculate_cosine_similarity(vectors1, vectors2):
    """
    Calculate the cosine similarity score between two batches of vectors.
    vectors1, vectors2: Tensors of shape (N, D) where D is the dimension of the vectors.
    """
    # Normalize the vectors to unit length
    vectors1 = F.normalize(vectors1, p=2, dim=1)
    vectors2 = F.normalize(vectors2, p=2, dim=1)

    # Calculate cosine similarity (element-wise multiplication and sum)
    cosine_similarity = torch.sum(vectors1 * vectors2, dim=1)

    # Return the average cosine similarity score
    return torch.mean(cosine_similarity).item()

# Example usage:
# vectors1 = torch.randn(64, 512)
# vectors2 = torch.randn(64, 512)
# average_cosine_similarity = calculate_cosine_similarity(vectors1, vectors2)
