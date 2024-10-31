import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

def calculate_ssim_4D(images1, images2):
    """
    Calculate the SSIM score between two batches of images.
    images1, images2: Tensors of shape (N, 3, H, W) where H, W should be at least 11x11.
    """
    # Ensure images are moved to CPU and converted to NumPy arrays
    images1 = images1.permute(0, 2, 3, 1).cpu().numpy()  # Convert to NHWC for skimage
    images2 = images2.permute(0, 2, 3, 1).cpu().numpy()

    # Process each pair of images
    ssim_scores = []
    for img1, img2 in zip(images1, images2):
        # Ensure image is in float format with values ranging from 0 to 1

        # Compute SSIM
        score = ssim(img1, img2, multichannel=True, data_range=1,channel_axis=-1,  win_size=11)
        ssim_scores.append(score)

    # Return the average SSIM score
    return np.mean(ssim_scores)
