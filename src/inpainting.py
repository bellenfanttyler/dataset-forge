import numpy as np
import random
import cv2

def create_image_mask(base_image, min_mask_size=0.1, max_mask_size=0.9, aspect_ratio=1.0, fixed_size=None, num_masks=1):
    """
    Create image masks for inpainting on a base image.

    Parameters:
    - base_image: numpy array (base image)
    - min_mask_size: float, minimum mask size as a percentage of the base image area (default 10%)
    - max_mask_size: float, maximum mask size as a percentage of the base image area (default 90%)
    - aspect_ratio: float, aspect ratio of the mask (width:height), default 1.0 (square)
    - fixed_size: tuple(int, int) or None, fixed size of the mask (width, height), default None
    - num_masks: int, number of masks to create

    Returns:
    - mask_image: numpy array, image with masks (white for inpainting, black for keeping as is)
    - mask_bounding_boxes: list of tuples, bounding box locations in YOLO format


    Example usage
    base_image = cv2.imread('path_to_image.jpg')  # Example, replace with actual image loading
    mask, bboxes = create_image_mask(base_image, num_masks=2)
    """

    h, w = base_image.shape[:2]
    base_area = w * h
    mask_image = np.zeros((h, w), dtype=np.uint8)
    mask_bounding_boxes = []

    for _ in range(num_masks):
        # Determine mask size
        if fixed_size:
            mask_w, mask_h = fixed_size
        else:
            mask_area = random.uniform(min_mask_size * base_area, max_mask_size * base_area)
            mask_w = int(np.sqrt(mask_area * aspect_ratio))
            mask_h = int(mask_w / aspect_ratio)
            # Adjust if the size is too large
            mask_w, mask_h = min(mask_w, w), min(mask_h, h)

        # Random position such that mask fits inside the image
        x1 = random.randint(0, w - mask_w)
        y1 = random.randint(0, h - mask_h)
        x2, y2 = x1 + mask_w, y1 + mask_h

        # Create mask
        mask_image[y1:y2, x1:x2] = 255

        # Calculate YOLO format bounding box
        bbox_x_center = (x1 + x2) / 2 / w
        bbox_y_center = (y1 + y2) / 2 / h
        bbox_width = mask_w / w
        bbox_height = mask_h / h
        mask_bounding_boxes.append((bbox_x_center, bbox_y_center, bbox_width, bbox_height))

    return mask_image, mask_bounding_boxes


def apply_gaussian_blur_to_bboxes(base_image, bboxes, blur_strength=2.0, blur_size_percentage=0.05):
    """
    Apply Gaussian blur around the edges of bounding boxes on the inpainted image.

    Parameters:
    - base_image: numpy array, the base image
    - bboxes: list of tuples, bounding boxes in YOLO format (center x, center y, width, height)
    - blur_strength: float, sigma for the Gaussian blur
    - blur_size_percentage: float, width of the blur area around each bounding box as a percentage of image dimensions

    Returns:
    - blurred_image: numpy array, the image with Gaussian blur applied around the bounding boxes

    # Example usage
    base_image = cv2.imread('path_to_image.jpg')  # Example, replace with actual image loading
    mask, bboxes = create_image_mask(base_image, num_masks=2)
    generated_image = = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
    processed_image = apply_gaussian_blur_to_bboxes(generated_image, bboxes, blur_strength=2, blur_size_percentage=0.05)

    TODO: Update this so it doesn't fill in the whole boudning box, just the edges. Also, catch error when
    the bounding box is close, or at, the edge and it tries to apply blur outside of image.
    """

    h, w = base_image.shape[:2]
    blur_size = int(max(h, w) * blur_size_percentage)

    # Create a mask for the areas to blur
    blur_mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in bboxes:
        # Convert YOLO format to rectangle coordinates
        x_center, y_center, bbox_w, bbox_h = bbox
        x_center *= w
        y_center *= h
        bbox_w *= w
        bbox_h *= h
        x1 = int(x_center - bbox_w / 2) - blur_size
        y1 = int(y_center - bbox_h / 2) - blur_size
        x2 = int(x_center + bbox_w / 2) + blur_size
        y2 = int(y_center + bbox_h / 2) + blur_size

        # Ensure coordinates are within image boundaries
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        # Set the mask for this bounding box
        blur_mask[y1:y2, x1:x2] = 255

    # Apply Gaussian blur on the masked areas
    blurred_image = base_image.copy()
    for y in range(h):
        for x in range(w):
            if blur_mask[y, x] == 255:
                x1 = max(x - blur_size, 0)
                y1 = max(y - blur_size, 0)
                x2 = min(x + blur_size, w - 1)
                y2 = min(y + blur_size, h - 1)
                region = base_image[y1:y2, x1:x2]
                blurred_image[y, x] = cv2.GaussianBlur(region, (0, 0), blur_strength)[y - y1, x - x1]

    return blurred_image



