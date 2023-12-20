import os
import glob
import numpy as np
from PIL import Image
from .inpainting import apply_gaussian_blur_to_bboxes, create_image_mask
import torch
from diffusers import StableDiffusionInpaintPipeline


def load_stable_diffusion_inpainting_pipeline():
    """
    Load the Stable Diffusion Inpainting Pipeline.

    This function initializes and returns the Stable Diffusion Inpainting Pipeline
    from Hugging Face's model hub. It is configured to use the 'runwayml/stable-diffusion-inpainting'
    model with 'fp16' revision for optimized performance.

    Returns:
    - pipe: StableDiffusionInpaintPipeline object, the loaded inpainting pipeline
    """
    device = "cuda"
    model_path = "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)

    return pipe

def generate_dataset(input_dir, output_dir, prompts_classes_dict, blur_strength=2.0, blur_size_percentage=0.05):
    """
    Generate an image dataset using inpainting based on provided prompts and classes.

    This function reads images from an input directory, performs inpainting based on a set of prompts,
    applies Gaussian blur post-inpainting, and saves the results in an output directory. Each inpainted
    image is saved with a corresponding bounding box file in YOLO format.

    Parameters:
    - input_dir: str, path to the directory containing input images
    - output_dir: str, path to the directory where the output dataset will be saved
    - prompts_classes_dict: dict, a dictionary where keys are prompts and values are corresponding class names
    - blur_strength: float, the sigma value for Gaussian blur, default is 2.0
    - blur_size_percentage: float, the width of the blur area around bounding boxes as a percentage of image dimensions, default is 0.05

    The function processes all compatible image files (jpg, jpeg, png) in the input directory.

    Example usage:
    prompts_classes_dict = {"Face of a yellow cat, high resolution, sitting on a park bench": "cat"}
    generate_dataset('path/to/input/dir', 'path/to/output/dir', prompts_classes_dict)
    """
    # Load the inpainting pipeline
    pipe = load_stable_diffusion_inpainting_pipeline()
    guidance_scale=7.5
    num_samples = 3
    generator = torch.Generator(device="cuda").manual_seed(1) # change the seed to get different results

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = glob.glob(input_dir + '/*.[pj][np][g]')  # jpg, png, jpeg
    for file_path in image_files:
        base_image = Image.open(file_path).convert("RGB").resize((512, 512))
        base_image_np = np.array(base_image)

        for serial, (prompt, class_name) in enumerate(prompts_classes_dict.items(), start=1):
            # Generate mask and perform inpainting
            mask, bboxes = create_image_mask(base_image_np, num_masks=1)  # Adjust parameters as needed
            mask_image = Image.fromarray(mask).convert("RGB").resize((512, 512))
            inpainted_image = pipe(prompt=prompt, 
                                   image=base_image,
                                   mask_image=mask_image,
                                   guidance_scale=guidance_scale,
                                   generator=generator,
                                   num_images_per_prompt=num_samples,
                                   ).images[0]

            # Save the image
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"{base_filename}_{class_name}_{format(serial, '04d')}.png"
            inpainted_image.save(os.path.join(output_dir, output_filename))

            # Save bounding box information
            bbox_filename = os.path.join(output_dir, f"{base_filename}_{class_name}_{format(serial, '04d')}.txt")
            with open(bbox_filename, 'w') as bbox_file:
                for bbox in bboxes:
                    # Convert bbox to YOLO format and write to file
                    yolo_bbox = ' '.join(map(str, [class_name, *bbox]))
                    bbox_file.write(yolo_bbox + '\n')
