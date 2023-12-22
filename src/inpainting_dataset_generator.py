import os
import glob
import yaml
import numpy as np
import random
import shutil
from PIL import Image
import torch
from diffusers import StableDiffusionInpaintPipeline
from .inpainting import create_image_mask

def load_config(config_file):
    """
    Load configuration from a YAML file.
    """
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def load_stable_diffusion_inpainting_pipeline(config):
    """
    Load the Stable Diffusion Inpainting Pipeline.
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        config['model_path'],
        torch_dtype=torch.float16,
    ).to(config['device'])
    return pipe

def process_image(file_path, pipe, config, prompts_classes_dict):
    """
    Process a single image with inpainting based on prompts.
    """
    base_image = Image.open(file_path).convert("RGB").resize(config['image_size'])
    base_image_np = np.array(base_image)
    generator = torch.Generator(device=config['device']).manual_seed(random.randrange(1000000))

    for serial, (prompt, class_name) in enumerate(prompts_classes_dict.items(), start=1):
        mask, bboxes = create_image_mask(base_image_np, num_masks=1)
        mask_image = Image.fromarray(mask).convert("RGB").resize(config['image_size'])
        prompt = '(' + prompt + ')'    # This will add emphasis to the user's prompt
        enhanced_prompt = prompt + ", " + config['prompt_enhancement']  # This adds additional enhancement words to the prompt
        inpainted_image = pipe(prompt=enhanced_prompt,
                               negative_prompt=config['negative_prompt'], 
                               image=base_image,
                               mask_image=mask_image,
                               guidance_scale=config['guidance_scale'],
                               generator=generator,
                               num_images_per_prompt=config['num_samples'],
                               ).images[0]
        yield inpainted_image, bboxes, file_path, class_name, serial

def save_results(inpainted_image, bboxes, file_path, class_name, serial, output_dir):
    """
    Save the inpainted image and bounding box information.
    """
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    output_filename = f"{base_filename}_{class_name}_{format(serial, '04d')}.png"
    inpainted_image.save(os.path.join(output_dir, output_filename))

    bbox_filename = os.path.join(output_dir, f"{base_filename}_{class_name}_{format(serial, '04d')}.txt")
    with open(bbox_filename, 'w') as bbox_file:
        for bbox in bboxes:
            yolo_bbox = ' '.join(map(str, [class_name, *bbox]))
            bbox_file.write(yolo_bbox + '\n')

def generate_dataset(config_path):
    """
    Generate an image dataset using inpainting based on provided prompts and classes in the config file.
    """
    config = load_config(config_path)
    pipe = load_stable_diffusion_inpainting_pipeline(config)
    input_dir = config['input_directory']
    output_dir = config['output_directory']
    prompts_classes_dict = config['prompts_classes_dict']

    if not os.path.exists(output_dir):
        output_config_path = output_dir + "/config"
        os.makedirs(output_dir)
        os.makedirs(output_config_path)
        shutil.copy(config_path, output_config_path + '/' + os.path.basename(config_path))

    image_files = glob.glob(input_dir + '/*.[pj][np][g]')  # jpg, png, jpeg
    for file_path in image_files:
        for result in process_image(file_path, pipe, config, prompts_classes_dict):
            save_results(*result, output_dir)
