
![data-forge-logo](/static/logo_round.png)

# Dataset Forge
## Overview
This project is designed to create a diverse, labeled image dataset using Generative AI inpainting techniques, leveraging the Stable Diffusion Inpainting Pipeline. It includes Python functions to programatically create randomized image masks (for identifying regions for inpainting) and to automate the process of creating an object detection image dataset (YOLO Format) from a collection of base images and text prompts.
![example_outputs](/static/example_output_pic_labeled.png)

## Features
1. Inpainting of objects into images based on textual prompts.
2. Generation of masks for selective inpainting.
3. Automated creation of a labeled image dataset in YOLO bounding box format.

## Requirements
To run this project, you will need Python 3.10.x along with several dependencies listed in `requirements.txt`. It is recommended to use a virtual environment. Perform the Pytorch installation according to the official installation instructions for the version number listed in the requirements file and for your system configuration (https://pytorch.org/get-started/locally/). Then, install the remaining requirements using:

```bash
pip install -r requirements.txt
```

## Structure
```bash
dataset-forge/
│
├── src/                                 # Source code
│   ├── inpainting dataset_generator.py  # Main dataset generation script
│   ├── inpainting_config.yaml           # Default prompts/parameters 
│   └── inpainting.py                    # Inpainting functions
│
├── data/                                # Data directory (input/output)
│   ├── base_images/                     # Base images to inpaint objects into
│   └── output/                          # Output images and bounding box files
│
├── tools/                               # Tools to make datasets            
│   └── generate_inpaint.py              # Script to generate dataset
│
├── notebooks/                           # Jupyter notebooks for demos
│
├── requirements.txt                     # Project dependencies
├── LICENSE                              # License information
└── README.md                            # Project overview (this file)
```

## Usage
To use this project, follow these steps:

1. Prepare your dataset: Place your input images in `data/input`. These images should represent the environments you want to inpaint objects into.
2. Set your prompts: Edit the config file at 'src/inpainting_config.yaml' with your desired prompts and corresponding object classes for the dataset.
3. Run the script: Run the following from the project root directory. WARNING - This will download the specified model in the config file from Hugging Face on the first run. Ensure you have enough space.
```bash
tools/generate_inpaint.py
```
4. Check the results: The output images and YOLO formatted label text files will be saved in data/output/.

## Notebooks
The notebooks/ directory contains Jupyter notebooks for demonstration and testing. These notebooks provide examples of how to use the functions in this project.

## Known Issues
1. Sometimes the inpainting fails to generate any noticeable changes.
2. Bounding boxes are slightly loose (not tightly hugging the generated objects).
3. Doesn't work with all diffusion models on Hugging Face.

## License
The diffusion model weights are licensed under creativeml-openrail-m. 
The code is licensed under CC-BY-NC.

## Contact
For any queries or suggestions, please contact me here on Github.

## Citation
```
    @InProceedings{Rombach_2022_CVPR,
        author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
        title     = {High-Resolution Image Synthesis With Latent Diffusion Models},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2022},
        pages     = {10684-10695}
    }
```
