# AI Art Generator

This script generates AI-based artwork based on a text prompt. It uses various models from both local sources and Hugging Face's model hub. The script can generate a collage of all created images, and also creates an HTML page to display all the images.



### Requirements
- NVIDIA GPU and the required CUDA drivers installed, see https://developer.nvidia.com/cuda-downloads
- Python 3.x

To install the required libraries, navigate to the directory where the `requirements.txt` is located and run this command in your shell:

## Text-to-Image Models
By default the script is configured to automatically download and use several models hosted by Hugging Face. The user has the option to manually download and use .safetensor models that can be found on various websites.

Models included by default include:
 - plasmo/vox2
 - dreamlike-art/dreamlike-photoreal-2.0
 - diffusersfan/fking_civitai
 - Lykon/DreamShaper
 - prompthero/openjourney
 - nitrosocke/Ghibli-Diffusion
 - SG161222/Realistic_Vision_V1.4
 - gsdf/Counterfeit-V2.5
 - CompVis/stable-diffusion-v1-4
 - runwayml/stable-diffusion-v1-5
 - stabilityai/stable-diffusion-2-1

## Installation

```bash
pip install -r requirements.txt
```


### Usage

```bash
python art_generator.py [--no-watermark] [--no-collage]
```

- `--no-watermark` : Use this flag if you want to disable watermarking on the generated images.
- `--no-collage` : Use this flag if you want to disable the creation of a collage from the generated images.

## Image Prompt

The program will first attempt to read the contents of a text file named "prompt.txt" and use that for the image prompt. If that file does not exist, or the content is not plain text, it will prompt the user for input. Any line in "prompt.txt" starting with a "#" will be ignored.

## Output

The generated images will be stored in a folder named "generated-images". If the folder does not exist, it will be created. The collage, if created, will also be stored in the same folder.

## License

This project is licensed under the terms of the MIT license.
