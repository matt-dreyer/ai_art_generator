from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from PIL import Image, ImageFont, ImageDraw
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from os.path import isfile, join
import numpy as np
import cv2
import argparse
import glob


# Define the directory to store the generated images
output_dir = "generated-images"

#this function will generate images using models sourced the "local-models" folder
def generate_local_model_images(prompt, watermarks):
    local_model_dir = "./local-models"
    model_files = glob.glob(os.path.join(local_model_dir, "*.safetensors"))

    # If no local models, return
    if not model_files:
        print("No local models found.")
        return

    collage_list = []
    for model_file in model_files:
        model_file = model_file.replace("\\", "/")
        model_name = model_file.split("/")[2]

        print("\n*** Generating image with local model " + model_file + "\n")

        # create the image
        pipe = StableDiffusionPipeline.from_single_file(model_file, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")
        image = pipe(prompt).images[0]


        # Add watermark
        if watermarks:
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("arial.ttf", 14)
            watermark_text = f"{model_name}"
            draw.text((11, 11), watermark_text, fill=(0, 0, 0, 128), font=font)  # Shadow
            draw.text((10, 10), watermark_text, fill=(255, 255, 255, 128), font=font)  # Text

        # Save the image
        image_name = os.path.join("generated-images", "genart-" + f"{model_name}" + ".png")
        image.save(image_name)

        collage_list.append(image_name)
        #os.startfile(image_name)

    return collage_list


#this function will generate images using models sourced from Huggingface
#models will be automatically downloaded from the Huggingface repo and cached locally
def generate_image(model_id, prompt, watermarks):
    print("\n*** Generating image with model " + model_id + "\n")
    model_name = model_id.split("/")

    #create the image
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    image = pipe(prompt).images[0]

    # Add watermark
    if watermarks:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", 14)
        watermark_text = f"{model_name[1]}"
        draw.text((11, 11), watermark_text, fill=(0, 0, 0, 128), font=font)  # Shadow
        draw.text((10, 10), watermark_text, fill=(255, 255, 255, 128), font=font)  # Text

        # save the image
        image_name = "genart-" + model_name[1] + ".png"
        image.save(os.path.join(output_dir, image_name))

    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # save the image
    image_name = "genart-" + model_name[1] + ".png"
    image.save(os.path.join(output_dir, image_name))

    #open the image we just created (useful for some debugging scenarios)
    #os.startfile(os.path.join(output_dir, image_name))

    return os.path.join(output_dir, image_name)

#this function will generate a collage using all images in a directory
def create_collage():
    print("\n*** Creating a mosaic")

    # Get all png images in 'generated-images' folder
    image_dir = "generated-images"
    collage_list = glob.glob(os.path.join(image_dir, "*.png"))

    if not collage_list:
        print("No images found for mosaic.")
        return

    # Calculate the size of the collage
    collage_size = int(np.ceil(np.sqrt(len(collage_list))))

    # Resize all the images to the same size
    src_images = [cv2.imread(file) for file in collage_list]
    resized_images = [cv2.resize(img, (800, 800)) for img in src_images]

    # Add blank images if necessary
    total_images_required = collage_size ** 2
    blank_images_required = total_images_required - len(resized_images)
    blank_image = np.zeros((800, 800, 3), np.uint8)  # Create a blank black image
    for _ in range(blank_images_required):
        resized_images.append(blank_image)

    # Convert list to numpy array
    resized_images = np.asarray(resized_images)

    # Split the list into rows
    rows = [resized_images[i * collage_size:(i + 1) * collage_size] for i in range(collage_size)]

    # Stack images horizontally and vertically
    collage_rows = [np.hstack(row) for row in rows]
    final_collage = np.vstack(collage_rows)

    cv2.imwrite(image_dir + "/mosaic.png", final_collage)
    #os.startfile(os.path.join(output_dir, "mosaic.png"))

#create an html page that displays all of the images
def create_html(prompt):

    print("\n*** Creating a HTML gallery")

    output_dir = "generated-images"

    # get all the image paths
    images = [f for f in os.listdir(output_dir) if isfile(join(output_dir, f)) and f.endswith(".png")]

    # create html string
    html_str = """
    <!doctype html>
    <html>
        <head>
            <title>Generated Art</title>
            <style>
                body {
                    background-color: #f0f0f0;
                    text-align: center;
                }
                img {
                    margin: 10px;
                    box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
                    transition: 0.3s;
                    width: 400px;
                }
                img:hover {
                    box-shadow: 0 8px 16px 0 rgba(0, 0, 0, 0.2);
                }
            </style>
        </head>
        <body>
            <h1>AI Generated Art</h1>
    """

    html_str += f'<p><i>{prompt}</i></p>\n'

    # insert each image into the html
    for image in images:
        if image == "mosaic.png":
            html_str += f'<br><hr>Collage<br><a href="{image}"><img src="{image}" alt="{image}"></a>\n'
        else:
            html_str += f'<a href="{image}"><img src="{image}" alt="{image}"></a>\n'

    # close the tags
    html_str += """
        </body>
    </html>
    """

    # write the html string to a file
    with open(output_dir + "/index.html", "w") as file:
        file.write(html_str)

    # open the file in the default web browser
    os.startfile(os.path.join(output_dir, "index.html"))


########################################################################
def main():


    #Pick a rendering model
    model_list = [
                  "plasmo/vox2",
                  "dreamlike-art/dreamlike-photoreal-2.0",
                  "diffusersfan/fking_civitai",
                  "Lykon/DreamShaper",
                  "prompthero/openjourney",
                  "nitrosocke/Ghibli-Diffusion",
                  "SG161222/Realistic_Vision_V1.4",
                  "gsdf/Counterfeit-V2.5",
                  "CompVis/stable-diffusion-v1-4",
                  "runwayml/stable-diffusion-v1-5",
                  "stabilityai/stable-diffusion-2-1"
                  ]

    # Command line arguments parsing
    parser = argparse.ArgumentParser(description='Generate AI Art.')
    parser.add_argument('--no-watermark', action='store_true', help='Disable watermarking on the images')
    parser.add_argument('--no-collage', action='store_true', help='Disable creation of collage')
    args = parser.parse_args()


    # Try to read the prompt from "prompt.txt", else ask user for input
    try:
        with open("prompt.txt", "r") as file:
            lines = file.readlines()
        prompt = "".join(line for line in lines if not line.strip().startswith("#"))
        if not prompt:
            prompt = input("\nDescribe the image I should create for you: ")
    except (FileNotFoundError, IOError):
        prompt = input("\nDescribe the image I should create for you: ")

    print("\nGenerating images with the following prompt:\n\n" + prompt)


    #first we will generate images using the locally stored models
    generate_local_model_images(prompt, not args.no_watermark)

    #next we will generate images using the models from huggingface
    for model_id in model_list:
        image_name = generate_image(model_id, prompt, not args.no_watermark)


    # Generate Collage image that aggregates all images into one jumbo image (useful for sharing)
    if not args.no_collage:
        create_collage()

    #Generage HTML Page that lists all images on one page (useful for viewing locally)
    create_html(prompt)

if __name__ == "__main__":
    main()