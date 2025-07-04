from PIL import Image
from birefnet_rembg import remove_background
import os

# Set up input files and output file
INPUT_DIR = "input/"
OUTPUT_DIR = "output/"
input_files = ["brad-pitt-jpeg.jpg", "brad-pitt-png.png"]
input_file_paths = [os.path.join(INPUT_DIR, file) for file in input_files]
output_file_paths = [os.path.join(OUTPUT_DIR, file.replace(".jpg", ".png")) for file in input_files]

for input_file_path in input_file_paths:
        image = Image.open(input_file_path)         # read input file
        
        output_image = remove_background(image)     # remove background
        
        # remove background and save output in specified path
        # output_image = remove_background(image, save=True, output_file_path=output_file_paths[input_file_paths.index(input_file_path)])     
        
        output_image.show()