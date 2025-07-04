
# Remove background from image

### Step 1. Clone this repo 
```bash
git clone https://github.com/hashABCD/RemoveBGProject
cd /RemoveBGProject
```

### Step 2. Install requirements
```
pip install -r requirements.txt
```
*cuda* is not supported in the current configuration view *Notes* for more information

### Step 3. Download the pretrained model with weights
Download the trained model [*BiRefNet-general-epoch_244.pth*](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-resolution_512x512-fp16-epoch_216.pth) from this [git repo](https://github.com/ZhengPeng7/BiRefNet/releases) into assets folder

```bash
cd assets
wget https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-resolution_512x512-fp16-epoch_216.pth

```

### Step 4. Remove background
*remove_background* function can be used as follows
```python
from PIL import Image
from birefnet_rembg import remove_background

image = Image.open("path/to/image")     #convert image to PIL image
output_image = remove_background(image) # get image without background 
output_image.show()                     # view image without background
```
===================================================================================================================
### Example
Example of usage is provided in *main.py* file
A png and a jpg image from input directory are converted into png images after removing background
 
Image type | Input Image  | Output Image |
|:---:|:---:|:---:|
| png |  ![input_image_2](input/brad-pitt-png.png) | ![output_image_2](output/brad-pitt-png.png) |
| jpeg | ![input_image_1](input/brad-pitt-jpeg.jpg) | ![output_image_1](output/brad-pitt-jpeg.png) |



