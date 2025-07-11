
# Remove background from image

### Step 1. Clone this repo 
```bash
git clone https://github.com/hashABCD/RemoveBGProject
cd RemoveBGProject
```

### Step 2. Activate virtual environment and install requirements
Activate virtual environment
```
python -m venv .venv        #create virtual environment

# activate virtual environment
source .venv/Scripts/activate       #Linux/MacOS 
.venv\Scripts\activate              #Windows
```

Install requirements
```
pip install -r requirements.txt
```

### Step 3. Download the pretrained model with weights
Download the trained model [*BiRefNet-general-epoch_244.pth*](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth) from this [git repo](https://github.com/ZhengPeng7/BiRefNet/releases) into assets folder

```bash
cd asset

wget https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth
#OR
curl -kLSs -O https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-epoch_244.pth
```
### Step 4. See the remove background function in action
```
cd ..
python main.py
```
This should show two images of Brad Pitt without background as shown in the example below.

### Step 5. Usage
How to use the function:
- Pillow Image input and output
- Numpy ndarray input and output 
### Step 5.1: Pillow Image input and output
*remove_background* function can be used as below
```python
from PIL import Image
from birefnet_rembg import remove_background

image = Image.open("path/to/image")     #convert image to PIL image
output_image = remove_background(image) # get image without background 
output_image.show()                     # view image without background
```
### Step 5.2: Numpy ndarray input and output
*remove_background_ndarray* function can be used as below
```python
import numpy as np
from PIL import Image
from birefnet_rembg import remove_background_ndarray


image = np.array(Image.open("path/to/image"))     #read image as ndarray
output_image = remove_background_ndarray(image) # get image without background as ndarray
Image.fromarray(output_image).show()             # view image without background
```
<hr><br>

### Example of usage

Example of usage is provided in *main.py* file
A png and a jpg image from input directory are converted into png images after removing background
 
Image type | Input Image  | Output Image |
|:---:|:---:|:---:|
| png |  ![input_image_2](input/brad-pitt-png.png) | ![output_image_2](output/brad-pitt-png.png) |
| jpeg | ![input_image_1](input/brad-pitt-jpeg.jpg) | ![output_image_1](output/brad-pitt-jpeg.png) |



