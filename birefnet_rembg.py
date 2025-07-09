from PIL import Image
import torch
from torchvision import transforms

from models.birefnet import BiRefNet
from image_proc import refine_foreground
import numpy as np


############################# BiRefNet Set up ##########################################
PRETRAINED_MODEL_PATH = 'asset/BiRefNet-general-epoch_244.pth'

# Loading model and weights from local disk:
from utils import check_state_dict
birefnet = BiRefNet(bb_pretrained=False)
state_dict = torch.load(PRETRAINED_MODEL_PATH, map_location='cpu', weights_only=True)
state_dict = check_state_dict(state_dict)
birefnet.load_state_dict(state_dict)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

print("Device used : ", device)
torch.set_float32_matmul_precision(['high', 'highest'][0])
birefnet.to(device)
birefnet.eval()
print('BiRefNet is ready to use.')

# Input Data
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=[torch.float16, torch.bfloat16][0])
autocast_ctx = torch.amp.autocast(device_type='cuda', dtype=[torch.float32, torch.bfloat16][0])    # [INFO] no data error change 16 to 32, works with cuda & cpu

############################# BiRefNet Set up END ##########################################


def remove_background(image : Image, save: bool = False, output_file_path: str = None) -> Image:
    """
        input 
            image: Pillow Image [REQUIRED]
                Input image with background in pillow format
            save : boolean [OPTIONAL] 
                True : To save image without background 
                False(Default): To not save
            output_file_path : str [OPTIONAL]
                If save: True provide output file path to save image without background.
                Note: output should be in png format
        output: 
            Image without background in pillow format
    """

    image = image.convert("RGB") if image.mode != "RGB" else image
    input_images = transform_image(image).unsqueeze(0).to(device)

    # Prediction
    with autocast_ctx, torch.no_grad():
        print("prediction stated...")
        preds = birefnet(input_images)[-1].sigmoid().to(torch.float32).cpu() # will work with or without cuda
        # preds = birefnet(input_images)[-1].sigmoid().to(torch.float32).cuda()  # if cuda available
        print("prediction completed...")

    pred = preds[0].squeeze()

    # Generate mask
    pred_pil = transforms.ToPILImage()(pred)

    #apply mask
    image_masked = refine_foreground(image, pred_pil, device=device)
    image_masked.putalpha(pred_pil.resize(image.size)) 

    if save:
        image_masked.save(output_file_path)

    return image_masked

def remove_background_ndarray(image_np : np.ndarray) -> np.ndarray:
    """
    This function takes image in ndarray format with background and returns image in ndarray format without background.
    This is a wrapper around remove_background function which takes PIL image.

    Args:
        image (np.ndarray): Image with background in ndarray format

    Returns:
        np.ndarray: image without background in ndarray format
    """

    # Convert ndarray to PIL Image
    image = Image.fromarray(image_np)

    # Call remove_background function
    output_image = remove_background(image)
    
    # Convert PIL Image to ndarray
    output_image_np = np.array(output_image)

    #Return output image
    return output_image_np


