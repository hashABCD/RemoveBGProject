Remove Background
- Function
    - accepts PIL image [Tested with png and jpg]
    - returns PIL image
    - save image option available check function docstring for more information

- Cuda support
  - Not available 
  - import tagged torch libraries to support cuda version
    - eg : pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    - latest cuda version 12.9 not supported by torch

- Weights from pth file in assets folder