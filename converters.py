from PIL import Image
import numpy as np
import torch
from torchvision.transforms import v2

def generate_noise(image, upscaling_factor, mean = 0, std_dev = 0.2):
  img_size = np.array(image).shape
  # next line taken from : https://stackoverflow.com/questions/1781970/multiplying-a-tuple-by-a-scalar
  img_size = tuple(i * upscaling_factor for i in img_size)
  noise = np.random.normal(mean, std_dev, img_size)
  #noise = (noise-np.min(noise))/(np.max(noise)-np.min(noise))
  return Image.fromarray(noise)

#this function turns an image to a tensor with values betwwen -1 and 1 (since edge net is trained for such tensors and I just followed their convention)
def image_to_tensor(image):
  tensor = v2.PILToTensor()(image)
  tensor = (2.0*tensor-(torch.min(tensor)+torch.max(tensor)))/(torch.max(tensor)-torch.min(tensor))
  #tensor = v2.Normalize([0.5], [0.5])(tensor)
  return tensor.unsqueeze(0)

#this function makes the tensor to be first in the range between 0 and 1 and then converts it to an image
def tensor_to_image(tensor):
  tensor = tensor.squeeze()
  tensor = (tensor + 1.0) / 2.0
  return v2.ToPILImage()(tensor)

def turn_to_grayscale(image):
  return v2.Grayscale()(image)
