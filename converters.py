import torch
from torchvision.transforms import ToPILImage, ToTensor

def image_to_tensor(image):
  tensor = ToTensor()(image)
  return tensor.unsqueeze(0)

def tensor_to_image(tensor):
  tensor = tensor.squeeze()
  return ToPILImage()(tensor)

def turn_to_grayscale(image):
  return torch.transforms.v2.Grayscale()(image)
  
