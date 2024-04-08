import torch
import torchvision.transforms as T

def image_to_tensor(image):
  tensor = T.ToTensor()(image)
  return tensor.unsqueeze(0)

def tensor_to_image(tensor):
  tensor = tensor.squeeze()
  return T.ToPILImage()(tensor)

def turn_to_grayscale(image):
  return T.v2.Grayscale()(image)
  
