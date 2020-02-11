import math
import numpy as np
import PIL
from matplotlib import pyplot as plt
from PIL import Image


def read_image(image_path):
  """Read an image into a numpy array.

  Args:
    image_path: Path to the image file.

  Returns:
    Numpy array containing the image
  """
  np_array = np.asarray(PIL.Image.open(image_path))
  print(np_array)
  return np_array
 
img = read_image("example.png")


def write_image(image, out_path):
  """Writes a numpy array as an image file.
  
  Args:
    image: Numpy array containing image to write
    out_path: Path for the output image
  """
  img = Image.fromarray(image)
  img.save(out_path)

write_image(img,"testing.png")


def display_image(image):
  """Displays a grayscale image using matplotlib.

  Args:
    image: HxW Numpy array containing image to display.
  """
  pass


def convert_to_grayscale(image):
  """Convert an RGB image to grayscale.

  Args:
    image: HxWx3 uint8-type Numpy array containing the RGB image to convert.

  Returns:
    uint8-type Numpy array containing the image in grayscale
  """
  pass


def convert_to_float(image):
  """Convert an image from 8-bit integer to 64-bit float format

  Args:
    image: Integer-valued numpy array with values in [0, 255]
  Returns:
    Float-valued numpy array with values in [0, 1]
  """
  pass


def convolution(image, kernel):
  """Convolves image with kernel.

  The image should be zero-padded so that the input and output image sizes
  are equal.
  Args:
    image: HxW Numpy array, the grayscale image to convolve
    kernel: hxw numpy array
  Returns:
    image after performing convolution
  """
  pass


def gaussian_blur(image, ksize=3, sigma=1.0):
  """Blurs image by convolving it with a gaussian kernel.

  Args:
    image: HxW Numpy array, the grayscale image to blur
    ksize: size of the gaussian kernel
    sigma: variance for generating the gaussian kernel

  Returns:
    The blurred image
  """
  pass


def sobel_filter(image):
  """Detects image edges using the sobel filter.

  The sobel filter uses two kernels to compute the vertical and horizontal
  gradients of the image. The two kernels are:
  G_x = [-1 0 1]      G_y = [-1 -2 -1]
        [-2 0 2]            [ 0  0  0]
        [-1 0 1]            [ 1  2  1]
  
  After computing the two gradients, the image edges can be obtained by
  computing the gradient magnitude.

  Args:
    image: HxW Numpy array, the grayscale image
  Returns:
    HxW Numpy array from applying the sobel filter to image
  """
  pass
 

def dog(image, ksize1=5, sigma1=1.0, ksize2=9, sigma2=2.0):
  """Detects image edges using the difference of gaussians algorithm

  Args:
    image: HxW Numpy array, the grayscale image
    ksize1: size of the first gaussian kernel
    sigma1: variance of the first gaussian kernel
    ksize2: size of the second gaussian kernel
    sigma2: variance of the second gaussian kernel
  Returns:
    HxW Numpy array from applying difference of gaussians to image
  """
  pass


def dft(image):
  """Computes the discrete fourier transform of image

  This function should return the same result as
  np.fft.fftshift(np.fft.fft2(image)). You may assume that
  image dimensions will always be even.

  Args:
    image: HxW Numpy array, the grayscale image
  Returns:
    NxW complex Numpy array, the fourier transform of the image
  """
  pass


def idft(ft_image):
  """Computes the inverse discrete fourier transform of ft_image.

  For this assignment, the complex component of the output should be ignored.
  The returned array should NOT be complex. The real component should be
  the same result as np.fft.ifft2(np.fft.ifftshift(ft_image)). You
  may assume that image dimensions will always be even.

  Args:
    ft_image: HxW complex Numpy array, a fourier image
  Returns:
    NxW float Numpy array, the inverse fourier transform
  """
  pass


def visualize_kernels():
  """Visualizes your implemented kernels.

  This function should read example.png, convert it to grayscale and float-type,
  and run the functions gaussian_blur, sobel_filter, and dog over it. For each function,
  visualize the result and save it as example_{function_name}.png e.g. example_dog.png.
  This function does not need to return anything.
  """
  pass


def visualize_dft():
  """Visualizes the discrete fourier transform.

  This function should read example.png, convert it to grayscale and float-type,
  and run dft on it. Try masking out parts of the fourier transform image and
  recovering the original image using idft. Can you create a blurry version
  of the original image? Visualize the blurry image and save it as example_blurry.png.
  This function does not need to return anything.
  """
  pass

