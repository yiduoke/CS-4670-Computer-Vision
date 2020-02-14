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
  return np_array

#img = read_image("example.png")


def write_image(image, out_path):
  """Writes a numpy array as an image file.

  Args:
    image: Numpy array containing image to write
    out_path: Path for the output image
  """
  img = Image.fromarray(image)
  if (len(image.shape) < 3):
    img = img.convert("L")
  img.save(out_path)

#TODO: never tested display yet
def display_image(image):
  """Displays a grayscale image using matplotlib.

  Args:
    image: HxW Numpy array containing image to display.
  """
  plt.imshow(image)
  plt.show()


def convert_to_grayscale(image):
  """Convert an RGB image to grayscale.

  Args:
    image: HxWx3 uint8-type Numpy array containing the RGB image to convert.

  Returns:
    uint8-type Numpy array containing the image in grayscale
  """
  new_image = image[:,:,0] * 0.299 + image[:,:,1] * 0.587 + image[:,:,2] * 0.114
  return np.uint8(new_image)


# write_image(grey_rainbow, "grey_rainbow.png")

#TODO: delete all tests later

def convert_to_float(image):
  """Convert an image from 8-bit integer to 64-bit float format

  Args:
    image: Integer-valued numpy array with values in [0, 255]
  Returns:
    Float-valued numpy array with values in [0, 1]
  """
  new_image = image / 255.0
  return new_image

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
  (H,W) = image.shape
  (h,w) = kernel.shape
  h_half = int((h-1)/2)
  w_half = int((w-1)/2)
  padding_image = np.zeros([H+h-1, W+w-1])
  padding_image[h_half:(H+h_half),w_half:(W+w_half)] = image
  new_image = np.zeros((H,W))

  for i in range(h_half,(H+h_half)):
      i_new = i-h_half
      for j in range(w_half,(W+w_half)):
          j_new = j-w_half
          acc = 0
          for i_k in range(0,h):
              for j_k in range(0,w):
                  diff_i = i_k-h_half
                  diff_j = j_k-w_half
                  acc = acc + kernel[i_k,j_k]*padding_image[i-diff_i,j-diff_j]
          new_image[i_new,j_new] = acc
  return new_image




def gaussian_blur(image, ksize=3, sigma=1.0):
  """Blurs image by convolving it with a gaussian kernel.

  Args:
    image: HxW Numpy array, the grayscale image to blur
    ksize: size of the gaussian kernel
    sigma: variance for generating the gaussian kernel

  Returns:
    The blurred image
  """
  # all dues to stackoverflow
  kernel = np.fromfunction(lambda x, y: (1/(2*math.pi*sigma**2)) * math.e ** ((-1*((x-(ksize-1)/2)**2+(y-(ksize-1)/2)**2))/(2*sigma**2)), (ksize, ksize))
  kernel /= np.sum(kernel)
  return convolution(image, kernel)

# blurred_rainbow = gaussian_blur(grey_rainbow, 5, 3)
# write_image(blurred_rainbow, "blurred_rainbow.png")


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
  G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

  G_x_grad = convolution(image, G_x)
  G_y_grad = convolution(image, G_y)
  x_mag = np.power(G_x_grad, 2)
  y_mag = np.power(G_y_grad, 2)
  return np.power((x_mag + y_mag), 0.5)

# rose = read_image("rose.png")
# sobel_rose = sobel_filter(convert_to_grayscale(rose))
# write_image(sobel_rose, "sobel_rose.png")




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
  convolved1 = gaussian_blur(image, ksize1, sigma1)
  convolved2 = gaussian_blur(image, ksize2, sigma2)
  return convolved1 - convolved2

# example_dog = dog(img)
# write_image(example_dog, "example_dog.png")

def fourier_basis(image):
    """Computes the discrete fourier transform of image
        
    This function should return the same result as
    np.fft.fftshift(np.fft.fft2(image)). You may assume that
    image dimensions will always be even.
    
    Args:
    image: MXN numpy array, the grayscale image
    Returns:
    MXNXMXN complex Numpy array, the fourier basis image array
    """
    (M,N) = image.shape
    B = np.zeros([M,N,M,N],dtype = complex)
    for i_basis in range(-1*M/2, M/2):
        for j_basis in range(-1*N/2, N/2):
            k = i_basis
            l = j_basis
            for x in range(0,M):
                for y in range(0,N):
                    rad = 2*np.pi*k*x/M+2*np.pi*l*y/N
                    ele = complex(math.cos(rad),math.sin(rad))
                    B[i_basis,j_basis,x,y] = ele
    return B

def dft(image):
    
    """Computes the discrete fourier transform of image

        This function should return the same result as
        np.fft.fftshift(np.fft.fft2(image)). You may assume that
        image dimensions will always be even.

        Args:
        image: HxW Numpy array, the grayscale image
        Returns:
        HxW complex Numpy array, the fourier transform of the image
    """
    (M,N) = image.shape
    B = fourier_basis(image)
    F = np.zeros([M,N],dtype=complex)

    for x in range(M): # going down the rows of Fourier basis
        for y in range(N): #going down the columns of Fourier basis
            for m in range(M): # going down the rows of image
                for n in range(N): #going down the columns of image
                    F[x,y] += B[x,y,m,n] * image[m,n]
    return F

example_small = read_image("example_small.png")
small_dft = dft(example_small)
write_image(small_dft.real, "small_dft.png")

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
    (M,N) = ft_image.shape
    B = fourier_basis(ft_image)
    f = np.zeros([M,N],dtype=complex)

    for x in range(M): # going down the rows of Fourier basis
        for y in range(N): #going down the columns of Fourier basis
            for m in range(M): # going down the rows of image
                for n in range(N): #going down the columns of image
                    real = B[x,y,m,n].real # cos(-x) = cos(x)
                    imag = -1 * B[x,y,m,n].imag # sin(-x) = -sin(x)
                    combined = complex(real, imag)
                    f[x,y] += combined * ft_image[m,n]
    return f/M/N

small_back = idft(small_dft)
write_image(small_back.real, "small_back.png")

def visualize_kernels():
  """Visualizes your implemented kernels.

  This function should read example.png, convert it to grayscale and float-type,
  and run the functions gaussian_blur, sobel_filter, and dog over it. For each function,
  visualize the result and save it as example_{function_name}.png e.g. example_dog.png.
  This function does not need to return anything.
  """
  img = read_image("example.png")
  example_gb = gaussian_blur(img)
  write_image(example_gb, "example_gaussian_blur.png")
  example_sf = sobel_filter(img)
  write_image(example_sf, "example_sobel_filter.png")
  example_dog = dog(img)
  write_image(example_dog, "example_dog.png")


def visualize_dft():
  """Visualizes the discrete fourier transform.

  This function should read example.png, convert it to grayscale and float-type,
  and run dft on it. Try masking out parts of the fourier transform image and
  recovering the original image using idft. Can you create a blurry version
  of the original image? Visualize the blurry image and save it as example_blurry.png.
  This function does not need to return anything.
  """
  pass
