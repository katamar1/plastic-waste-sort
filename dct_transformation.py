#imports libraries needed
from numpy import asarray, zeros, uint8, clip
import cv2
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

#takes an image path and takes the given image and converts it to greyscale
requested = input("Image path?")
image = cv2.imread(requested)
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#converts the image to a numpy array
data = asarray(grey_image)

#partitions and applys DCT to the array
height, width = data.shape
block_size = 8
num_blocks_width = width // block_size
num_blocks_height = height // block_size

dct_coefficients_list = []

for column in range(0, num_blocks_height):
    for row in range(0, num_blocks_width):
        start_row = column * block_size
        end_row = (column + 1) * block_size
        start_col = row * block_size
        end_col = (row + 1) * block_size
        #builds the partitions

        block_pixel = data[start_row:end_row, start_col:end_col]
        block_pixel = block_pixel.astype(float)
        block_pixel -= block_pixel.mean()
        block_pixel = clip(block_pixel, 0, 255).astype(uint8)
        #applies dct

        dct_coefficients = dct(dct(block_pixel, axis=0), axis=1).astype(float)
        dct_coefficients_list.append(dct_coefficients)

reconstruction = zeros(data.shape, dtype = uint8)

for i in range(len(dct_coefficients_list)):
    block_row = i // num_blocks_width
    block_col = i % num_blocks_width
    start_row = block_row * block_size
    end_row = start_row + block_size
    start_col = block_col * block_size
    end_col = start_col + block_size
    #builds the image reconstruction

    block = idct(idct(dct_coefficients_list[i], axis=0), axis=1).astype(float)
    block += dct_coefficients_list[i].mean()
    block = clip(block, 0, 255).astype(uint8)
    reconstruction[start_row:end_row, start_col:end_col] = block

#displays the image
plt.imshow(reconstruction, cmap = "grey")
plt.title("DCT Image")
plt.show()