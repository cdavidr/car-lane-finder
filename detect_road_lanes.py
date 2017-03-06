import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from optparse import OptionParser
import time


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



# These lines are saved to be used in case of unreasonable slopes!
left_lines_global = [[0,0,0,0]]
right_lines_global = [[0,0,0,0]]

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    global left_lines_global
    global right_lines_global

    if lines is not None:

        left_lines = [[0,0,0,0]]
        right_lines = [[0,0,0,0]]
        num_left_lines = 0
        num_right_lines = 0


        # 1. Average left and right lines positions x1's, y1's, x2's, y2's
        for line in lines:
            for x1,y1,x2,y2 in line:
                # Left lines
                if ((y2-y1)/(x2-x1)) < 0:
                    left_lines += line
                    num_left_lines += 1
                # Right lines
                if ((y2-y1)/(x2-x1)) > 0:
                    right_lines += line
                    num_right_lines += 1


        left_divisor = np.full((1,4), num_left_lines, dtype=float)
        right_divisor = np.full((1,4), num_right_lines, dtype=float)

        left_lines = left_lines/left_divisor
        right_lines = right_lines/right_divisor

        # 2. Extrapolate to the top and bottom of the lane
        x1_left = left_lines[0][0]
        y1_left = left_lines[0][1]
        x2_left = left_lines[0][2]
        y2_left = left_lines[0][3]

        x1_right = right_lines[0][0]
        y1_right = right_lines[0][1]
        x2_right = right_lines[0][2]
        y2_right = right_lines[0][3]

        slope_left = (y2_left - y1_left)/(x2_left - x1_left)
        slope_right = (y2_right - y1_right)/(x2_right - x1_right)

        y1_left_new = img.shape[0]
        x1_left_new = x2_left - (y2_left - y1_left_new)/slope_left

        y1_right_new = img.shape[0]
        x1_right_new = x2_right - (y2_right - y1_right_new)/slope_right

        y2_left_new = (2/3)*img.shape[0]
        x2_left_new = x1_left_new + (y2_left_new - y1_left_new)/slope_left

        y2_right_new = (2/3)*img.shape[0]
        x2_right_new = x1_right_new + (y2_right_new - y1_right_new)/slope_right

        left_lines[0][0] = x1_left_new
        left_lines[0][1] = y1_left_new
        left_lines[0][2] = x2_left_new
        left_lines[0][3] = y2_left_new

        right_lines[0][0] = x1_right_new
        right_lines[0][1] = y1_right_new
        right_lines[0][2] = x2_right_new
        right_lines[0][3] = y2_right_new

        # Slope values should fall around [0.4-0.9]
        # Otherwise use the previous frame slope to predict the current frame line
        # Previous slope and lines are saved in global variables
        overall_left_slope = (-1*(y2_left_new-y1_left_new)/(x2_left_new-x1_left_new))
        overall_right_slope = ((y2_right_new-y1_right_new)/(x2_right_new-x1_right_new))


        # Each line goes from (x1,y1) to (x2,y2)
        # Check for unreasonable slopes to use the last 'good' line instead
        if overall_left_slope < 0.4 or overall_left_slope > 0.9 or \
        (np.isnan(left_lines[0][0]) == True or np.isnan(left_lines[0][1]) == \
        True or np.isnan(left_lines[0][2]) == True or np.isnan(left_lines[0][3]) == True):
            for x1,y1,x2,y2 in left_lines_global:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        else:
            for x1,y1,x2,y2 in left_lines:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                left_lines_global = left_lines


        if overall_right_slope < 0.4 or overall_right_slope > 0.9 or \
        (np.isnan(right_lines[0][0]) == True or np.isnan(right_lines[0][1]) == \
        True or np.isnan(right_lines[0][2]) == True or np.isnan(right_lines[0][3]) == True):
            for x1,y1,x2,y2 in right_lines_global:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        else:
            for x1,y1,x2,y2 in right_lines:
                if np.isnan(x1) == False and np.isnan(x2) == False and np.isnan(y1) == False and np.isnan(y2) == False:
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                    right_lines_global = right_lines

    if lines is None:
        for x1,y1,x2,y2 in left_lines_global:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        for x1,y1,x2,y2 in right_lines_global:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def gray_threshold_region(img, threshold):
    """
    `img` is a standard unprocessed image

    Returns a masked image, only keeping the grayscale value of the image above a certain threshold.
    """
    gray = grayscale(img)

    gray[gray < threshold] = 0

    return gray

def process_single_frame(image):
    #gray = grayscale(image)

    # Apply grayscale and black out values below the threshold
    # More effective than just applying grayscale
    gray = gray_threshold_region(image, 180)

    # Higher smoothing helps a lot to clean up noise in this data set
    blur = gaussian_blur(gray,13)

    # The high threshold should target the bright luminance of the lane lines
    # The low threshold should reject the darker colors, like the road by the lane
    edges = canny(blur,80,160)

    # Vertices created to hopefully scale with other image sizes
    vertices = np.array([[(10,image.shape[0]),
                      ((3/7)*image.shape[1], (2/3)*image.shape[0]),
                      ((5/7)*image.shape[1], (2/3)*image.shape[0]),
                      (image.shape[1]-10,image.shape[0])]],
                    dtype=np.int32)

    # Discard values outside region of interest
    mask = region_of_interest(edges, vertices)

    # Used to test videos output by seeing where hough_lines tries to create lines
    #edges3 = np.zeros_like(image)
    #edges3[:,:,0] = mask
    #edges3[:,:,1] = mask
    #edges3[:,:,2] = mask

    # Parameters: image, rho, theta, threshold, min_line_length, max_line_gap
    # Rho and theta can be increase to be more flexible of what makes up a line
    # Threshold is the minimum number of votes through a grid to be included in the output
    # min_line_length is the minimum number of pixels making up a line
    # max_line_gap is the maximum gap in pixels between connectable line segments
    lines = hough_lines(mask, 1, np.pi/180, 100, 70, 50)

    lines_transparency = weighted_img(lines, image, α=0.8, β=1., λ=0.)
    #plt.figure()
    #plt.imshow(lines_transparency)
    return lines_transparency

def process_image(input_file, output_file):
    # final output is saved to output_file
    result = process_single_frame(mpimg.imread(input_file))
    plt.imsave(output_file, result)

def process_video(input_file, output_file):
    # Given input_file video, save processed video to output_file
    video = VideoFileClip(input_file)
    result = video.fl_image(process_single_frame)
    result.write_videofile(output_file, audio=False)

# Script to run lane detection on image or video
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--input_file", dest="input_file",
                    help="Input video/image file")
    parser.add_option("-o", "--output_file", dest="output_file",
                    help="Output destination for video/image file")
    parser.add_option("-I", "--image_only",
                    action="store_true", dest="image_only", default=False,
                    help="Process image file containing road lines (defaults to video)")

    # Get and parse command line options
    options, args = parser.parse_args()
    input_file = options.input_file
    output_file = options.output_file
    image_only = options.image_only

    if image_only:
        process_image(input_file, output_file)
    else:
        start_time = time.time()
        process_video(input_file, output_file)
        print("--- %s seconds to process video ---" % (time.time() - start_time))
