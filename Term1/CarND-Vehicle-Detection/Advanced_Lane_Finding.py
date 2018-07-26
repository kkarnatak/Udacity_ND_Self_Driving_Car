import cv2
import numpy as np
from collections import deque
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define constants here that will be used in the video_pipeline. Move this later to py file

_x, _y = 360, 258
offset_1 = 48
offset_2 = 2

src = np.float32([[int(_x - offset_1), _y],
                  [int(_x + offset_1), _y],
                  [int(0 + offset_2), 390],
                  [int(720 - offset_2), 390]])
dst = np.float32([[0, 0], [720, 0], [0, 405], [720, 405]])

# B-Channel threshold values
b_thresh_min = 145
b_thresh_max = 200

# L-Channel threshold values
l_thresh_min = 215
l_thresh_max = 255

# S-Channel threshold values
s_thresh_min = 180
s_thresh_max = 255

abs_sobel_threshold_value = (10, 230)
mag_threshold_value = (30, 150)
threshold_range_value = (0.7, 1.3)
ksize = 7



#
# Scale 1280 to 720
scale = 720/1280

nx = 9
ny = 6

# Create the object points

objp = np.zeros((ny*nx,3), np.float32)
objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

# Store the points
obj_points = [] # World space (3D)
img_points = [] # Image plane (2D)



def calibrate_camera():
    # Inside corners in the object
    nx = 9
    ny = 6

    # Prepare the object points (3D for e.g. (0,0,0)..(2,0,0) etc)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Read & iterate  all the calidration imagees
    calibration_images = glob.glob('camera_cal/calibration*.jpg')

    for file in calibration_images:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        is_present, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if is_present == True:
            obj_points.append(objp)
            img_points.append(corners)

# This function takes an image, gradient orientation
# and threshold min / max values

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Use OpenCV Sobel() function using the absolute value
    # to apply gradient x and y
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    binary_output = np.zeros_like(scaled_sobel)

    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


# This function will explore the direction, or orientation, of the gradient.

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


# This function returns the magnitude of the gradient
# for a given sobel kernel size and threshold values

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # X & Y sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Compute the gradient magnitude and rescale
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)

    # Binary image: 1 where thresholds are met, otherwise 0
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output


# This class contains methods which will be used for fitting a polynomial in the lane pixels obtained from the
# binary threshold images

class PolyfitUtils:
    def __init__(self):

        # Check for previous frame found
        self.isFound = False

        # X and Y values
        self.X = None
        self.Y = None

        # Store recent x intercepts for averaging across frames
        self.x_cap = deque(maxlen=15)
        self.top = deque(maxlen=15)

        # Remember last x intercept
        self.lastx_cap = None
        self.last_top = None

        # Remember radius of curvature
        self.radius = None

        # Store recent polynomial coefficients for averaging across frames
        self.coeff0 = deque(maxlen=15)
        self.coeff1 = deque(maxlen=15)
        self.coeff2 = deque(maxlen=15)
        self.coeff_x = None
        self.pts = []

    def heuristic_search(self, x, y, image):
        '''
        The function is invoked when lane pixels are not found in the previous frames. It uses sliding window approach.
        '''
        x_vals = []
        y_vals = []

        if self.isFound == False:
            i = 720
            j = 405
            while j >= 0:
                histogram = np.sum(image[j:i, :], axis=0)
                if self == rightLane:
                    peak = np.argmax(histogram[405:]) + 405
                else:
                    peak = np.argmax(histogram[:405])
                x_idx = np.where((((peak - 30) < x) & (x < (peak + 30)) & ((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    x_vals.extend(x_window)
                    y_vals.extend(y_window)
                i -= 90
                j -= 90
        if np.sum(x_vals) > 0:
            self.isFound = True
        else:
            y_vals = self.Y
            x_vals = self.X

        return x_vals, y_vals, self.isFound

    def search_lane(self, x, y):
        '''
        This function is invoked when lane pixels are found in the previous frames. It checks in the range of 35 pixels.
        '''
        x_vals = []
        y_vals = []

        if self.isFound == True:
            i = 720
            j = 405
            while j >= 0:
                y_val = np.mean([i, j])
                x_val = (np.mean(self.coeff0)) * y_val ** 2 + (np.mean(self.coeff1)) * y_val + (np.mean(self.coeff2))
                x_idx = np.where((((x_val - 30) < x) & (x < (x_val + 30)) & ((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    np.append(x_vals, x_window)
                    np.append(y_vals, y_window)
                i -= 90
                j -= 90
        if np.sum(x_vals) == 0:
            self.isFound = False  # If no lane pixels were detected then perform blind search
        return x_vals, y_vals, self.isFound

    def find_radius(self, x_vals, y_vals):

        # Compute the radius of the curvature
        # ym_per_pix = 30/405
        # xm_per_pix = 3.7/720
        ym_per_pix = 30 / 720
        xm_per_pix = 3.7 / 700

        fit_cr = np.polyfit(y_vals * ym_per_pix, x_vals * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * np.max(y_vals) + fit_cr[1]) ** 2) ** 1.5) \
                   / np.absolute(2 * fit_cr[0])
        return curverad

    def apply_sort(self, x_vals, y_vals):
        # Sort the values using numpy
        sorted_index = np.argsort(y_vals)
        sorted_y_vals = y_vals[sorted_index]
        sorted_x_vals = x_vals[sorted_index]
        return sorted_x_vals, sorted_y_vals

    def get_intercepts(self, polynomial):
        bottom = polynomial[0] * 720 ** 2 + polynomial[1] * 720 + polynomial[2]
        top = polynomial[0] * 0 ** 2 + polynomial[1] * 0 + polynomial[2]
        return bottom, top


calibrate_camera()
# Initialize the objects of the PolyfitUtils class
leftLane = PolyfitUtils()
rightLane = PolyfitUtils()


# Pipeline to process the test videos

def video_pipeline(image):
    # resize the image to the size which was used for calibration
    # scale is used from the 2nd cell on this notebook

    rows, cols, chs = image.shape
    image = cv2.resize(image, (int(cols * scale), int(rows * scale)))
    img_size = (image.shape[1], image.shape[0])

    # Calibrate camera and undistort image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size, None, None)
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # Perform perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(undist, M, img_size)
    grad_x = abs_sobel_thresh(warped, orient='x', sobel_kernel=3, thresh=abs_sobel_threshold_value)
    grad_y = abs_sobel_thresh(warped, orient='y', sobel_kernel=3, thresh=abs_sobel_threshold_value)
    # TODO: Not used, need to investigate and use the resulting binary images properly
    # Comment to self: The binary image combination is slowing down the performance
    # Need to optimize the process

    # create binary thresholded images using B and L channel
    b_channel = cv2.cvtColor(warped, cv2.COLOR_RGB2Lab)[:, :, 2]
    l_channel = cv2.cvtColor(warped, cv2.COLOR_RGB2LUV)[:, :, 0]

    # Threshold color channel
    s_channel = cv2.cvtColor(warped, cv2.COLOR_BGR2HLS)[:, :, 2]

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel > b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel > l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    combined_binary = np.zeros_like(b_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1) | (s_binary == 1) |
                    ((grad_x == 1) & (grad_y == 1))] = 1

    # combined_binary[(l_binary == 1) | (b_binary == 1) | (s_binary == 1)
    # | ((grad_x == 1) & (grad_y == 1)) | ((magnitude_binary == 1)
    # & (threshold_range_binary == 1))] = 1

    # Get the valid pixels from the binary image
    x, y = np.nonzero(np.transpose(combined_binary))

    # If previous frame exists
    if leftLane.isFound == True:
        leftx, lefty, leftLane.found = leftLane.search_lane(x, y)
    if rightLane.isFound == True:
        rightx, righty, rightLane.found = rightLane.search_lane(x, y)

    # Otherwise..
    if rightLane.isFound == False:
        rightx, righty, rightLane.found = rightLane.heuristic_search(x, y, combined_binary)
    if leftLane.isFound == False:
        leftx, lefty, leftLane.found = leftLane.heuristic_search(x, y, combined_binary)

    # Initialize left and right x,y arrays
    lefty = np.array(lefty).astype(np.float32)
    leftx = np.array(leftx).astype(np.float32)
    righty = np.array(righty).astype(np.float32)
    rightx = np.array(rightx).astype(np.float32)

    # Fit polynomial on pixels detected on the left
    left_fit = np.polyfit(lefty, leftx, 2)

    # Get top and bottom intercept
    leftx_cap, left_top = leftLane.get_intercepts(left_fit)

    # Extend the area using intercept and then compute mean
    leftLane.x_cap.append(leftx_cap)
    leftLane.top.append(left_top)
    leftx_cap = np.mean(leftLane.x_cap)
    left_top = np.mean(leftLane.top)
    leftLane.lastx_cap = leftx_cap
    leftLane.last_top = left_top

    # Add the new computed values to current
    leftx = np.append(leftx, leftx_cap)
    lefty = np.append(lefty, 720)
    leftx = np.append(leftx, left_top)
    lefty = np.append(lefty, 0)

    # Apply sort to reoder the pixel vals. Without sorting, its total chaos on the lane!!
    leftx, lefty = leftLane.apply_sort(leftx, lefty)

    # Assign values to the class member vars
    leftLane.X = leftx
    leftLane.Y = lefty

    # Recalculate polynomial with intercepts and average across n frames
    left_coeff = np.polyfit(lefty, leftx, 2)

    # Append the coefficients from the above line
    leftLane.coeff0.append(left_coeff[0])
    leftLane.coeff1.append(left_coeff[1])
    leftLane.coeff2.append(left_coeff[2])

    # Compute the mean of the coefficients
    left_coeff_mean = [np.mean(leftLane.coeff0),
                       np.mean(leftLane.coeff1),
                       np.mean(leftLane.coeff2)]

    # Fit a new polynomial on the new values..Finally!
    left_coeff_x = left_coeff_mean[0] * lefty ** 2 + left_coeff_mean[1] * lefty + left_coeff_mean[2]
    leftLane.coeff_x = left_coeff_x

    # Repeat the same for right side..
    # TODO: May be make a new method in the PolyUtils class to do it in one go..this is tedious

    right_fit = np.polyfit(righty, rightx, 2)

    rightx_cap, right_top = rightLane.get_intercepts(right_fit)

    rightLane.x_cap.append(rightx_cap)
    rightx_cap = np.mean(rightLane.x_cap)
    rightLane.top.append(right_top)
    right_top = np.mean(rightLane.top)
    rightLane.lastx_cap = rightx_cap
    rightLane.last_top = right_top
    rightx = np.append(rightx, rightx_cap)
    righty = np.append(righty, 720)
    rightx = np.append(rightx, right_top)
    righty = np.append(righty, 0)

    rightx, righty = rightLane.apply_sort(rightx, righty)
    rightLane.X = rightx
    rightLane.Y = righty

    right_coeff = np.polyfit(righty, rightx, 2)
    rightLane.coeff0.append(right_coeff[0])
    rightLane.coeff1.append(right_coeff[1])
    rightLane.coeff2.append(right_coeff[2])
    right_mean_coeff = [np.mean(rightLane.coeff0), np.mean(rightLane.coeff1), np.mean(rightLane.coeff2)]

    right_coeff_x = right_mean_coeff[0] * righty ** 2 + right_mean_coeff[1] * righty + right_mean_coeff[2]
    rightLane.coeff_x = right_coeff_x

    # Radius of curvature for each lane
    left_curverad = leftLane.find_radius(leftx, lefty)
    right_curverad = rightLane.find_radius(rightx, righty)

    # Assign the value to the member variable of class PolyUtils
    leftLane.radius = left_curverad
    rightLane.radius = right_curverad

    # Calculate the vehicle position relative to the center of the lane
    position = (rightx_cap + leftx_cap) / 2
    distance_from_center = abs((360 - position) * 3.7 / 700)

    Minv = cv2.getPerspectiveTransform(dst, src)

    warp_zeros = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zeros, warp_zeros, warp_zeros))

    # Stack the left and right points and fill the polygon
    left_points = np.array([np.flipud(np.transpose(np.vstack([leftLane.coeff_x, leftLane.Y])))])
    right_points = np.array([np.transpose(np.vstack([right_coeff_x, rightLane.Y]))])
    total_points = np.hstack((left_points, right_points))

    cv2.polylines(color_warp, np.int_([total_points]), isClosed=False, color=(0, 0, 255), thickness=40)
    cv2.fillPoly(color_warp, np.int_(total_points), (34, 255, 34))

    # Unwarp using the Inverse matrix and image shape info
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)

    # Add overlay info on the video
    if position > 360:
        cv2.putText(result, 'The car is {:.2f}m to the left from center'.format(distance_from_center), (50, 80),
                    fontFace=16, fontScale=1, color=(255, 255, 255), thickness=2)
    else:
        cv2.putText(result, 'The car is {:.2f}m to the right from center'.format(distance_from_center), (50, 80),
                    fontFace=16, fontScale=1, color=(255, 255, 255), thickness=2)
    # Print radius of curvature on video
    cv2.putText(result, 'Curvature Radius: {}(m)'.format(int((leftLane.radius + rightLane.radius) / 2)), (120, 140),
                fontFace=16, fontScale=1, color=(255, 255, 255), thickness=2)

    return result


