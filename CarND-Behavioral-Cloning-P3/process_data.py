import cv2
import numpy as np
import csv

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

###############################################################################

# Constants

BATCH_SIZE = 128
MIN_SAMPLPES = 1250
MAX_SAMPLES = 1500
STEERING_CORRECTION = 0.2

##############################################################################


def initialize_file_paths():
    udacity_data = "./data/udacity_data/driving_log.csv"
    my_data = "./data/my_data/driving_log.csv"
    return udacity_data, my_data

##############################################################################


# Read the csv inside the data folders
def process_csv(udacity_data, my_data):

    """
        This function will process the csv files of udacity and my recorded data. It will prepare list of the
        entire images and the steering angle values
    Returns
        Two arrays for training and validation. I am using sklearn train_test_split method to split the dataset
    """

    lines = []

    # Open a csv reader
    with open(udacity_data) as csv_file:
        reader = csv.reader(csv_file)
        dir=udacity_data.replace("driving_log.csv","")
        for line in reader:

            line[0]=dir+line[0]
            line[1] = dir + line[1]
            line[2] = dir + line[2]
            #print(line)
            lines.append(line)
        num_udacity_image = len(lines)
        print("Udacity images: ", len(lines))

    if my_data != "":
        # Open a csv reader
        with open(my_data) as csv_file:
            reader = csv.reader(csv_file)
            dir = my_data.replace("driving_log.csv", "")
            for line in reader:

                line[0]=dir+line[0]
                line[1] = dir + line[1]
                line[2] = dir + line[2]
                #print(line)
                lines.append(line)
        print("Recorded images", len(lines) - num_udacity_image)
    # Use sklearn shuffle to shuffle the data
    # It helps in reducing the validation loss
    lines_shuffled = shuffle(lines)

    print("Total: ", len(lines_shuffled))

    # Use train_test_split to split the dataset into train and validation  set
    train_lines, validation_lines = train_test_split(lines_shuffled, test_size=0.2)

    return np.array(train_lines), np.array(validation_lines)

###############################################################################


def pre_process_image(input_image, convert_to_color=cv2.COLOR_BGR2YUV,
                      convert_to_binary=False):
    """
        This function will receive an raw image array. It does the following:
        1. Converts the image to YUV color space, as suggested in the NVIDIA paper
        2. I crop the image such that only the meaningful area is visible
        3. I also tried to use the previous project learning, i.e. to convert the image
            into a binary threshold, however, the I the validation loss increased, thus
            I have commented the code for binary threshold.
    Returns
        Processed image
    """
    cropped_image = input_image[60:140, :, :]
    processed_image = cv2.cvtColor(cropped_image, convert_to_color)

    if convert_to_binary:
        imghsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

        imghsv[:, :, 2] = [[max(pixel - 25, 0) if pixel < 190 else min(pixel + 25, 255) for pixel in row] for row in
                           imghsv[:, :, 2]]
        # cv2.imshow('contrast', cv2.cvtColor(imghsv, cv2.COLOR_HSV2BGR))
        # B-Channel threshold values
        b_thresh_min = 145
        b_thresh_max = 200

        # L-Channel threshold values
        l_thresh_min = 215
        l_thresh_max = 255

        # S-Channel threshold values
        s_thresh_min = 180
        s_thresh_max = 255
        # print(img.shape)

        # create binary thresholded images using B and L channel
        b_channel = cv2.cvtColor(imghsv, cv2.COLOR_RGB2Lab)[:, :, 2]
        l_channel = cv2.cvtColor(imghsv, cv2.COLOR_RGB2LUV)[:, :, 0]

        # Threshold color channel
        s_channel = cv2.cvtColor(imghsv, cv2.COLOR_BGR2HLS)[:, :, 2]

        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

        combined_binary = np.zeros_like(b_binary)
        combined_binary[(l_binary == 1) | (b_binary == 1) | (s_binary == 1)] = 1
        # Convert to 3 channels to support the input layer of the network
        combined_binary = np.dstack((combined_binary, combined_binary, combined_binary))
        # Convert to float as int values are not good enough for human eyes to visualize
        # Multiply by 255 or some factor, if needed to visualize
        processed_image = np.asarray(combined_binary, dtype="float")

    return processed_image

###############################################################################


def uniform_split_data(image_dataset):
    """
    This function will try to uniformly distribution of the image dataset It has higher and lower threshold number
    of bins in the histogram. I tried with higher number as the distribution was wide spread. This is visible in the
    Visualization.ipynb.

    The purpose is to do data augmentation
    """

    image_dataset_output = image_dataset.copy()
    
    # Create histogram
    steering_angle = np.asarray(image_dataset_output[:,3], dtype='float')
    num_hist, _index = np.histogram(steering_angle, 25)
    
    add_entries = np.empty([1,7])
    delete_entries = np.empty([1,1])
    
    for i in range(1, len(num_hist)):
        if num_hist[i-1]<MIN_SAMPLPES:

            # Find values within the desired range
            match_idx = np.where((steering_angle>=_index[i-1]) & (steering_angle<_index[i]))[0]

            # Random choice until minimum number
            need_to_add = image_dataset_output[np.random.choice(match_idx,MIN_SAMPLPES-num_hist[i-1]),:]
            
            add_entries = np.vstack((add_entries, need_to_add))

        elif num_hist[i-1]>MAX_SAMPLES:
            
            # Find values within the desired range
            index = np.where((steering_angle>=_index[i-1]) & (steering_angle<_index[i]))[0]
            
            # Random choice until minimum number
            delete_entries = np.append(delete_entries, np.random.choice(index,num_hist[i-1]-MAX_SAMPLES))

    # Add and delete into dataset to make it a uniform distribution
    image_dataset_output = np.delete(image_dataset_output, delete_entries, 0)
    image_dataset_output = np.vstack((image_dataset_output, add_entries[1:,:]))
    
    return image_dataset_output

###############################################################################


def data_generator(image_dataset, batch_size=BATCH_SIZE, is_flip=False):
    """

    :param image_dataset: preprocessed images supplied to the network
    :param batch_size: Number of batches in which data is fed to the network
    :param is_flip: If true, this method will flip the images and will create images on the fly
    :return: Image and steering angle as a list

    This function will add the steering correction as suggested in the lecture video.
    I tried to use opencv flip method to create more image dataset using the existing data,
    however, the network didnt person so well. I will try to check this later, so marking it as TODO
    """

    while True:
        for index in range(0, len(image_dataset), batch_size):
            batch_obs = shuffle(image_dataset[index:index+batch_size])

            center_images = []
            left_images = []
            right_images = []

            flip_center_images = []
            flip_left_images = []
            flip_right_images = []

            steering_angle_center = []
            steering_angle_left = []
            steering_angle_right = []

            flip_steering_angle_center = []
            flip_steering_angle_left = []
            flip_steering_angle_right = []

            for observation in batch_obs:

                center_image_path = observation[0]
                left_image_path = observation[1]
                right_image_path = observation[2]

                center_images.append(pre_process_image(cv2.imread(center_image_path),cv2.COLOR_RGB2YUV))
                steering_angle_center.append(float(observation[3]))

                left_images.append(pre_process_image(cv2.imread(left_image_path),cv2.COLOR_RGB2YUV))
                steering_angle_left.append(float(observation[3]) + STEERING_CORRECTION)

                right_images.append(pre_process_image(cv2.imread(right_image_path),cv2.COLOR_RGB2YUV))
                steering_angle_right.append(float(observation[3]) - STEERING_CORRECTION)

                # Not used for final model generation
                # The steering correction is opposite in case of flipped images
                if is_flip:
                    flip_center_images.append(cv2.flip(pre_process_image(cv2.imread(center_image_path)), 1))
                    flip_steering_angle_center.append(float(observation[3]))
                    flip_left_images.append(cv2.flip(pre_process_image(cv2.imread(left_image_path)), 1))
                    flip_steering_angle_left.append(float(observation[3]) - STEERING_CORRECTION)
                    flip_right_images.append(cv2.flip(pre_process_image(cv2.imread(right_image_path)), 1))
                    flip_steering_angle_right.append(float(observation[3]) + STEERING_CORRECTION)

            images = center_images + left_images + right_images
            steering_angles = steering_angle_center + steering_angle_left + steering_angle_right

            # Not used for final model generation
            if is_flip:
                images = flip_center_images + flip_left_images + flip_right_images
                steering_angles \
                    = flip_steering_angle_center + flip_steering_angle_left + flip_steering_angle_right

            X = np.array(images)
            y = np.array(steering_angles)

            yield shuffle(X, y)

###############################################################################
