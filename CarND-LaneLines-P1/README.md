# Udacity Term 1 - Project 1
## Finding Lane Lines on the Road

The repository contains the code for the lane detection. The various computer vision algorithms/apis are used.

### Prerequisites
As specified in the task

### File and code structure

The main code is under P1.ipynb notebook file. I am also using a compile binary to use the **susan** algorithm for edge detection. However, this binary will be platform dependent, thus if someone wants to try susan method, they will have to compile the C code for the susan file and replace the existing susan (binary)

The edge detection method can be switched in the following way:

There is a global variable "mode" which can have 3 values:
1. canny ( It will execute api for standard canny edge detection )
2. auto ( It will do automatic computation of the low and high threshold values which are used by standard canny algo )
3. susan ( It will use the pre-compiled binary for the susan edge detection ).


The write_up.md file contains the write up as required by the task. It contains the detailed info about the steps used.

The folders test_images_output and test_videos_output contains the output of the given input image and video files.
