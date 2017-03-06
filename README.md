#**Finding Lane Lines on the Road**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

## Dependencies
* Python 3.5
* NumPy
* OpenCV
* Matplotlib
* MoviePy

## How to run
To run the script stand-alone:

```
Usage: detect_road_lanes.py [options]

Options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file=INPUT_FILE
                        Input video/image file
  -o OUTPUT_FILE, --output_file=OUTPUT_FILE
                        Output destination for video/image file
  -I, --image_only      Process image file containing road lines (defaults to
                        video)
```

For example, to detect lanes lines on the video 'white.mp4', run:

```python detect_road_lanes.py -i solidWhiteRight.mp4 -o white_out.mp4```

To detect lane lines on a single image, for example test_images/solidYellowLeft.jpg, run:

```python detect_road_lanes.py -i test_images/solidYellowLeft.jpg -o sYL_out.jpg -I```

For detailed explanation of what the code does and the project guideline, refer to P1.ipynb via ```jupyter notebook```
