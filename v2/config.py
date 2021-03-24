
###  Main
"""
OUTPUT_FOLDER: path to save all the output files
START_FRAME: program starts from this frame.
    Give it a number to skip the beginning part of the video
AOI: Area of Interest. Crop the image before processing
    (x_left, x_right, y_left, y_right)
    Default full image (0, -1, 0, -1)
"""
OUTPUT_FOLDER = './output/'
START_FRAME = 0
# START_FRAME = frames_per_sec * 15
AOI = (0, -1, 0, -1)

### Function Variables
## get_hSv_value_and_extract_face
"""
INTERVAL: skip this number of frames before processing one
    Increase this number will speed up the program but lower the accuracy, since we are skipping frames
N_FACE: number of faces expecting from the input
    ***MODIFY THIS NUMBER ACCORDING TO THE INPUT VIDEO***
"""
INTERVAL = 15
N_FACE = 2
## find_candidates
# cut paper
"""
thresh_low/high: variables for detecting papers/contracts
frame_length: (frame_length * INTERVAL) is the number of frame we expect the paper staying in front of the camera
    Lower this value if no contract is found
"""
_thresh_low = 0
_thresh_high = 0.65
_frame_length = 10
CUT_PAPER_ARGS = _thresh_low, _thresh_high, _frame_length
## crop id card
"""
ID_THRESH: Confidence threshold for id card detector
    Lower this value if no card is found
DRAW_BOX: Whether to draw the box around the card
    Turn on this will hurt performance slightly, in terms of speed. But good for debugging purposes
"""
ID_THRESH = 0.98
DRAW_BOX = False
