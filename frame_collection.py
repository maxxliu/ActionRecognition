import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import os

# NOTE: these scripts are specifically for

# NOTE: this path is for locating the videos when running on google collab
# data folder location
HMDB_DATA = 'drive/My Drive/Computer Vision/HMDB/hmdb51_org/'
TRAINING_PATH = 'drive/My Drive/Computer Vision/HMDB/saved_training/'

# example call that we will run:
# collect_frames_from_video(num_frames=6, frame_processing=flatten_frames, force_resize=(150, 150))
# collect_frames_from_video(num_frames=5, frame_processing=flatten_frames, force_resize=(128, 128), bypass_func=optical_flow_frames)

def collect_frames_from_video(num_frames=1, save_frames=False,
                              frame_processing=None,
                              resize_large=True,
                              resize_small=False,
                              frame_dtype=np.float16,
                              force_resize=None,
                              bypass_func=None):
    '''
    Collect frames from each of the videos and save both the frames and the
    labels for each frame
    NOTE: currently we are taking num_frames as frames that are evenly spaced
          throughout the video

    inputs:
        num_frames - number of frames to extract from each of the videos
        save_frames - after getting the initial raw frames should the function
                      save the frames to a file? this will most likely end
                      up being a very large file and the system may crash when
                      trying to write it to a file
        frame_processing - after collecting the frames we may need to further
                           process each inidivdual frame. user can specify a
                           specific processing function here. choices are:
                           + flatten_frames
        resize_large - resize all of the images to the largest dimensions
                       NOTE: if both resize_large and resize_small are False
                             the frames will not be resized
        resize_small - resize all of the images to the smallest dimensions
        frame_dtype - what dtype should the pixels in each frame be stored as
        force_resize - force all of the images to be resized to these dims,
                       if the argument is given should be in the form:
                       (x dim, y dim)
        bypass_func - this is a function that a user can choose to use instead
                      of just selecting frames from the video, choose from:
                      + optical_flow_frames
    '''
    # get the labels and paths of the video files
    video_files = get_videos_and_labels()

    # select x many frames from each video
    max_x_dim = 0
    max_y_dim = 0
    min_x_dim = 9999999
    min_y_dim = 9999999
    frame_data = []
    labels = []
    count = 0
    print('Working on collecting frames...')
    for label in video_files.keys():
        print('Working on ' + label)
        for video in video_files[label]:
            # select x many evenly spaced frames from the video
            cap = cv2.VideoCapture(video)
            nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            space = int(nframe / (num_frames + 1))
            if bypass_func:
                bypass_func(frame_data, labels, cap, nframe, label, num_frames, frame_dtype)
            else:
                # NOTE: should move this into its own function and let that function
                #       be an argument that the user can choose as a type of
                #       bypass_func
                for i in range(num_frames):
                    frame_pos = (i + 1) * space
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                    ret, frame = cap.read()
                    assert ret, 'ERROR: returned no frame'
                    frameGscl = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # normalize
                    frameGscl = frameGscl / 255
                    assert frameGscl.ndim == 2, 'ERROR: frame has wrong dimensions'
                    labels.append(label)
                    if force_resize:
                        x_dim = force_resize[0]
                        y_dim = force_resize[1]
                        frameGscl = cv2.resize(frameGscl, (y_dim, x_dim))
                    frame_data.append(frameGscl.astype(frame_dtype))
            # read a frame just to get dims
            cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
            ret, frame = cap.read()
            assert ret, 'ERROR: returned no frame'
            frameGscl = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # keep track of largest dimensions to pad every image
            max_x_dim = max(max_x_dim, frameGscl.shape[0])
            max_y_dim = max(max_y_dim, frameGscl.shape[1])
            # keep track of smallest dimensions to pad every image
            min_x_dim = min(min_x_dim, frameGscl.shape[0])
            min_y_dim = min(min_y_dim, frameGscl.shape[1])
        count += 1
        print('\tDone with %d/51' % count)

    # resize the frames if needed
    print('Resizing frames...')
    if force_resize:
        # do not need to resize these again
        pass
    else:
        if resize_large:
            for i, frame in enumerate(frame_data):
                frame = cv2.resize(frame.astype(np.float64), (max_y_dim, max_x_dim))
                frame_data[i] = frame.astype(frame_dtype)
        elif resize_small:
            for i, frame in enumerate(frame_data):
                frame = cv2.resize(frame.astype(np.float64), (min_y_dim, min_x_dim))
                frame_data[i] = frame.astype(frame_dtype)
        else:
            # do not resize any of the frames
            pass

    # time to identify the name of this dataset
    ts = str(int(time.time()))

    # saving full frames
    if save_frames:
        print('Saving full frames...')
        labels_file = 'labels_' + str(num_frames) + '_frame_' + ts + '.npy'
        np.save(TRAINING_PATH + labels_file, labels)
        data_file = 'frame_data_' + str(num_frames) + '_frame_' + ts + '.npy'
        np.save(TRAINING_PATH + data_file, frame_data)

    # moving onto frame processing step
    # NOTE: this can only be done if the images were resized to be the same
    # NOTE: after this processing step, the function will automatically save
    #       the processed data and upon returning the labels will be saved
    if frame_processing and (resize_large or resize_small):
        print('Processing frames...')
        frame_processing(frame_data, frame_dtype, force_resize, num_frames, ts)
        labels_file = 'labels_' + str(num_frames) + '_frame_' + ts + '.npy'
        np.save(TRAINING_PATH + labels_file, labels)

    print('Done')
    return


def get_videos_and_labels():
    '''
    get the file path for all of the videos and the labels of the videos
    '''
    print('Getting video file paths...')
    file_count = 0
    video_files = {}
    for label in os.listdir(HMDB_DATA):
        if '.' in label:
            # this is not a real directory
            pass
        else:
            video_files[label] = []
            new_dir = HMDB_DATA + label + '/'
            for video in os.listdir(new_dir):
                video_files[label].append(new_dir + video)
                file_count += 1
    print('Found %d files' % file_count)

    return video_files


def flatten_frames(frame_data, frame_dtype, force_resize, num_frames, ts):
    '''
    takes the frame data and resizes the frames again if needed and flattens
    the frames
    '''
    if force_resize:
        x_dim = int(force_resize[0])
        y_dim = int(force_resize[0])
    else:
        x_dim = int(frame_data[0].shape[0])
        y_dim = int(frame_data[1].shape[1])

    # flatten
    flat_size = x_dim * y_dim
    for i, frame in enumerate(frame_data):
        if frame.shape != (x_dim, y_dim):
            frame = cv2.resize(frame.astype(np.float64), (y_dim, x_dim))
        frame = frame.flatten().astype(frame_dtype)
        assert len(frame) == flat_size, 'ERROR: flat frame is not correct size'
        frame_data[i] = frame

    # save the frame data
    data_file = 'frame_data_' + str(num_frames) + '_frame_' + 'flat_' + ts + '.npy'
    np.save(TRAINING_PATH + data_file, frame_data)

    return


def optical_flow_frames(frame_data, labels, cap, nframe, label, num_frames, frame_dtype):
    '''
    get num_frames but the frames are actually optical flow frames
    '''
    interval = 2 # example: optical flow beteern frame 10 and 12
    start = int(nframe / 2) - num_frames
    end = int(nframe / 2) + num_frames
    for i in range(start, end, interval):
        # INITIAL FRAME
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        # NEXT FRAME
        cap.set(cv2.CAP_PROP_POS_FRAMES, i + 2)
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # OPTICAL FLOW
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        rgbGscl = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        # store optical flow frame and label
        labels.append(label)
        frame_data.append(rgbGscl.astype(frame_dtype))

    return
