# Action Recognition

This repository contains scripts and notebooks that were used for a 
preliminary study on action recognition using equal space sampling and
a 2D CNN. The notebooks were originally run on a google colab instance
that had a maximum RAM of 12gb which was our primary limitation in working
with larger datasets.

## Files

- frame_collection.py - contains the functions that retrieve the video files
and samples frames from each video file. Also has functionality to collect
optical flow data instead. Returns back flattened frames.

- frame_collection.ipynb - used to create training data

- cnn_test.ipynb - contains the main 2D CNN model that we are using

- cnn.ipynb - secondary model that runs much faster than cnn_test but 
produces worse accuracy

- feature_testing.ipynb - notebook that was used to explore features of
the videos and frames

- visualizations.py - has the results from our initial study hard coded in
so that we can visualize the training and validation results for each test

**Note:** None of these files will run as they all have paths hardcoded in
that were specific to file locations that we mounted from google drive
(this way we did not need to store large files locally and could share
the same dataset when testing on multiple machines)
