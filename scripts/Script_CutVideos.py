#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script adapted from  


"""


import argparse  # module used to parse command line arguments
import cv2  # still used to save images out
import os
import sys
import numpy as np
from decord import VideoReader
from decord import cpu, gpu

# Parsing

# Arguments provided in command line will essentially be the parameters of extract_frames()

# Positional (mandatory) arguments
parser = argparse.ArgumentParser(description="cut a video into B&W png images")
parser.add_argument("path", help="path of the video (e.g.: ./videos/example.mp4)")
parser.add_argument("dir", help="folder where images will be saved (e.g.: ./images)")

# Optional arguments
parser.add_argument("-o", "--overwrite", action="store_true", help="enable image overwriting")
parser.add_argument("--start", type=int, default=-1, help="start frame")
parser.add_argument("--end", type=int, default=-1, help="end frame")
parser.add_argument("--every", type=int, default=1, help="frame spacing")

# Parse arguments
args = parser.parse_args()

# Since default parameters are already set using argparse, there's no need for it in a function's definition
def extract_frames(video_path, frames_dir, overwrite, start, end, every):
    """
    Extract frames from a video using decord's VideoReader
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param overwrite: to overwrite frames that already exist?
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """

    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    # video_path = os.path.normpath("/home/julan/Downloads/TestVideo-2020-08-07_16.20.54.mp4")  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible
    # frames_dir = os.path.normpath("/home/julan/Downloads/Temporal")  # make the paths OS (Windows) compatible

    video_dir, video_filename = os.path.split(video_path)  # get the video path and filename from the path
    # video_dir, video_filename = os.path.split("/home/julan/Downloads/Temporal")
    
    # return video_path, video_dir, video_filename
    try:
        assert os.path.exists(video_path)  # assert the video file exists
    except AssertionError:
        print("Error loading file on path '{}'".format(args.path))


    # load the VideoReader
    vr = VideoReader(video_path, ctx=cpu(0))  # can set to cpu or gpu .. ctx=gpu(0)
                     
    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = len(vr)

    frames_list = list(range(start, end, every))
    saved_count = 0

    if every > 25 and len(frames_list) < 1000:  # this is faster for every > 25 frames and can fit in memory
        frames = vr.get_batch(frames_list).asnumpy()

        for index, frame in zip(frames_list, frames):  # lets loop through the frames until the end
            save_path = os.path.join(frames_dir, video_filename, "{:010d}.jpg".format(index))  # create the save path
            if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # save the extracted image
                saved_count += 1  # increment our counter by one

    else:  # this is faster for every <25 and consumes small memory
        for index in range(start, end):  # lets loop through the frames until the end
            frame = vr[index]  # read an image from the capture
            
            if index % every == 0:  # if this is a frame we want to write out based on the 'every' argument
                save_path = os.path.join(frames_dir, "{:010d}.jpg".format(index))  # create the save path
                if not os.path.exists(save_path) or overwrite:  # if it doesn't exist or we want to overwrite anyways
                    cv2.imwrite(save_path, cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR))  # save the extracted image
                    saved_count += 1  # increment our counter by one

    return saved_count  # and return the count of the images we saved


# Call function using argparse

extract_frames(args.path, args.dir, args.overwrite, args.start, args.end, args.every)