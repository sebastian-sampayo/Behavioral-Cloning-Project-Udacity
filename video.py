# =========================================================================== #
# Udacity Nanodegree: Self-Driving Car Engineering - December cohort
# Project 3: Behavioral Cloning
# Date: 12th February 2017
# 
# Author: Sebastián Lucas Sampayo
# e-mail: sebisampayo@gmail.com
# file: video.py
# =========================================================================== #
# This file is used to generate a video with the images taken from the
# center camera of the car while driving autonomously.
# This file was provided by Udacity.

from moviepy.editor import ImageSequenceClip
import argparse


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    video_file = args.image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(args.image_folder, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
