# -*- coding: utf-8 -*-
"""
The functions defined in this script can be used to prepare a video for the
orb-slam example application 'mono_tum'. This example requires us to supply
the video frames as images rather than as a video. Furthermore, it needs
a file 'rgb.txt' which contains the timestamps and filenames of the files.
"""

VIDEOFILE = "IMG_4145"
SUFFIX = ".mov"

# Target image size/scaling. Only set one parameter to non-zero.
SCALE = 0
WIDTH = 640

# To select part of the image, specify the first and the last frame here. If
# any of those values is 0, it will be ignored. For FIRST, this is the same as
# starting with the first frame.
FIRST = 0
LAST = 0

import os # To create directories

def video_to_frames(videofile, suffix, firstframe, lastframepp):
    import cv2
    
    # Determine the required scaling for the target image size
    video = cv2.VideoCapture(videofile + suffix)
    ret, frame = video.read()
    if not ret:
        raise(Exception('Error while reading the first frame from the video file (it does not seem to have any frames).'))
    initshape = frame.shape
    if SCALE:
        scale = SCALE
    elif WIDTH:
        scale = WIDTH / initshape[1]
    
    # We store the frame-images in a folder called the same as the movie filename.
    # Here, ensure that this directory exists (mkdir -p).
    os.makedirs(videofile + '/rgb', exist_ok=True)
    
    # Reload the video so that we do not miss the first frame
    video = cv2.VideoCapture(videofile + suffix)
    
    framecount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for framenum in range(0, framecount):
        ret, frame = video.read()
        if not ret:
            # The exception here is commented out because in one of Patrick's
            # slow-motion videos, the determined framecount was wrong. Therefore,
            # just return here. But before, remember that the correct
            # framecount.
            #raise(Exception('Error while reading a frame from the video file.'))
            print('WARNING: Error while reading a frame from the video file. Maybe, the determined framecount is wrong?')
            framecount = framenum
            break
        print(framenum)
        if framenum < firstframe:
            continue
        if lastframepp and framenum >= lastframepp:
            break
        smallframe = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        cv2.imshow('asdf', smallframe)
        framefile = videofile+"/rgb/{:05d}.png".format(framenum)
        cv2.imwrite(framefile, smallframe)
        cv2.waitKey(1)
    ret, frame = video.read()
    if ret:
        print('WARNING: It seems that the video was not finished yet. Maybe, the determined framecount is wrong?')
    cv2.destroyAllWindows()
    return framecount-1

def generate_rgb_framesfile(firstframe, lastframepp, foldername=None):
    if foldername:
        os.makedirs(foldername, exist_ok=True)
    filename = foldername + '/rgb.txt' if foldername else 'rgb.txt'
    with open(filename, 'w') as file:
        file.write('# first line to skip\n')
        file.write('# second line to skip\n')
        file.write('# third line to skip\n')
        timestamp = 0
        for filenum in range(firstframe, lastframepp):
            nextline = "{} rgb/{:05d}.png\n".format(timestamp, filenum)
            timestamp += .015 # INCREASE VIDEO SPEED :)
            file.write(nextline)

if __name__ == "__main__":
    framecount = video_to_frames(VIDEOFILE, SUFFIX, FIRST, LAST)
    lastframepp = min(LAST, framecount+1) if LAST else framecount+1
    generate_rgb_framesfile(FIRST, lastframepp, VIDEOFILE)