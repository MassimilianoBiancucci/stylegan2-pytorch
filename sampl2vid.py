import cv2
import numpy as np
import glob
from tqdm import tqdm
from skvideo.io import vwrite, FFmpegWriter

frameSize = 256
frameSize = (frameSize, frameSize)

delay = 1
samples_path = '/home/ubuntu/hdd/stylegan2-pytorch/sample/'

if __name__ == '__main__':

    samples_list = glob.glob(samples_path + '*.png')
    samples_list.sort()

    writer = FFmpegWriter("train_video.mp4")

    for i, filename in tqdm(enumerate(samples_list)):
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.writeFrame(img)

    writer.close()


    print('video generated')