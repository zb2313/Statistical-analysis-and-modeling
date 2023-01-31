import cv2
import pandas as pd
import numpy as np

# 定义视频参数
VIDEO_HEIGHT = 720
VIDEO_WIDTH = 1280
VIDEO_FPS = 10


def main():
    # keypoint文件读取
    kp_file_path = './data/level_0/level_0 (12).csv'
    kp_df = pd.read_csv(kp_file_path)
    for frame_idx in range(1, int(max(kp_df['frame'])) + 1):
        # 逐帧可视化keypoint
        kps = kp_df[kp_df['frame'] == frame_idx].reset_index(drop=True)
        show_bg = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3))
        for kp_idx in range(len(kps)):
            cv2.circle(show_bg, (int(kps['x'][kp_idx]), int(kps['y'][kp_idx])), 1, (0, 0, 255), 1)
        cv2.imshow("kp vis", show_bg)
        cv2.waitKey(int(1 / VIDEO_FPS * 2000))


if __name__ == '__main__':
    main()
