import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# 眼睛关键点的索引
# 左眼
left_top_left_eye = 37
left_top_right_eye = 38
left_bottom_left_eye = 41
left_bottom_right_eye = 40
left_left_eye_point = 36
left_right_eye_point = 39
# 右眼
right_top_left_eye = 43
right_top_right_eye = 44
right_bottom_left_eye = 47
right_bottom_right_eye = 46
right_left_eye_point = 42
right_right_eye_point = 45

# 嘴关键点索引
left_mouth_point = 48
right_mouth_point = 54
top_left_mouth = 50
top_middle_mouth = 51
top_right_mouth = 52
bottom_left_mouth = 58
bottom_middle_mouth = 57
bottom_right_mouth = 56

# 视频的帧数
video_length = 50

# p70,可以调整眼睛闭合比例的大小即pxx
# p80,可以调整眼睛闭合比例的大小即pxx
p = 0.2

# F为疲劳度阈值，f>F即为疲劳
F = 0.3

# 嘴巴纵横比阈值
openThresh = 0.6

# f值，即perclos(按照一条csv的close_thresh计算)值
ear_self_level0 = []
ear_self_level1 = []
ear_self_level2 = []
# f值，按照mean_ear计算
ear_mean_level0 = []
ear_mean_level1 = []
ear_mean_level2 = []
# ear的差值（一条csv中，earList的max-min）
ear_level0_difference = []
ear_level1_difference = []
ear_level2_difference = []
# level0的close_thresh
mean_ear = 0

# 世界坐标系(UVW)：填写3D参考点，该模型参考http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
object_pts = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                         [1.330353, 7.122144, 6.903745],  # 29左眉右角
                         [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                         [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                         [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                         [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                         [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                         [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                         [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                         [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                         [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                         [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                         [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                         [0.000000, -7.415691, 4.070434]])  # 6下巴角
# 相机坐标系(XYZ)：添加相机内参
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
# 像素坐标系(xy)：填写凸轮的本征和畸变系数
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coffees = np.array(D).reshape(5, 1).astype(np.float32)
# 重新投影3D点的世界坐标轴以验证结果姿势
reprojectSrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])


# 计算眼部的纵横比
def compute_ear(data, frame):
    # (|(p37-p41)|+|(p38-p40)|)/2*|(p36-p39)|
    # 减少计算次数
    index = frame * 68
    # 左眼计算
    left_eye_top_left = index + left_top_left_eye
    left_eye_bottom_left = index + left_bottom_left_eye
    left_eye_top_right = index + left_top_right_eye
    left_eye_bottom_right = index + left_bottom_right_eye
    left_vertical_left = compute_line_distance(data["x"][left_eye_top_left], data["x"][left_eye_bottom_left],
                                               data["y"][left_eye_top_left], data["y"][left_eye_bottom_left])
    left_vertical_right = compute_line_distance(data["x"][left_eye_top_right], data["x"][left_eye_bottom_right],
                                                data["y"][left_eye_top_right], data["y"][left_eye_bottom_right])
    # 得到纵向长度
    left_vertical_length = left_vertical_right + left_vertical_left
    # 计算横向长度
    left_eye_left_index = index + left_left_eye_point
    left_eye_right_index = index + left_right_eye_point
    left_horizontal_length = compute_line_distance(data["x"][left_eye_left_index], data["x"][left_eye_right_index],
                                                   data["y"][left_eye_left_index], data["y"][left_eye_right_index])
    left_ear = left_vertical_length / (2 * left_horizontal_length)
    # 右眼计算
    right_eye_top_left = index + right_top_left_eye
    right_eye_bottom_left = index + right_bottom_left_eye
    right_eye_top_right = index + right_top_right_eye
    right_eye_bottom_right = index + right_bottom_right_eye
    right_vertical_left = compute_line_distance(data["x"][right_eye_top_left], data["x"][right_eye_bottom_left],
                                                data["y"][right_eye_top_left], data["y"][right_eye_bottom_left])
    right_vertical_right = compute_line_distance(data["x"][right_eye_top_right], data["x"][right_eye_bottom_right],
                                                 data["y"][right_eye_top_right], data["y"][right_eye_bottom_right])
    # 得到纵向长度
    right_vertical_length = right_vertical_right + right_vertical_left
    # 计算横向长度
    right_eye_left_index = index + right_left_eye_point
    right_eye_right_index = index + right_right_eye_point
    right_horizontal_length = compute_line_distance(data["x"][right_eye_left_index], data["x"][right_eye_right_index],
                                                    data["y"][right_eye_left_index], data["y"][right_eye_right_index])
    right_ear = right_vertical_length / (2 * right_horizontal_length)

    if abs(right_ear-left_ear)>0.01:
        ear=max(right_ear,left_ear)
    else:
        ear = (left_ear + right_ear) / 2
    return ear


# 目前仅为二分类，即疲劳，不疲劳
# 可以将f作为特征值输入三分类模型进行训练
def compute_perclos(data, type):
    ear_list = []
    longest_close_eye = 0
    for frame in range(0, video_length):
        ear = compute_ear(data, frame)
        ear_list.append(ear)
    if type == 0:
        # 计算阈值
        close_eye_thresh = min(ear_list) + (max(ear_list) - min(ear_list)) * p
        ear_level0_difference.append(max(ear_list) - min(ear_list))
    elif type == 1:
        close_eye_thresh = min(ear_list) + (max(ear_list) - min(ear_list)) * p
        ear_level1_difference.append(max(ear_list) - min(ear_list))
    else:
        close_eye_thresh = min(ear_list) + (max(ear_list) - min(ear_list)) * p
        ear_level2_difference.append(max(ear_list) - min(ear_list))
    # 统计闭眼次数
    close_count = 0
    temp_longest = 0
    for ear in ear_list:
        # if ear < close_eye_thresh:
        if ear < close_eye_thresh:
            close_count += 1
            temp_longest += 1
        else:
            if longest_close_eye < temp_longest:
                longest_close_eye = temp_longest
            temp_longest = 0
    f = close_count / video_length
    if type == 0:
        ear_mean_level0.append(f)
    elif type == 1:
        ear_mean_level1.append(f)
    else:
        ear_mean_level2.append(f)
    close_count = 0
    for ear in ear_list:
        if ear < close_eye_thresh:
            close_count += 1
    if type == 0:
        ear_self_level0.append(f)
    elif type == 1:
        ear_self_level1.append(f)
    else:
        ear_self_level2.append(f)
    return f, longest_close_eye


# 计算嘴部纵横比
def compute_mar(data, frame):
    # (| (p61-p67) | + | (p62 - p66) |+|(p63-p65)|) / 3 * | (p48 - p54) |
    # 减少计算次数
    index = frame * 68
    # 计算纵向长度
    # 左侧索引
    left_top_index = index + top_left_mouth
    left_bottom_index = index + bottom_left_mouth
    left_vertical_length = compute_line_distance(data["x"][left_top_index], data["x"][left_bottom_index],
                                                 data["y"][left_top_index], data["y"][left_bottom_index])
    # 右侧索引
    right_top_index = index + top_right_mouth
    right_bottom_index = index + bottom_right_mouth
    right_vertical_length = compute_line_distance(data["x"][right_top_index], data["x"][right_bottom_index],
                                                  data["y"][right_top_index], data["y"][right_bottom_index])
    # 中部索引
    middle_top_index = index + top_middle_mouth
    middle_bottom_index = index + bottom_middle_mouth
    middle_vertical_length = compute_line_distance(data["x"][middle_top_index], data["x"][middle_bottom_index],
                                                   data["y"][middle_top_index], data["y"][middle_bottom_index])
    vertical_length = left_vertical_length + right_vertical_length + middle_vertical_length
    # 计算横向长度
    mouth_left_index = index + left_mouth_point
    mouth_right_index = index + right_mouth_point
    horizontal_length = compute_line_distance(data["x"][mouth_left_index], data["x"][mouth_right_index],
                                              data["y"][mouth_left_index], data["y"][mouth_right_index])
    mar = vertical_length / (3 * horizontal_length)
    return mar


# 眼部特征
def eye_feature(data, type):
    # 得到f值
    return compute_perclos(data, type)


# 嘴部特征
def mouth_feature(data):
    # 超过5帧大于阈值即计为哈欠
    # 首先初始化哈欠标志位yawn_flag和哈欠计数器yawn_counter为0，当检测到嘴巴张开(MAR > 0.75)
    # 时，yawn_counter自加1，当连续3次检测到MAR > 0.75
    # 即认为驾驶人正在张嘴，开始计时并将yawn_counter置1。当检测到驾驶人嘴巴闭合时，若张嘴持续时间大于等于1.5秒，则认为打了一次哈欠
    # ，哈欠次数yawns自加1。每60秒统计一次哈欠次数，当达到规定的3次 / min时触发警报。
    yawns = 0
    yawn_counter = 0
    for frame in range(0, video_length):
        mar = compute_mar(data, frame)
        if mar > openThresh:
            yawn_counter += 1
            # 这个十五帧需要进行调参
            if yawn_counter >= 3:
                yawns += 1
                yawn_counter = 0
        else:
            yawn_counter = 0
    return yawns


def compute_line_distance(x1, x2, y1, y2):
    x = x1 - x2
    y = y1 - y2
    distance = math.sqrt(x ** 2 + y ** 2)
    return distance


def compute_feature(data, frame):
    return [compute_ear(data, frame), compute_mar(data, frame), *get_head_pose(data, frame)]


def return_feature(filePath):
    data = pd.read_csv(filePath)
    feature = []
    for i in range(0, video_length):
        feature.append(compute_feature(data, i))
    return feature, int(filePath[11])


def get_head_pose(shape, index):  # 头部姿态估计
    index = index * 68
    # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
    # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
    # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    image_pts = np.float32(
        [[shape["x"][index + 17], shape["y"][index + 17]], [shape["x"][index + 21], shape["y"][index + 21]],
         [shape["x"][index + 22], shape["y"][index + 22]], [shape["x"][index + 26], shape["y"][index + 26]],
         [shape["x"][index + 36], shape["y"][index + 36]],
         [shape["x"][index + 39], shape["y"][index + 39]], [shape["x"][index + 42], shape["y"][index + 42]],
         [shape["x"][index + 45], shape["y"][index + 45]], [shape["x"][index + 31], shape["y"][index + 31]],
         [shape["x"][index + 35], shape["y"][index + 35]],
         [shape["x"][index + 48], shape["y"][index + 48]], [shape["x"][index + 54], shape["y"][index + 54]],
         [shape["x"][index + 57], shape["y"][index + 57]], [shape["x"][index + 8], shape["y"][index + 8]]])
    # solvePnP计算姿势——求解旋转和平移矩阵：
    # rotation_vec表示旋转矩阵，translation_vec表示平移矩阵，cam_matrix与K矩阵对应，dist_coeffs与D矩阵对应。
    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coffees)
    # projectPoints重新投影误差：原2d点和重投影2d点的距离（输入3d点、相机内参、相机畸变、r、t，输出重投影2d点）
    reprojectDst, _ = cv2.projectPoints(reprojectSrc, rotation_vec, translation_vec, cam_matrix, dist_coffees)
    reprojectDst = tuple(map(tuple, reprojectDst.reshape(8, 2)))  # 以8行2列显示

    # 计算欧拉角calc euler angle
    # 参考https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)  # 罗德里格斯公式（将旋转矩阵转换为旋转向量）
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))  # 水平拼接，vconcat垂直拼接
    # decomposeProjectionMatrix将投影矩阵分解为旋转矩阵和相机矩阵
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    return pitch, yaw, roll
    # return reprojectDst, euler_angle  # 投影误差，欧拉角


def compute_head_pose(file):
    hCOUNTER = 0
    hTOTAL = 0
    NOD_AR_CONSEC_FRAMES = 3
    pitch_list = []
    for frame in range(0, video_length):
        har = get_head_pose(file,frame)[0]  # 取pitch旋转角度
        pitch_list.append(har)
        if abs(har - pitch_list[frame - 1]) > 5 and frame - 1 >= 0:  # 点头阈值0.3
            hCOUNTER += 1
        else:
            # 如果连续3次都小于阈值，则表示瞌睡点头一次
            if hCOUNTER >= NOD_AR_CONSEC_FRAMES:  # 阈值：3
                hTOTAL += 1
            # 重置点头帧计数器
            hCOUNTER = 0
    return hTOTAL
