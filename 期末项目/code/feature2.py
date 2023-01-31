import pandas as pd
from feature_engineering import eye_feature, mouth_feature, compute_head_pose
from sklearn.svm import SVC

eye_data = []
eye_data2 = []
mouth_data = []
mouth_data2 = []
head_data = []


def get_mouth_fatigue(f1):
    if f1.shape[0] == 0: return  # 有文件是空的
    frame_list = f1['frame'].drop_duplicates().tolist()

    def find_index(data_col, val):
        val_list = []

        val_list.append(val)
        val_list.append("end")
        index = data_col.isin(val_list).idxmax()

        return index

    first_index_list = []
    for item in frame_list:
        first_index_list.append(find_index(f1.frame, item))

    list = []
    keshui = 0
    time = 0
    time_temp = 0
    for i in range(0, len(frame_list)):
        f11 = f1.loc[f1['frame'] == frame_list[i]]
        x_55 = f11.loc[54 + first_index_list[i]][2]
        x_49 = f11.loc[48 + first_index_list[i]][2]
        y_51 = f11.loc[50 + first_index_list[i]][3]
        y_59 = f11.loc[58 + first_index_list[i]][3]
        y_53 = f11.loc[52 + first_index_list[i]][3]
        y_57 = f11.loc[56 + first_index_list[i]][3]
        w_mouth = x_55 - x_49
        h_mouth = (abs(y_51 - y_59) + abs(y_53 - y_57)) / 2
        list.append({'frame': frame_list[i], 'w': w_mouth, 'h': h_mouth, 'K': h_mouth / w_mouth})
        if h_mouth / w_mouth > 0.75:
            keshui += 1
            time_temp += 1
        else:
            if time_temp > time:
                time = time_temp
            time_temp = 0
    # print("时长："+str(time))
    list = pd.DataFrame(list)
    # print(keshui)
    return keshui / len(frame_list)
    #return time


def getData():
    # 循环示例，后续读取final_data_list.txt

    fileList = open("final_data_list.txt", "r")
    lines = fileList.readlines()
    fileList.close()
    # 画图用factor
    ear_level0y = []
    ear_level1y = []
    ear_level2y = []
    for i in range(0, len(lines)):
        line = lines[i].strip()
        line = line.replace("\\", "/")
        file = pd.read_csv(line)
        if line[11] == '0':
            f, f1 = eye_feature(file, 0)
            ear_level0y.append(0)
            eye_data.append(f)
            eye_data2.append(f1)
        elif line[11] == '1':
            f, f1 = eye_feature(file, 1)
            ear_level1y.append(1)
            eye_data.append(f)
            eye_data2.append(f1)
        else:
            f, f1 = eye_feature(file, 2)
            ear_level2y.append(2)
            eye_data.append(f)
            eye_data2.append(f1)
        mf = mouth_feature(file)
        mouth_data.append(mf)

        m2 = get_mouth_fatigue(file)
        mouth_data2.append(m2)

        hd = compute_head_pose(file)
        head_data.append(hd)
        # print(mouth_feature(file))

    ear_y = ear_level0y + ear_level1y + ear_level2y
    merge = pd.DataFrame(data=[eye_data, mouth_data, head_data, eye_data2, mouth_data2, ear_y],
                         index=['perclos','yawn_times','nod_times','Maximum_eye_closure_time','mouth_time','label']).T
    merge = merge.sample(frac=1).reset_index(drop=True)
    return merge