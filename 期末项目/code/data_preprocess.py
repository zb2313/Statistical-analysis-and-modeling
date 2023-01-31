import numpy as np
import pandas as pd
import os


def read_file(file_name, is_skip_header=True):
    if is_skip_header:
        file = pd.read_csv(file_name, header=None, sep=',')
    else:
        file = pd.read_csv(file_name, sep=',')
    return file


def delete_empty_file(root='origin'):
    levels = os.listdir(root)
    with open('non_empty_file_list.txt', 'w') as f:
        for level in levels:
            level_path = os.path.join(root, level)
            file_dirs = os.listdir(level_path)
            for file_name in file_dirs:
                file_path = os.path.join(level_path, file_name)
                file = read_file(file_path)
                if len(file) > 1:
                    f.write(file_path + '\n')


def delete_lack_frame_file(rate=0.08):
    f = open("non_empty_file_list.txt", "r")
    lines = f.readlines()
    f.close()
    count = 0
    with open("non_lack_frame_file_list.txt", "w") as f:
        for line in lines:
            line = line.strip()
            video_file = read_file(line.strip(), False)
            max_frame = max(video_file["frame"].drop_duplicates().tolist())
            if len(video_file) / max_frame / 68 > 1 - rate:
                f.write(line + '\n')
            else:
                count += 1
    print(count / len(lines))


def interpolate(method="linear"):
    f = open("non_lack_frame_file_list.txt", "r")
    lines = f.readlines()
    f.close()
    count = 0
    for line in lines:
        line = line.strip()
        video_file = read_file(line.strip(), False)
        video_file.columns = ["No", "frame", "x", "y"]
        line = line.replace("origin", "data")
        frames = video_file["frame"].drop_duplicates().tolist()
        max_frame = max(frames)
        if len(video_file) == max_frame * 68:
            video_file.to_csv(line, index=False)
        else:
            j = 0
            for i in range(1, max_frame + 1):
                if frames[j] != i:
                    df1 = video_file.iloc[:(i-1)*68, :]
                    df2 = video_file.iloc[(i-1)*68:, :]
                    df_add = np.full((68, 4), np.nan)
                    df_add[:, 1] = i
                    df_add[:, 0] = range(68)
                    df_add = pd.DataFrame(df_add, columns=["No", "frame", "x", "y"])
                    video_file = pd.concat((df1, df_add, df2), ignore_index=True)
                else:
                    j += 1
            for i in range(68):
                video_file[i:-1:68] = video_file[i:-1:68].interpolate(method=method)
            video_file.to_csv(line, index=False)


def split_or_delete_frames(limit=50, root='data'):
    count_del = 0
    count = 0
    with open("final_data_list.txt", "w") as f:
        levels = os.listdir(root)
        for level in levels:
            level_path = os.path.join(root, level)
            file_dirs = os.listdir(level_path)
            for file_name in file_dirs:
                count += 1
                file_path = os.path.join(level_path, file_name)
                video_file = read_file(file_path, False)
                frames = video_file["frame"].drop_duplicates().tolist()
                max_frame = max(frames)
                if max_frame < limit:
                    count_del += 1
                    continue
                if max_frame / limit < 1.5:
                    start_frame = int((max_frame - limit) // 2)
                    video_file[start_frame * 68: int((start_frame + limit)) * 68].to_csv(file_path, index=False)
                    f.write(file_path + '\n')
                else:
                    video_file[:limit*68].to_csv(file_path.replace(".csv", "1.csv"), index=False)
                    f.write(file_path.replace(".csv", "1.csv") + '\n')
                    video_file[-limit*68:-1].to_csv(file_path.replace(".csv", "2.csv"), index=False)
                    f.write(file_path.replace(".csv", "2.csv") + '\n')
        print(count_del / count)

