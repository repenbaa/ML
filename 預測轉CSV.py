import csv
import os
import random
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model

layer_name = 'global_max_pooling2d'
model = tf.keras.models.load_model(
    'E:/AL_ML_DL/_Project/models/Types_resnet.h5')
intermediate_layer_model = Model(
    inputs=model.input, outputs=model.get_layer(layer_name).output)

# 读取音频数据


def load_data(data_path):
    wav, sr = librosa.load(data_path, sr=16000)
    # 切割特徵區塊
    intervals = librosa.effects.split(wav, top_db=20)
    # 讀取並儲存每一個區塊
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])

    # if len(wav_output) < 8000:
        #raise Exception("有效音频小于0.5s")

    wav_output = np.array(wav_output)
    ps = librosa.feature.melspectrogram(
        y=wav_output, sr=sr, hop_length=256).astype(np.float32)
    ps = ps[np.newaxis, ..., np.newaxis]
    return ps


def infer(audio_path):
    data = load_data(audio_path)
    feature = intermediate_layer_model.predict(data)
    return feature


def get_data_list(list_path, a1=0):
    List_Data = os.listdir(list_path)
    data_path = []
    for i in range(len(List_Data)):
        # fn
        s = List_Data[i]
        # if '-3' in s or a1: # 指定要存的檔案
        _s = s.split(".")
        if _s[1] == "wav":
            data_path.append(list_path + s)
    print("{}個檔案".format(len(data_path)))
    # 透過try跳過假路徑
    # 人1路徑+音檔 人2 .., 比較相似度回傳預測結果並儲存於字典
    data_path.sort()
    _sum = {}
    for person_1 in data_path:  # 讀路徑清單
        print("一輪開始", person_1)
        try:
            feature1 = infer(person_1)[0]  # 固定一個路徑並比較所有list
            for person_2 in data_path:
                # 實作不重覆組合排列
                if person_1 != person_2:
                    # split(".wav的上一層資料夾")
                    _p1 = person_1.split('_sum/')
                    _p2 = person_2.split('_sum/')
                    # 關鍵-每次比較前比較現有的list
                    if (_p2[1] + '相似度' + _p1[1]) in _sum:
                        continue
                    else:
                        feature2 = infer(person_2)[0]
                        # 对角余弦值
                        dist = np.dot(feature1, feature2) / \
                            (np.linalg.norm(feature1) * np.linalg.norm(feature2))
                        _sum[_p1[1] + '相似度' +
                             _p2[1]] = dist  # 字典儲存預測結果
                else:
                    print(end="")
                    #print("重複 %s == %s" % (person_1, person_2))
        except FileNotFoundError:
            print("不存在之檔案, 跳過本次循環")
            continue
        except Exception as e:
            print(end="")
        #print("一輪結束", person_1)
    return _sum


total_sum = get_data_list('E:/AL_ML_DL/_Project/audio_sets/__WAVE/_sum/')

# something=''
#a=lambda _retu:[_retu for _retu in something]
#x=lambda i:[i for i in ary]
ary = []
for i, d in enumerate(total_sum):
    _d = d.split(',')
    #print(_d[0]+" : "+str(total_sum[_d[0]]))
    ary.append(total_sum[_d[0]])
ary.sort()
ary.reverse()

print("原始預測總數{}".format(len(total_sum)))
print("重新排列總數{}".format(len(ary)))

al = []
for i in ary:
    _al = list(total_sum.keys())[list(total_sum.values()).index(i)], i
    al.append(_al)
# print(al)

with open('al.csv', 'w+', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['檔案', '預測值'])
    for k, v in enumerate(al):
        # print("k:{}\nv:{}".format(v[0],v[1]))
        # break
        writer.writerow([v[0], v[1]])
