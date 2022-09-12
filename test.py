import os
import random
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import math
layer_name = 'global_max_pooling2d'
model = tf.keras.models.load_model('./models/Types_resnet.h5')
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


def get_data_list(list_path, a1):
    List_Data = os.listdir(list_path)
    data_path = []
    for i in range(len(List_Data)):
        # fn
        s = List_Data[i]
        # if '-3' in s or a1: # 指定要存的檔案
        data_path.append(list_path + s)
    print(len(data_path))
    # 透過try跳過假路徑
    # 人1路徑+音檔 人2 .., 比較相似度回傳預測結果並儲存於字典
    _sum = {}
    for person_1 in data_path:  # 讀路徑清單
        print("一輪開始", person_1)
        try:
            feature1 = infer(person_1)[0]  # 固定一個路徑並比較所有list
            for person_2 in data_path:
                # 重複不比
                if person_1 != person_2:
                    feature2 = infer(person_2)[0]
                    # 对角余弦值
                    dist = np.dot(feature1, feature2) / \
                        (np.linalg.norm(feature1) * np.linalg.norm(feature2))
                    _sum[person_1[-14:] + ' 相似度 ' +
                         person_2[-14:]] = dist  # 字典儲存預測結果
                else:
                    print("%s == %s" % (person_1, person_2))
        except FileNotFoundError:
            print("不存在之檔案, 跳過本次循環")
            continue
        print("一輪結束", person_1)
    return _sum


total_sum = get_data_list('./audio_sets/__WAVE/_sum/', '-0')
print(total_sum)


"""
'/wavForm/1.wav 相似度 /wavForm/2.wav': 0.6502193,
'/wavForm/1.wav 相似度 /wavForm/3.wav': 0.6505083,
'/wavForm/1.wav 相似度 /wavForm/7.wav': 0.68979335,
'/wavForm/2.wav 相似度 /wavForm/3.wav': 0.6546757,
'/wavForm/0.wav 相似度 /wavForm/2.wav': 0.5201596,
'/wavForm/2.wav 相似度 /wavForm/7.wav': 0.5954958,
'/wavForm/3.wav 相似度 /wavForm/7.wav': 0.5812369,

'/wavForm/0.wav 相似度 /wavForm/1.wav': 0.29091185,
'/wavForm/0.wav 相似度 /wavForm/3.wav': 0.46422616,
'/wavForm/0.wav 相似度 /wavForm/4.wav': 0.3791647,
'/wavForm/0.wav 相似度 /wavForm/6.wav': 0.22999106,
'/wavForm/0.wav 相似度 /wavForm/5.wav': 0.20531543,
'/wavForm/0.wav 相似度 /wavForm/7.wav': 0.26977596,
'/wavForm/1.wav 相似度 /wavForm/4.wav': 0.3845537,
'/wavForm/1.wav 相似度 /wavForm/6.wav': 0.32802272,
'/wavForm/1.wav 相似度 /wavForm/5.wav': 0.22208509,
'/wavForm/2.wav 相似度 /wavForm/4.wav': 0.45653847,
'/wavForm/2.wav 相似度 /wavForm/6.wav': 0.32722226,
'/wavForm/2.wav 相似度 /wavForm/5.wav': 0.25329474,
'/wavForm/3.wav 相似度 /wavForm/4.wav': 0.442932,
'/wavForm/3.wav 相似度 /wavForm/6.wav': 0.2905521,
'/wavForm/3.wav 相似度 /wavForm/5.wav': 0.17504582,
'/wavForm/4.wav 相似度 /wavForm/6.wav': 0.36295635,
'/wavForm/4.wav 相似度 /wavForm/5.wav': 0.29075405,
'/wavForm/4.wav 相似度 /wavForm/7.wav': 0.44140747,
'/wavForm/6.wav 相似度 /wavForm/5.wav': 0.3610018,
'/wavForm/6.wav 相似度 /wavForm/7.wav': 0.4891555,
'/wavForm/5.wav 相似度 /wavForm/7.wav': 0.21941252
"""
