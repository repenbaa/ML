# 音檔
import IPython
# 取得音檔的屬性
import wave
# cut
from pydub import AudioSegment
# draw
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import math
import os

# 被切割的音檔路徑
path = 'C:/Users/user/Desktop/VoiceprintRecognition-Tensorflow-master/VoiceprintRecognition-Tensorflow-master/audio_sets/__FROM/_sum/'
# 切割完後存到新的路徑
new_path = 'C:/Users/user/Desktop/VoiceprintRecognition-Tensorflow-master/VoiceprintRecognition-Tensorflow-master/audio_sets/__WAVE/_sum/'
# 合併路徑與對象並自動切割等於其總數(含名稱)
audios = os.listdir(path)

# 多少音檔執行多少次
for i in range(len(audios)):
    # cut
    wavfile = AudioSegment.from_wav(path+audios[i])
    # 初始化:每兩秒切割一次
    n = 0
    # 2.04秒切割一次會有除不盡問題所以偶數+0, 奇數+1; =測試=> 3
    k = n+3
    # 總次數
    total = 0
    # 音檔總秒數
    t = round(librosa.get_duration(filename=path+audios[i]), 0)

    # 決定要切幾段; 2.04秒切割一次會有除不盡問題所以偶數+0, 奇數+1
    if t % 3 != 0:
        # +1得到偶數次
        total = math.floor(int(t+1)/3)
        # print('if%d'%total)
    else:
        # +0得到偶數次
        total = math.floor(int(t+0)/3)
        # print('else%d'%total)
    print('對{}切成{}塊總長{}從{}開始到{}為止'.format(audios[i], total, t, n, k))
    # break
    # 每段要多長
    for x in range(total):
        # while total:
        wavfile[(n*1000):(k*1000)].export(new_path +
                                          str(n)+"-"+audios[i], format="wav")
        # print(n,k)
        n += 3
        k = n+3
        # total-=1
        # print(total)
