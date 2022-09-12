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

# wav內容


def wavState(path, filename, wav):
    # 取得音檔的屬性d
    #import wave
    # path
    readFile = path + filename + wav
    # 秒數=總禎除頻率
    f = wave.open(readFile)
    print(f'取樣頻率={f.getframerate()}, 幀數={f.getnframes()}, ' +
          f'聲道={f.getnchannels()}, 精度={f.getsampwidth()}, ' +
          f'檔案秒數={f.getnframes() / f.getframerate():.2f}')
    f.close()


path = '../../../AudioClass/AudioClassification-Tensorflow-master/dataset/WavCut/'
# 指定檔案
filename = '每_0_秒cut_佳_手動裁切_佳'
wav = '.wav'

# Show wav state
wavState(path, filename, wav)

# 音檔路徑
filename_load = path+filename+wav

# 總樣本與採樣率==總秒數
data, sr = librosa.load(filename_load)

# 讀取mel頻譜圖
spec = librosa.feature.melspectrogram(y=data, sr=sr)

# 顯示mel頻譜圖
db_spec = librosa.power_to_db(spec, ref=np.max,)
librosa.display.specshow(db_spec, y_axis='mel', x_axis='s', sr=sr)
plt.colorbar()

# show 音檔
IPython.display.Audio(path + filename + wav, autoplay=False)

# 頻率圖
librosa.display.waveshow(data, sr)

wavfile = AudioSegment.from_wav(filename_load)

# 每秒單位的檔案
n = 0
for n in range(3):
    wavfile[(n*10000):((n+1)*10000)].export('WavCut/' +
                                            filename+'_第'+str(n)+'秒cut'+wav, format="wav")
    # 大於60秒就不要再切了, 麻煩, 所以一開始讀檔的時候要看長度
    # if(n>=10):
    #wavfile[:(n+1)*1000].export('WavCut/'+filename+'_cut'+wav, format="wav")
    # break
print(n)

# 讀取切割的檔案
filename_load = 'WavCut/'+filename+'_第0秒cut'+wav
data, sr = librosa.load(filename_load)

# 讀取頻譜圖
spec = librosa.feature.melspectrogram(y=data, sr=sr)

# 顯示頻譜圖
db_spec = librosa.power_to_db(spec, ref=np.max,)
librosa.display.specshow(db_spec, y_axis='mel', x_axis='s', sr=sr)
plt.colorbar()
IPython.display.Audio(filename_load, autoplay=False)

# 讀取切割的檔案
filename_load = 'WavCut/'+filename+'_第1秒cut'+wav
data, sr = librosa.load(filename_load)

# 讀取頻譜圖
spec = librosa.feature.melspectrogram(y=data, sr=sr)

# 顯示頻譜圖
db_spec = librosa.power_to_db(spec, ref=np.max,)
librosa.display.specshow(db_spec, y_axis='mel', x_axis='s', sr=sr)
plt.colorbar()
IPython.display.Audio(filename_load, autoplay=False)

# 讀取切割的檔案
filename_load = 'WavCut/'+filename+'_第2秒cut'+wav
data, sr = librosa.load(filename_load)

# 讀取頻譜圖
spec = librosa.feature.melspectrogram(y=data, sr=sr)

# 顯示頻譜圖
db_spec = librosa.power_to_db(spec, ref=np.max,)
librosa.display.specshow(db_spec, y_axis='mel', x_axis='s', sr=sr)
plt.colorbar()
IPython.display.Audio(filename_load, autoplay=False)

'''
# wavCut的原始範例
from pydub import AudioSegment
# 音频的原始文件record.wav
filename='record'
# 读取音频文件
wav = AudioSegment.from_wav(filename+'.wav')
# 读取前45分钟的音频并保存在record_cut1.wav中
wav[:45*60*1000].export(filename+'_cut1.wav', format="wav")
# 读取45分钟以后的音频并保存在record_cut2.wav中
wav[45*60*1000:].export(filename+'_cut2.wav', format="wav")
# 來源檔案
wavfile = AudioSegment.from_wav(newFile_wav)
'''
# 手動裁切
wavfile[(1.0*1000):(1.8*1000)].export('WavCut/' +
                                      filename+'_手動裁切_勲'+wav, format="wav")

filename_load = 'WavCut/'+filename+'_手動裁切_勲'+wav
data, sr = librosa.load(filename_load)

# 讀取頻譜圖
spec = librosa.feature.melspectrogram(y=data, sr=sr)

# 顯示頻譜圖
db_spec = librosa.power_to_db(spec, ref=np.max,)
librosa.display.specshow(db_spec, y_axis='mel', x_axis='s', sr=sr)
plt.colorbar()
IPython.display.Audio(filename_load, autoplay=False)

librosa.display.waveshow(data, sr)

# 和音與敲擊音分離(Harmonic/Percussive Separation)
y_h, y_p = librosa.effects.hpss(data)
spec_h = librosa.feature.melspectrogram(y_h, sr=sr)
spec_p = librosa.feature.melspectrogram(y_p, sr=sr)
db_spec_h = librosa.power_to_db(spec_h, ref=np.max)
db_spec_p = librosa.power_to_db(spec_p, ref=np.max)

plt.subplot(2, 1, 1)
librosa.display.specshow(db_spec_h, y_axis='mel', x_axis='s', sr=sr)
plt.colorbar()

plt.subplot(2, 1, 2)
librosa.display.specshow(db_spec_p, y_axis='mel', x_axis='s', sr=sr)
plt.colorbar()

plt.tight_layout()

# 播放和音(y_h)或敲擊音(y_p)或原檔(data)
IPython.display.Audio(data=y_h, rate=sr)

IPython.display.Audio(data=y_p, rate=sr)

IPython.display.Audio(data=data, rate=sr)
