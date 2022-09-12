import IPython
from pydub import AudioSegment
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

path = '../VoiceprintRecognition-Tensorflow-master/audio_sets/__FROM/_4/'
# 指定檔案
filename = 'ex-4'
wav = '.wav'
# 音檔路徑
filename_load = path+filename+wav

wavfile = AudioSegment.from_wav(filename_load)

# 直接讀
data, sr = librosa.load(filename_load)

# 設定頻譜圖
spec = librosa.feature.melspectrogram(y=data, sr=sr)

# 顯示頻譜圖
db_spec = librosa.power_to_db(spec, ref=np.max,)
librosa.display.specshow(db_spec, y_axis='mel', x_axis='s', sr=sr)
plt.colorbar()
IPython.display.Audio(filename_load, autoplay=False)

# 砍完才讀
# 手動裁切
wavfile[:20.0*1000].export(path+filename+'_test'+wav, format="wav")

data, sr = librosa.load(path+filename+'_test'+wav)

# 設定頻譜圖
spec = librosa.feature.melspectrogram(y=data, sr=sr)

# 顯示頻譜圖
db_spec = librosa.power_to_db(spec, ref=np.max,)
librosa.display.specshow(db_spec, y_axis='mel', x_axis='s', sr=sr)
plt.colorbar()
IPython.display.Audio(path+filename+'_test'+wav, autoplay=False)

