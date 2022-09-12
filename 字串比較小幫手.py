import os
import winsound
from tqdm import tqdm
frequency = 2000
duration = 1000
winsound.Beep(frequency, duration)
path = 'E:/AL_ML_DL/DataSets/TypeSets/ST-CMDS-20170001_1-OS/'
# file='E:/AL_ML_DL/DataSets/TypeSets/ST-CMDS-20170001_1-OS/20170001P00001A0001.txt'

_list = os.listdir(path)

r, _f = '', ''
for i in range(len(_list)):
    with open(path+_list[i], 'r', encoding='utf-8') as fr:
        r = fr.read()
    for j in tqdm(range(len(_list))):
        if '.txt' in _list[j]:
            with open(path+_list[j], 'r', encoding='utf-8') as f:
                _f = f.read()
            if _list[i] is _list[j]:
                continue
            elif r == _f:
                print("{} equal {}, \n搜尋完畢".format(r, _f))
                break
print("{} !equal {}, \n搜尋完畢".format(r, _f))

frequency = 2000
duration = 1000
winsound.Beep(frequency, duration)
