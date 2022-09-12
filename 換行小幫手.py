import os

txt='../../新文字文件.txt'
with open(txt, 'r', encoding='utf-8') as f:
    for i in f:
        cnt=0
        with open('../../test.txt', 'a+', encoding='utf-8') as fw:
            for j in range(len(i)):
                if i[j]=='，' or i[j]=='。' or i[j]=='「' or i[j]=='：' or i[j]=='」' or i[j]==',' or i[j]=='（'  or i[j]=='）'  or i[j]=='‧'  or i[j]=='？' or i[j]==' ' or i[j]=='．':
                    continue
                elif cnt<=9:
                    #print("..",cnt)
                    fw.write(i[j])
                    cnt+=1
                else:
                    fw.write(i[j])
                    fw.write('\n')
                    cnt=0