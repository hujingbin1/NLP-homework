# -*- coding: utf-8 -*-
import os
from aip import AipSpeech
import pygame
import sys
import STT
from predict import predict

""" 你的 APPID AK SK """
APP_ID = '41547735'                            # 自己的app——id
API_KEY = 'AGIwnVVCO7vZOh9AbQHXbpLN'             # 自己的API_KEY
SECRET_KEY = 'XEmjdfbVKoZiNKQBFU1x4Eu2yunx7OMa'   # SECRET_KEY

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

s=STT.ASR()
s=predict(s)
print(s)
result = client.synthesis(s, 'zh', 1, {       # zh代表中文
    'vol': 5,
})

# 返回的是一个音频流，需要保存成mp3文件
# 识别正确返回语音二进制 错误则返回dict 参照下面错误码
if not isinstance(result, dict):
    with open('audio2.mp3', 'wb') as f:        # 创建mp3文件并具有写权限，用二进制的方式打开
        f.write(result)

# 使用pygame播放音频
pygame.mixer.init()
pygame.mixer.music.load('audio2.mp3')
pygame.mixer.music.play()
while pygame.mixer.music.get_busy():
    continue