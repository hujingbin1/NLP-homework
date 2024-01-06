import pyaudio
import time
import wave
import _thread
from aip import AipSpeech


class Recorder():
    def __init__(self, chunk=1024, channels=1, rate=16000):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []


    def start(self):
        _thread.start_new_thread(self.__recording, ())

    def __recording(self):
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        frames_per_buffer=self.CHUNK)
        while (self._running):
            data = stream.read(self.CHUNK)
            self._frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self._running = False

    def save1(self, filename):
        p = pyaudio.PyAudio()
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()

    def record(self):
        print('请按下回车键开始录音：')
        a = input()
        if str.__len__(a) == 0:
            begin = time.time()
            print("Start recording")
            self.start()
            print("请按下回车键结束录音：")
            b = input()
            if str.__len__(b) == 0:
                print("Stop recording")
                self.stop()
                fina = time.time()
                t = fina - begin
                print('录音时间为%ds' % t)

    def positive(self):
        print('开始样本的录制')
        self.record()
        self.save1("test2.wav")


def save2(result, filename):
    if not filename.endswith(".txt"):
        filename = filename + ".txt"
    wf = open(filename, encoding='utf-8', mode='a+')
    wf.writelines(result)
    wf.writelines(['\n'])
    wf.close()


def ASR():
    rec = Recorder()
    rec.positive()
    print('识别中......')
    APP_ID = '34026373'  # 自己创建的百度智能云语音识别应用的APPID
    API_KEY = 't7VrCwh6z5oPz6X5oWE2GxDp'  # 自己创建的百度智能云语音识别应用的API-key
    SECRET_KEY = 'LFtPnEaCreyBqWn7IM0fY9FBR9XEjQu4'  # 自己创建的百度智能云语音识别应用的secret_key
    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    # 读取文件
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    # 识别本地文件
    res = client.asr(get_file_content('test2.wav'), 'wav', 16000, {'dev_pid': 1536, })
    if res and res['err_no'] == 0:
        result = res['result'][0]
        print("识别结果为：")
        print(result)
        result2 = res['result'][0]
        save2(result2, 'wenben.txt')
    return result
