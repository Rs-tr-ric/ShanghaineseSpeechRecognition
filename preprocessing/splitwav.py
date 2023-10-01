import wave, os

root = 'G:\\ShanghaineseSpeechRecognitionDatasets\\useless\\ximalaya\\Split_WAV'

def split(root, audio):
    wf = wave.open(os.path.join(root, audio), "rb")
    nchannels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    nframes = wf.getnframes()

    duration = nframes / framerate
    print("duration: %.2fs" % duration)

    # 设置分割的长度为2s
    length = 10 * framerate
    start = 0
    i = 0

    while start < nframes:
        # 截取片段
        wf.setpos(start)
        data = wf.readframes(length)

        # 保存为新文件
        new_wf = wave.open(os.path.join(root, 'splited', '%s_%d.wav' %(audio.split('.')[0], i)), "wb")
        new_wf.setnchannels(nchannels)
        new_wf.setsampwidth(sampwidth)
        new_wf.setframerate(framerate)
        new_wf.writeframes(data)
        new_wf.close()
        
        # 更新起始位置
        start += length
        i += 1

def main(root):
    for file in os.listdir(root):
        if file.split('.')[-1] == 'wav':
            split(root, file)

if __name__ == '__main__':
    main(root)