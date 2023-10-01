from load_model import load_wav2py_model, load_py2py_model, load_py2text_model

def main(model_sequence, waves):
    contents = waves
    for model in model_sequence:
        contents = model(contents)

    for idx, text in enumerate(contents):
        print('%03d' % idx, text)

if __name__ == '__main__':
    wav2py = load_wav2py_model()
    py2py = load_py2py_model()
    py2text = load_py2text_model()

    main([wav2py, py2py, py2text], 
         [('G:\\ShanghaineseSpeechRecognitionDatasets\\'
           'Shanghai_Dialect_Dict\\Split_WAV1\\%d.wav' % i)
           for i in range(100, 150)])