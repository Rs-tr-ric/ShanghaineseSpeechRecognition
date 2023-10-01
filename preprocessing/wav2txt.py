import os
import tqdm
import typing
from multiprocessing.pool import ThreadPool

def file2txt(root: str, file: str, model: typing.Callable, t: tqdm.tqdm) -> str:
    if 'wav' in file:
        try:
            text = model(os.path.join(root, file)).strip()
            t.set_postfix_str(text, refresh=False)
            t.update()
            return text + '|' + file + '\n'
        except Exception as e:
            print(type(e).__name__, e, sep=': ')
            print(file)

def main(root: str, model: typing.Callable) -> None:
    for root, _, files in os.walk(root):
        t = tqdm.tqdm(files)
        with ThreadPool(4) as pool:
            threads = [pool.apply_async(file2txt, 
                (root, file, model, t)) for file in files]
            
            pool.close()
            pool.join()

            lines = [thread.get() for thread in threads]
            lines = [line for line in lines if line and line.strip()]

        if len(lines):
            with open(os.path.join(root, 'transcribed_paddle.txt'), 'a', encoding='utf-8') as txt:
                txt.write(''.join(lines))

def find_omitted(root: str, model: typing.Callable) -> list[str]:
    with open(os.path.join(root, 'transcribed_paddle.txt')) as file:
        transcribed_files = [line.split('|')[-1].strip() for line in file]
    
    files = [file for file in os.listdir(root) if file.split('.')[-1] == 'wav']
    
    result = []
    for file in files:
        if not file in transcribed_files:
            result.append(file)

    t = tqdm.tqdm(result)
    result = [file2txt(root, file, model, t) for file in result]

    return result

def ximalaya_merge(transcriped_paddle: str, model: typing.Callable) -> None:
    file_dict = {}
    with open(transcriped_paddle, encoding='utf-8') as file:
        for line in file:
            text, wav = line.strip().split('|')
            father_wav, wav = wav.split('.')[0].split('_')
            father_wav, wav = int(father_wav), int(wav)
            if file_dict.get(father_wav) is None:
                file_dict[father_wav] = {wav: text}
            else:
                file_dict[father_wav][wav] = text

    for key in tqdm.tqdm(file_dict):
        file_dict[key] = sorted(file_dict[key].items(), key=lambda x: x[0])
        file_dict[key] = ''.join(i[1] for i in file_dict[key] if len(i[1]) < 100)
        # file_dict[key] = model(file_dict[key])
    
    with open(transcriped_paddle + '.txt', 'w', encoding='utf-8') as file:
        for key in file_dict:
            file.write(file_dict[key] + '|' + '%d.wav' % key + '\n')

if __name__ == '__main__':
    # from load_speech_model import load_asr_model
    from speech_model import load_text_model
    
    root = 'G:\\ShanghaineseSpeechRecognitionDatasets\\useless\\ximalaya\\Split_WAV\\splited\\transcribed_paddle.txt'

    # model = load_asr_model()
    model = load_text_model()
    # main(root, model)
    # print(find_omitted(root, model))
    ximalaya_merge(root, model)