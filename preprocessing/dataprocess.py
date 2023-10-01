import os, re, random, pypinyin
from tqdm import tqdm

test, train, dev = [[], []], [[], []], [[], []]

def init(root):
    global TEST, TRAIN, DEV

    total = []
    for root, _, files in os.walk(root):
        for file in files:
            # if 'wav' in file and (not 'useless' in root or 'ximalaya' in root) and not 'splited' in root:
            if 'wav' in file and not 'useless' in root:
                total.append(os.path.join(root, file))
                
    c = random.sample(range(len(total)), int(len(total) / 50))
    TRAIN, tmp = [], []
    for idx, file in enumerate(total):
        if idx in c:
            tmp.append(file)
        else:
            TRAIN.append(file)

    split = int(len(tmp) / 2)
    TEST, DEV = tmp[:split], tmp[split:]

    assert len(DEV) + len(TEST) + len(TRAIN) == len(total)

def copytxt2file(root, prompt_path, f):
    global test, train, dev, length_difference

    for line in tqdm(list(f)):
        wav, text = line[::-1].split('|', maxsplit=1)
        wav, text = wav[::-1].strip(), text[::-1].strip()
        text = pypinyin.pinyin(text, pypinyin.NORMAL, heteronym=True, errors='ignore')
        text = ' '.join(random.choice(i) for i in text)
        txt = os.path.splitext(wav)[0] + '.txt'
        txt = os.path.join(prompt_path, txt)
        wav = os.path.join(root, wav)
        with open(txt, 'r', encoding='utf-8') as f:
            text_pmt = ''.join(line.strip() for line in f.readlines())
            text_pmt = pypinyin.pinyin(text_pmt, pypinyin.NORMAL, heteronym=True, errors='ignore')
            text_pmt = ' '.join(random.choice(i) for i in text_pmt)

        if len(text_pmt) <= 6:
            ext = 'train'
        elif wav in TRAIN:
            if text_pmt in dev[1]:
                ext = 'dev'
            elif text_pmt in test[1]:
                ext = 'test'
            else:
                ext = 'train'
        elif wav in DEV:
            if text_pmt in test[1]:
                ext = 'test'
            elif text_pmt in train[1]:
                ext = 'train'
            else:
                ext = 'dev'
        elif wav in TEST:
            if text_pmt in dev[1]:
                ext = 'dev'
            elif text_pmt in train[1]:
                ext = 'train'
            else:
                ext = 'test'
        
        eval(f'{ext}[0].append(text)')
        eval(f'{ext}[1].append(text_pmt)')

def merge(root, file):
    # if 'transcribed_paddle' in file and not 'splited' in root:
    if 'transcribed_paddle' in file and not 'useless' in root:
        file = os.path.join(root, file)
        prompt_path = os.path.join(root, '..', 'Split_PROMPT')
        with open(file, 'r', encoding='utf-8') as f:
            copytxt2file(root, prompt_path, f)

def merge2file(root):
    for root, _, files in os.walk(root):
        for file in files:
            merge(root, file)


if __name__ == '__main__':
    init('G:\\ShanghaineseSpeechRecognitionDatasets')
    merge2file('G:\\ShanghaineseSpeechRecognitionDatasets')

    with open('data\\corpus.src.dev', 'w', encoding='utf-8') as f:
        f.write('\n'.join(dev[0]))
    with open('data\\corpus.tgt.dev', 'w', encoding='utf-8') as f:
        f.write('\n'.join(dev[1]))
    with open('data\\corpus.src.test', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test[0]))
    with open('data\\corpus.tgt.test', 'w', encoding='utf-8') as f:
        f.write('\n'.join(test[1]))
    with open('data\\corpus.src.train', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train[0]))
    with open('data\\corpus.tgt.train', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train[1]))

    print('train:', len(train[0]))
    print('test:', len(test[0]))
    print('dev:', len(dev[0]))

    with open('data\\corpus.tgt.test', encoding='utf-8') as file:
        test = file.readlines()

    with open('data\\corpus.tgt.train', encoding='utf-8') as file:
        train = file.readlines()
    
    for i in test:
        if i in train:
            print(i.strip())
