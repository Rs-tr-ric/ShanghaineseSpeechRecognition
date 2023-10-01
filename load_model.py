from paddlespeech.cli.text import TextExecutor
from paddlespeech.cli.asr import ASRExecutor
from torch.nn.utils.rnn import pad_sequence
from beam_decoder import beam_search
from transformer import make_model
import sentencepiece as spm
import numpy as np
import pypinyin
import torch


def load_wav2py_model(model='conformer_wenetspeech'):
    asr = ASRExecutor()
    asr.disable_task_loggers()

    def _model(wav_files):
        pinyins = []
        for audio_file in wav_files:
            asr_result = asr(
                audio_file=audio_file, 
                model=model, 
                lang='zh', 
                device='gpu:0', 
                force_yes=True, 
            )
            
            pinyin = pypinyin.lazy_pinyin(
                hans=asr_result, 
                style=pypinyin.NORMAL, 
                errors='ignore')
            
            pinyins.append(' '.join(pinyin))

        return pinyins
        
    return _model


def load_py2py_model():
    py2py = make_model(src_vocab=670, tgt_vocab=675, d_model=128, h=8, N=6, d_ff=512)
    py2py.load_state_dict(torch.load('models/py2py_model_2.pth'))
    py2py.to('cuda:0')

    py2py_src_tokenizer = spm.SentencePieceProcessor()
    py2py_src_tokenizer.Load('models/vocab/sh.src.model')
    py2py_tgt_tokenizer = spm.SentencePieceProcessor()
    py2py_tgt_tokenizer.Load('models/vocab/sh.tgt.model')
    
    def _model(pinyins):
        with torch.no_grad():
            tokens = [[2, *py2py_src_tokenizer.EncodeAsIds(sent), 3] for sent in pinyins]
            pinyins = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tokens], batch_first=True, padding_value=0).to('cuda:0')
            mask = (pinyins != 0).unsqueeze(-2)
            decode_result, _ = beam_search(py2py, pinyins, mask, 60, 0, 2, 3, 12, 'cuda:0')
            decode_result = [h[0] for h in decode_result]
            pinyins = py2py_tgt_tokenizer.Decode(decode_result, num_threads=16)
        
        return pinyins
    
    return _model

def load_py2text_model():
    py2txt = make_model(src_vocab=697, tgt_vocab=32768, d_model=128, h=8, N=6, d_ff=512)
    py2txt.load_state_dict(torch.load('models/py2text_model_2.pth'))
    # py2txt.load_state_dict(torch.load('models/pre-trained_model_py2text.pth'))
    py2txt.to('cuda:0')
    py2txt_src_tokenizer = spm.SentencePieceProcessor()
    py2txt_src_tokenizer.Load('models/vocab/ch.src.model')
    py2txt_tgt_tokenizer = spm.SentencePieceProcessor()
    py2txt_tgt_tokenizer.Load('models/vocab/ch.tgt.model')

    def _model(pinyins):
        tokens = [[2, *py2txt_src_tokenizer.EncodeAsIds(sent), 3] for sent in pinyins]
        pinyins = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tokens], batch_first=True, padding_value=0).to('cuda:0')
        mask = (pinyins != 0).unsqueeze(-2)
        decode_result, _ = beam_search(py2txt, pinyins, mask, 60, 0, 2, 3, 12, 'cuda:0')
        decode_result = [h[0] for h in decode_result]
        texts = py2txt_tgt_tokenizer.Decode(decode_result, num_threads=16)

        return texts
    
    return _model

def load_punc_model(model='ernie_linear_p7_wudao'):
    punc = TextExecutor()
    punc.disable_task_loggers()

    def _model(texts):
        return [punc(
            text=t, 
            task='punc', 
            model=model, 
            lang='zh', 
            device='cuda:0', 
        ) for t in texts]
        
    return _model


if __name__ == '__main__':
    from tqdm import tqdm

    with open('exp\\test.input', 'r', encoding='utf-8') as file:
        input_set = file.read()
    with open('exp\\test.ground_truth', 'r', encoding='utf-8') as file:
        ground_truth = file.read()

    py2py_model = load_py2py_model()
    py2text_model = load_py2text_model()
    text_model = load_punc_model()

    dataset = list(map(str.strip, input_set.split('\n')))
    ground_truth = list(map(str.strip, ground_truth.split('\n')))

    file = open('exp\\result.txt', 'w', encoding='utf-8')
    pinyins = py2py_model(dataset)
    texts = py2text_model(pinyins)
    # texts = text_model(texts)
    for idx, (a, b, c, d) in enumerate(zip(dataset, pinyins, texts, ground_truth)):
        print('%03d' % idx, a, file=file)
        print('   ', b, file=file)
        print('   ', c, file=file)
        print('   ', d, file=file)

    # for ext in ('dev', 'train', 'test'):
    #     with open(f'exp\\corpus.src.{ext}', 'r', encoding='utf-8') as file:
    #         lines = file.readlines()
    #         for idx in tqdm(range(0, len(lines), 512)):
    #             decode_result = py2py_model(lines[idx:idx + 512])
    #             for i in range(idx, min(idx + 512, len(lines))):
    #                 lines[i] = decode_result[i - idx]
        
    #     with open(f'exp\\corpus.sh.src.{ext}', 'w', encoding='utf-8') as file:
    #         file.write('\n'.join(lines))