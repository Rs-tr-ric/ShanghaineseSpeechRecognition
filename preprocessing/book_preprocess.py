import pypinyin
import re, random
from tqdm import tqdm

sh_corpus = '.\data\corpus.src'

def load_char_dict(corpus):
    char_dict = {}
    with open(corpus, encoding='utf-8') as file:
        for line in file:
            for char in line.strip():
                for p_list in pypinyin.pinyin(char, style=pypinyin.NORMAL, errors='ignor', heteronym=True):
                    for p in p_list:
                        if char_dict.get(p) is not None:
                            char_dict[p].add(char)
                        else:
                            char_dict[p] = set(char)

    return {key: list(char_dict[key]) for key in char_dict}

char_dict = load_char_dict(sh_corpus)

sents = []

def augmentation(sent):
    # sent = list(sent)
    # for _ in range(int(len(sent) / 3)):
    #     idx = random.randint(0, len(sent) - 1)
    #     rand_stat = random.random()
    #     if rand_stat < 0.2:
    #         sent.pop(idx)
    #     elif rand_stat < 0.4:
    #         sent.insert(idx, random.choice(sent))
    #     elif rand_stat < 0.8:
    #         p = pypinyin.pinyin(sent[idx], pypinyin.NORMAL, heteronym=True, errors='ignore')
    #         char_list = char_dict.get(random.choice(p[0])) if p else None
    #         if char_list is not None:
    #             sent[idx] = random.choice(char_list)
    #     elif rand_stat < 1.0:
    #         idx_2 = random.randint(0, len(sent) - 1)
    #         sent[idx], sent[idx_2] = sent[idx_2], sent[idx]

    psent = pypinyin.pinyin(sent, pypinyin.NORMAL, heteronym=False, errors='ignore')

    return ' '.join(random.choice(i) for i in psent)

for i in [6, 7, 8, *range(16, 25)]:
    with open('data\\books\\%d.txt' % i, encoding='utf-8') as file:
        for line in tqdm(file):
            sent = []
            for i in re.split(r'[？！…。—“”’‘●]+', line.strip()):
                if i.strip():
                    if len(i) <= 50:
                        sent.append(i)
                    else:
                        sent += [j.strip() for j in re.split('，', i) if j.strip() and len(i.strip()) <= 50]
            sents += sent

train_idx = random.sample(range(len(sents)), int(len(sents) / 20) * 19)
train_sents = [sents[idx] for idx in tqdm(train_idx)]
test_sents = [sents[idx] for idx in tqdm(set(range(len(sents))) - set(train_idx))]
test_sents, dev_sents = test_sents[:len(test_sents) // 2], test_sents[len(test_sents) // 2:]

with open('data\\corpus.books.src.train', 'w', encoding='utf-8') as file:
    file.write('\n'.join(augmentation(i) for i in tqdm(train_sents)))
with open('data\\corpus.books.tgt.train', 'w', encoding='utf-8') as file:
    # file.write('\n'.join(train_sents))
    file.write('\n'.join(augmentation(i) for i in tqdm(train_sents)))
with open('data\\corpus.books.src.test', 'w', encoding='utf-8') as file:
    file.write('\n'.join(augmentation(i) for i in tqdm(test_sents)))
with open('data\\corpus.books.tgt.test', 'w', encoding='utf-8') as file:
    # file.write('\n'.join(test_sents))
    file.write('\n'.join(augmentation(i) for i in tqdm(test_sents)))
with open('data\\corpus.books.src.dev', 'w', encoding='utf-8') as file:
    file.write('\n'.join(augmentation(i) for i in tqdm(dev_sents)))
with open('data\\corpus.books.tgt.dev', 'w', encoding='utf-8') as file:
    # file.write('\n'.join(dev_sents))
    file.write('\n'.join(augmentation(i) for i in tqdm(dev_sents)))
    