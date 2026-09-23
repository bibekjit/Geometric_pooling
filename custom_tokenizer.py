import numpy as np
import pickle
from tqdm.auto import tqdm
import json
from collections import Counter
import string
import re
from string import punctuation



class BytePairEncodingTokenizer:

    def __init__(self, maxlen=None, vocab_size=10000, new_special_tokens=[], hindi_encoding=False):

        self.special_toks = ['<pad>', '<cls>', '<sep>', '<unk>', '<mask>'] + new_special_tokens
        self.w2i = {}
        self.i2w = {}
        self.vocab_size = vocab_size
        self.tokens = {}
        self.tok_freq = None
        self.words = {}
        self.vocab = {}
        self.subwords = {}
        self.maxlen = maxlen
        self.encoding_map = {}
        self.hindi_encoding = hindi_encoding

        #with open('char_table.json','r') as f:
         #   self.base_akshar = json.load(f)

        #self.base_akshar = list(self.base_akshar.values())

    def __call__(self, text, batch=10000, max_word_count=300000):

    # get corpus

        self.tok_freq = Counter()

        steps = len(text) // batch if len(text) % batch == 0 else len(text) // batch + 1

        print("collecting word count...")

        for i in tqdm(range(steps)):

            sents = ' '.join(text[i * batch: (i + 1) * batch]).split()

            self.tok_freq.update(sents)

        self.tok_freq = {k: v for (k, v) in self.tok_freq.items()
                         if (k not in string.punctuation) and k not in self.special_toks}

        self.tok_freq = dict(sorted(self.tok_freq.items(), key=lambda x: x[1],
                                    reverse=True)[:max_word_count])

        self.i2w = {i: w for i, w in enumerate(self.tok_freq)}




    def split_chars(self,x):

        if len(x) > 1 and x[0] in ('#',"_"):
            pattern = r'(?:_[A-Za-z]+)+#[A-Za-z]+|[#*][A-Za-z]+|_[A-Za-z]+'
            # pattern = r'[#_*][^#_*]+'
            pattern = re.compile(pattern)
            chars = re.findall(pattern, x)
            return chars

        else:
            return list(x)

    # def _split_to_encoded_units(self, word):
    #     return re.findall(r'[#_+][^#_+]+', word)
    #
    # def _split_to_units(self, word):
    #     return list(word)

    def train(self, iterations=15000):

        fin_sw = {x: self.split_chars(x) + ['+'] for x in self.tok_freq}

        # if self.hindi_encoding:
        #     fin_sw = {x: self._split_to_encoded_units(x) + ['+'] for x in self.tok_freq}
        # else:
        #     fin_sw = {x: self._split_to_units(x) + ['+'] for x in self.tok_freq}

        pair_count = {}

        print("collecting pair count...")

        for idx, (word, count) in enumerate(tqdm(self.tok_freq.items())):
            sw = fin_sw[word]



            # Use a simple range instead of enumerate since we just need adjacent indices
            for i in range(len(sw) - 1):
                x = (sw[i], sw[i + 1])

                if len(x) > 1:
                    if x not in pair_count:
                        pair_count[x] = [count, {idx}]  # Use a set to track word indices
                    else:
                        pair_count[x][0] += count
                        pair_count[x][1].add(idx)

        print("training tokenizer...")

        for _ in tqdm(range(iterations)):

            if not pair_count:
                break  # Break if there are no more pairs to merge

            top_pair, meta = max(pair_count.items(), key=lambda x: x[1][0])
            freq, w_indices = meta[0], meta[1]
            new_char = ''.join(top_pair)

            suff_count = 0
            pref_count = 0

            # Iterate over a list copy of the set, as the set may change
            for word_idx in list(w_indices):
                word = self.i2w[word_idx]
                count = self.tok_freq[word]

                if word not in fin_sw:
                    continue

                sw = fin_sw[word]

                if new_char == word:
                    del fin_sw[word]
                    self.vocab['--' + new_char.replace('+', '')] = count
                    continue

                # The dynamic scanning loop replacing the brittle index caching
                i = 0
                while i < len(sw) - 1:
                    if (sw[i], sw[i + 1]) == top_pair:

                        # 1. Deduct counts for the old adjacent pairs that are destroyed by the merge
                        if i > 0:
                            x_left = (sw[i - 1], sw[i])
                            if x_left in pair_count:
                                pair_count[x_left][0] -= count
                                if pair_count[x_left][0] <= 0:
                                    del pair_count[x_left]

                        if i + 2 < len(sw):
                            x_right = (sw[i + 1], sw[i + 2])
                            if x_right in pair_count:
                                pair_count[x_right][0] -= count
                                if pair_count[x_right][0] <= 0:
                                    del pair_count[x_right]

                        # 2. MERGE the characters in the array
                        sw = sw[:i] + [new_char] + sw[i + 2:]

                        # 3. Add counts for the NEW adjacent pairs formed by the merge
                        if i > 0:
                            x_left_new = (sw[i - 1], sw[i])
                            if x_left_new not in pair_count:
                                pair_count[x_left_new] = [0, set()]
                            pair_count[x_left_new][0] += count
                            pair_count[x_left_new][1].add(word_idx)

                        if i + 1 < len(sw):
                            x_right_new = (sw[i], sw[i + 1])
                            if x_right_new not in pair_count:
                                pair_count[x_right_new] = [0, set()]
                            pair_count[x_right_new][0] += count
                            pair_count[x_right_new][1].add(word_idx)

                    else:
                        i += 1

                fin_sw[word] = sw

                # Replicate your prefix and suffix counting on the newly merged word array
                if len(sw) > 0:
                    if new_char == sw[0]:
                        pref_count += 1
                    elif len(sw) > 1 and new_char in sw[1:]:
                        suff_count += 1

                    if sw[0] == word:
                        del fin_sw[word]
                        self.vocab['--' + word.replace('+', '')] = count

            # if new_char.replace("+", '') in self.base_akshar:
            #     continue


            if pref_count > 0:
                new_char_pref = '--' + new_char.replace('+', '') if '+' in new_char else '--' + new_char
                if new_char_pref in self.vocab:
                    self.vocab[new_char_pref] += pref_count
                else:
                    self.vocab[new_char_pref] = pref_count

            if suff_count > 0:
                self.vocab[new_char.replace('+', '') if '+' in new_char else new_char] = suff_count

            if top_pair in pair_count:
                del pair_count[top_pair]

            if len(self.vocab) >= self.vocab_size:
                break



        self.i2w = {}
        all_toks = self.special_toks + ['--' + x for x in punctuation + '।'+'॥']

        # not_pref = ['_aa', '_i', '_ee', '_u', '_oo', '_ri', '_ae', '_ai', '_o', '_aw', '_n', '_nn', '_aii', '_au']


        # pref_base_akshar = ['--'+ x for x in self.base_akshar if x not in not_pref]

        all_toks = all_toks + list(self.vocab)
        self.i2w = {i: tok for i, tok in enumerate(all_toks)}
        self.w2i = {tok: i for i, tok in self.i2w.items()}
        self.vocab_size = len(self.i2w)

        print('subword splitting...')

        self.subwords = {word:self._split_oov_2(word)
                         for word in tqdm(self.tok_freq)}
        self.subwords = {k:v for k,v in self.subwords.items() if k != v[2:]}


    def _split_oov(self, word):

        """
        function to split an OOV token

        :param word: python string
        :return: subword split
        """

        if word in self.special_toks:
            return word

        elif "--" + word in self.w2i:
            return "--" + word

        elif word in self.w2i:
            return word

        elif word in self.subwords:
            return self.subwords[word]

        elif word in string.punctuation:
            return word

        word = "--" + word

        subwords = {} # for storing all possible subwords

        # get all possible subword sequences



        for p in self.vocab:
            tmp = []
            if (p in word) and (p not in subwords):
                if word.index(p) == 0:
                    # print(p)
                    subwords[p] = []
                    sw = re.sub(re.escape(p), '', word, 1)
                    for s in self.vocab:
                        if (s in sw) and (s not in subwords[p]):
                            subwords[p].append(s)
                else:
                    pass

        subw = [] # storing different subword sequence

        # keep only required subwords

        # print(subwords)

        for p in subwords:

            rem = re.sub(re.escape(p), '', word, 1)
            suff = subwords[p]
            suff = sorted(suff, key=lambda x: len(x))[::-1]
            sorter = [(sw, rem.index(sw)) for sw in suff]
            suff = sorted(sorter, key=lambda x: x[1])

            suff_order = suff
            tmp = [p]
            rpl = ''

            prev = -1
            # print(suff)
            for x in suff_order:

                s, i = x

                if i > prev:
                    rem_word = rem.replace('+', '')

                    if rem == '':
                        break

                    elif s == rem_word[:len(s)]:
                        rpl += '+' * len(s)
                        rem = rpl + rem_word[len(s):]
                        tmp.append(s)

                    elif s == rem[i:i+len(s)]:
                        rem = rem[:i] + '+' * len(s) + rem[i+len(s):]
                        tmp.append(s)

                    prev += 1



                else:
                    pass

                # print(tmp)



            fin_rem = rem.replace('+', '')

            if fin_rem in rem and fin_rem in self.vocab:
                tmp.append(fin_rem)

            subw.append(tmp)
            # print(tmp)

        # print(subw)

        # sort the subwords as per the sequence of the word
        subw_dict = {x[0]: x[1:] for x in subw}

        for idx,(p,suff) in enumerate(subw_dict.items()):
            tmp = word
            tmp = tmp[len(p):]
            suff = list(set(suff))
            suff = sorted(suff, key=lambda x: len(x), reverse=True)
            suff_order = []

            if ''.join(suff) == tmp:
                continue

            else:
                for sw in suff:
                    while sw in tmp:
                        i = tmp.index(sw)
                        tmp = tmp[:i] + '+'*len(sw) + tmp[i+len(sw):]
                        suff_order.append((sw,i))

                if tmp != '+' * len(tmp):
                    subw[idx] = []
                    continue

                suff_order = sorted(suff_order,key=lambda x: x[1])
                suff = [x[0] for x in suff_order]
                subw[idx] = [p] + suff

        subw = [x for x in subw if len(x) != 0]


        if len(subw) == 0:
            self.subwords[word[2:]] = '<unk>'
            return '<unk>'


        min_split = min([len(x) for x in subw])
        subw = [x for x in subw if len(x) == min_split]


        if len(subw) == 1:
            self.subwords[word[2:]] = ' '.join(subw[0])
            return self.subwords[word[2:]]

        else:
            complete = [(x,self.score_split(x)) for x in subw]
            output = max(complete, key=lambda x: x[1])[0]
            self.subwords[word[2:]] = ' '.join(output)
            return self.subwords[word[2:]]

    def _split_oov_2(self, word):
        # 0. Safety check for empty strings
        if not word:
            return ""

        # 1. Base cases (Quick returns)
        if word in self.special_toks:
            return word

        if ("--" + word) in self.w2i:
            return "--" + word
        elif word in self.w2i:
            return word

        if word in self.subwords:
            return self.subwords[word]

        if word in string.punctuation:
            return word

        # 2. Break the word into unbreakable phonetic atomic units
        units = self.split_chars(word)
        if not units:
            return ""

        subwords = []
        i = 0

        # 3. Greedy Left-to-Right Longest Match
        while i < len(units):

            print

            match_found = False

            for j in range(len(units), i, -1):
                candidate = ''.join(units[i:j])
                clean_candidate = candidate.replace('+', '')

                vocab_check = ("--" + clean_candidate) if i == 0 else clean_candidate

                if vocab_check in self.w2i:
                    subwords.append(vocab_check)
                    i = j
                    match_found = True
                    break

            # 4. THE FIX: Output <unk> instead of the raw, unknown character
            if not match_found:
                # We don't know this character. Force it to be the <unk> token.
                subwords.append('<unk>')
                i += 1

        # 5. Cache and return the result
        final_output = ' '.join(subwords)
        self.subwords[word] = final_output

        return final_output

    def score_split(self,seq):
        seq = [self.vocab[x] for x in seq]
        score = np.prod(seq) / np.sum(seq)
        return score


    def tokenize(self, seq):
        """
        Function for tokenizing the sequences.
        The sequence is fisrt truncated till it maxlen and then
        "<cls>" and "<sep>" tokens are added at beginning and end

        example:

        input -> "i am playing"
        output before tokenization -> "<cls> ##i ##am ##play ing <sep>"
        tokenized output -> [1,5,6,7,8,2]

        :param seq: string sequences in a list
        :return: tokenized sequences in a list
        """
        seq = np.asarray(seq.split()[:self.maxlen-2])
        split_tok = np.vectorize(self._split_oov_2)
        seq = split_tok(seq)
        seq = seq[:self.maxlen - 2]
        seq = ['<cls>'] + ' '.join(seq).split() + ['<sep>']
        for i,s in enumerate(seq):
            if s in self.w2i:
                seq[i] = self.w2i[s]
            else:
                seq[i] = self.w2i['<unk>']
        return seq

    def add_padding(self,seq):
        """
        Function for adding padding to the sequence
        :param seq: python list as tokenized sequence
        :return: padded tokenized sequence as 1d array
        """

        if len(seq) > self.maxlen:
            seq = seq[:self.maxlen]
            seq[-1] = self.w2i['<sep>']
            return seq
        elif len(seq) < self.maxlen:
            seq = seq + [0 for _ in range(self.maxlen - len(seq))]
            return seq
        else:
            return seq

    def load_tokenizer(self,file_path):
        """
        load attributes from pkl file
        :param file_path: path to pkl file
        """
        with open(file_path,'rb') as f:
            data = pickle.load(f)
        self.__dict__.update(data)

    def save_tokenizer(self,file_path):
        """
        save attributes to pkl file
        :param file_path: path to pkl file
        """
        attributes = self.__dict__
        with open(file_path,'wb') as f:
            pickle.dump(attributes,f)

    def encode_hindi_text(self,input_text):

        text = input_text.copy()

        fixes = ['़', '्']

        def standardize(pair):

            if pair[1] == fixes[0]:
                return 'N_' + pair[0]

            elif pair[1] == fixes[1]:
                return 'HALF_' + pair[0]

            else:
                print('Error in sequence')

        def encode_word(word,lookUp_table):

            chars = list(word)
            char_std = []

            try:

                for c in chars:
                    if c not in fixes:
                        char_std.append(c)
                    else:
                        pair = (char_std[-1], c)
                        pair_std = standardize(pair)

                        if pair_std is not None:
                            char_std[-1] = pair_std

                return ''.join([lookUp_table[x] for x in char_std])


            except Exception as e:

                return '<unk>'


        with open('char_table.json','r') as f:
            lookUp_table = json.load(f)


        for i,sent in enumerate(text):
            words = sent.split()
            for j,w in enumerate(words):
                if self._is_hindi(w):
                    w = encode_word(w,lookUp_table)
                    words[j] = w
                else:
                    pass

            text[i] = ' '.join(words)

        return text

    def _is_hindi(self,x):
        pattern = re.compile(r'[\u0900-\u097F]')
        if pattern.search(x):
            return True

        return False














# text = ["""
# walker walk playwalk player walking player walking playing""",
# """walked walkplayed learned earned learning earn""",
# """wished watching watched watcher watcher learns earns earning""",
# """watches cares scares caring watched cared share !"""]
#
# # text = [
# #     "खेलनेवाला खेल खिलाडी खेलता चलता चलते चलता खिलाडी खेलते",
# #     "चला पढा पढाई पढाता पढते पढता सीखा सीखता सीखते सीख",
# #     "देखनेवाला देखता देखा देखा देखते कमाता कमाते कमाया कमाना",
# #     "देखता डरता डराता डरा डराते देखा बाँटता बाँटा बाँटते"
# # ]
#
#
# #
# #
# tok = BytePairEncodingTokenizer()
# tok.load_tokenizer("tokenizer.pkl")

# tok(text)

# word = "capitalisation"
#
# try:
#     del tok.subwords[word]
#
# except:
#     pass
#
# # #
# print(tok._split_oov(word))







