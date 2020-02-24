import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random

puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》']
def preprocess_file(Config):
    # 语料文本内容
    files_content = ''
    with open(Config.file_name, 'r', encoding='utf-8') as f:
        for line in f:
            # 每行的末尾加上"]"符号代表一首诗结束
            for char in puncs:
                line = line.replace(char, "")
            files_content += line.strip() + "]"

    words = sorted(list(files_content))
    words.remove(']')
    counted_words = {}
    for word in words:
        if word in counted_words:
            counted_words[word] += 1
        else:
            counted_words[word] = 1

    # 去掉低频的字
    erase = []
    for key in counted_words:
        if counted_words[key] <= 2:
            erase.append(key)
    for key in erase:
        del counted_words[key]
    del counted_words[']']
    wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])

    words, _ = zip(*wordPairs)
    # word到id的映射
    word2num = dict((c, i + 1) for i, c in enumerate(words))
    num2word = dict((i, c) for i, c in enumerate(words))
    word2numF = lambda x: word2num.get(x, 0)
    return word2numF, num2word, words, files_content

class HotelModel(object):
    def __init__(self, config):
        self.model = None
        self.do_train = True
        self.loaded_model = False
        self.config = config

        # 文件预处理
        self.word2numF, self.num2word, self.words, self.files_content = preprocess_file(self.config)
        if os.path.exists(self.config.weight_file):
            self.model = build_model()
            self.model.load_weights(tf.train.latest_checkpoint(self.config.weight_file))
            # load_model(self.config.weight_file)
            self.model.summary()
        else:
            self.train()
        self.do_train = False
        self.loaded_model = True
        
    def preprocess_file(self, Config):
        puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》']
        # 语料文本内容
        files_content = ''
        with open(Config.poetry_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 每行的末尾加上"]"符号代表一首诗结束
                for char in puncs:
                    line = line.replace(char, "")
                files_content += line.strip() + "]"

        words = sorted(list(files_content))
        words.remove(']')
        counted_words = {}
        for word in words:
            if word in counted_words:
                counted_words[word] += 1
            else:
                counted_words[word] = 1

        # 去掉低频的字
        erase = []
        for key in counted_words:
            if counted_words[key] <= 2:
                erase.append(key)
        for key in erase:
            del counted_words[key]
        del counted_words[']']
        wordPairs = sorted(counted_words.items(), key=lambda x: -x[1])

        words, _ = zip(*wordPairs)
        # word到id的映射
        word2num = dict((c, i + 1) for i, c in enumerate(words))
        num2word = dict((i, c) for i, c in enumerate(words))
        word2numF = lambda x: word2num.get(x, 0)
        return word2numF, num2word, words, files_content

    def build_model(self):
        '''建立模型'''
        input_tensor = tf.keras.layers.Input(shape=(self.config.max_len,))
        embedd = tf.keras.layers.Embedding(len(self.num2word)+1, 300, input_length=self.config.max_len)(input_tensor)
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(embedd)
        dropout = tf.keras.layers.Dropout(0.6)(lstm)
        lstm = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True))(embedd)
        dropout = tf.keras.layers.Dropout(0.6)(lstm)
        flatten = tf.keras.layers.Flatten()(lstm)
        dense = tf.keras.layers.Dense(len(self.words), activation='softmax')(flatten)
        self.model = tf.keras.models.Model(inputs=input_tensor, outputs=dense)
        optimizer = tf.keras.optimizers.Adam(lr=self.config.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    def sample(self, preds, temperature=1.0):
        '''
        当temperature=1.0时，模型输出正常
        当temperature=0.5时，模型输出比较open
        当temperature=1.5时，模型输出比较保守
        在训练的过程中可以看到temperature不同，结果也不同
        '''
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def generate_sample_result(self, epoch, logs):
        print("\n==================Epoch {}=====================".format(epoch))
        for diversity in [0.5, 1.0, 1.5]:
            print("------------Diversity {}--------------".format(diversity))
            start_index = random.randint(0, len(self.files_content) - self.config.max_len - 1)
            generated = ''
            sentence = self.files_content[start_index: start_index + self.config.max_len]
            generated += sentence
            for i in range(20):
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(sentence[-6:]):
                    x_pred[0, t] = self.word2numF(char)

                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, diversity)
                next_char = self.num2word[next_index]
                generated += next_char
                sentence = sentence + next_char
            print(sentence)

    def predict(self, text):
        if not self.loaded_model:
                return
        with open(self.config.file_name, 'r', encoding='utf-8') as f:
            file_list = f.readlines()
        random_line = random.choice(file_list)
        # 如果给的text不到四个字，则随机补全
        if not text or len(text) != 4:
            for _ in range(4 - len(text)):
                random_str_index = random.randrange(0, len(self.words))
                text += self.num2word.get(random_str_index) if self.num2word.get(random_str_index) not in [',', '。','，'] else self.num2word.get(random_str_index + 1)
        seed = random_line[-(self.config.max_len):-1]
        res = ''
        seed = 'c' + seed
        for c in text:
            seed = seed[1:] + c
            for j in range(5):
                x_pred = np.zeros((1, self.config.max_len))
                for t, char in enumerate(seed):
                    x_pred[0, t] = self.word2numF(char)
                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = self.sample(preds, 1.0)
                next_char = self.num2word[next_index]
                seed = seed[1:] + next_char
            res += seed
        return res

    def data_generator(self):
        i = 0
        while 1:
            x = self.files_content[i: i + self.config.max_len]
            y = self.files_content[i + self.config.max_len]
            puncs = [']', '[', '（', '）', '{', '}', '：', '《', '》', ':']
            if len([i for i in puncs if i in x]) != 0:
                i += 1
                continue
            if len([i for i in puncs if i in y]) != 0:
                i += 1
                continue
            y_vec = np.zeros(
                shape=(1, len(self.words)),
                dtype=np.bool
            )
            y_vec[0, self.word2numF(y)] = 1.0
            x_vec = np.zeros(
                shape=(1, self.config.max_len),
                dtype=np.int32
            )
            for t, char in enumerate(x):
                x_vec[0, t] = self.word2numF(char)
            yield x_vec, y_vec
            i += 1
    def train(self):
        number_of_epoch = 10
        if not self.model:
            self.build_model()
        self.model.summary()
        self.model.fit_generator(
            generator=self.data_generator(),
            verbose=True,
            steps_per_epoch=self.config.batch_size,
            epochs=number_of_epoch,
            callbacks=[
                keras.callbacks.ModelCheckpoint(self.config.weight_file, save_weights_only=False),
                keras.callbacks.LambdaCallback(on_epoch_end=self.generate_sample_result)
            ])