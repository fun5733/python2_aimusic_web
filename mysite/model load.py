
# 0. 사용할 패키지 불러오기

from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.utils import np_utils
from keras.models import load_model
from keras.models import model_from_json
import random

# 랜덤시드 고정시키기
np.random.seed(5)

# 손실 이력 클래스 정의
class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.epoch = 0
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        print("epoch: {0} - loss: {1:8.6f}".format(self.epoch, logs.get('loss')))
        self.epoch += 1

# 데이터셋 생성 함수
def seq2dataset(seq, window_size):
    dataset = []
    for i in range(len(seq) - window_size):
        subset = seq[i:(i + window_size + 1)]
        dataset.append([code2idx[item] for item in subset])
    return np.array(dataset)

def open_seq(code_list):
    data = []
    learn_data = []
    data_ls = []
    data_num = []
    for i in range(len(code_list)):
        for j in range(len(code_list[i])):
            data.append(code_list[i][j])
    for i in data:
        if not i in data_ls:
            data_ls.append(i)
    data_ls.append(':|')
    data_ls.append('|:')
    for j in range(0, len(data_ls)):
        data_num.append(j)
    data2num = dict(zip(data_ls, data_num))
    num2data = dict(zip(data_num, data_ls))
    for i in range(len(data)):
        tmp = ''
        if data[i] is '|' and data[i-1] is ':':
            tmp = ':|'
            learn_data.pop()
        elif data[i] is ':' and data[i-1] is '|':
            tmp = '|:'
            learn_data.pop()
        else:
            tmp = data[i]
        learn_data.append(tmp)
    return learn_data, data2num, num2data

def open_file(filename):
    f = open(filename, 'r', encoding='utf-16')
    M = []
    L = []
    K = []
    Q = []
    X = []
    tmp = ''
    count = 0
    while True:
        line = f.readline()
        if not line: break
        if line[0] is 'X':
            count = 0
            X.append(tmp)
            tmp = ''
        if line[0] is 'M':
            M.append(line[2:])
        if line[0] is 'L':
            L.append(line[2:])
        if line[0] is 'K':
            K.append(line[2:])
            count = count + 1
            continue
        if line[0] is 'Q':
            Q.append(line[2:])
        if len(Q) < len(K):
            Q.append('no')
        if len(L) < len(K):
            L.append('no')
        if len(M) < len(K):
            M.append('no')
        if count is 1:
            tmp = tmp + line
    f.close()
    X.append(tmp)
    del X[0]
    return M, L, K, Q, X


# 2. 데이터셋 생성하기
n_steps = 4  # step
n_inputs = 1  # 특성수

n = int(input("0: 밝음 1: 잔잔 2: 긴박 ="))
if n is 0:
    rhythm, code_len, chords, quick, X_code = open_file("happy.txt")
elif n is 1:
    rhythm, code_len, chords, quick, X_code = open_file("calm.txt")
else:
    rhythm, code_len, chords, quick, X_code = open_file("thrill.txt")

seq, code2idx, idx2code = open_seq(X_code)
dataset = seq2dataset(seq, window_size=4)

print(dataset.shape)

# 입력(X)과 출력(Y) 변수로 분리하기
x_train = dataset[:, 0:n_steps]
y_train = dataset[:, n_steps]

max_idx_value = len(code2idx) - 1
# 입력값 정규화 시키기
x_train = x_train / float(max_idx_value)

# 입력을 (샘플 수, 타입스텝, 특성 수)로 형태 변환
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], n_inputs))

# 라벨값에 대한 one-hot 인코딩 수행
y_train = np_utils.to_categorical(y_train)

one_hot_vec_size = y_train.shape[1]

print("one hot encoding vector size is ", one_hot_vec_size)


# 8. 모델 사용하기
'''
json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("model_w.h5")
print("loaded model from disk")
'''
model = keras.models.load_model('model.h5')
#model = load_model('model.h5', custom_objects= {'lstm':UnifiedLSTM})
pred_count = 50  # 최대 예측 개수 정의


# 곡 전체 예측

seq_in = ['G', 'F', 'E', 'D']
seq_out = seq_in
seq_in = [code2idx[it] / float(max_idx_value) for it in seq_in]  # 코드를 인덱스값으로 변환

for i in range(pred_count):
    sample_in = np.array(seq_in)
    sample_in = np.reshape(sample_in, (1, n_steps, n_inputs))  # 샘플 수, 타입스텝 수, 속성 수
    pred_out = model.predict(sample_in)
    idx = np.argmax(pred_out)
    seq_out.append(idx2code[idx])
    seq_in.append(idx / float(max_idx_value))
    seq_in.pop(0)

model.reset_states()

m_result = ''.join(random.sample(rhythm, 1))
l_result = ''.join(random.sample(code_len, 1))
k_result = ''.join(random.sample(chords, 1))


print("full song prediction : ")
print("X: 1\nT: sample\nM: %s\nL: %s\nK: %s" %(m_result, l_result, k_result))
print(''.join(seq_out))

'''
import music21

note_seq = ""
for note in seq_out:
    note_seq += note + " "

conv_midi = music21.converter.subConverters.ConverterMidi()

m = music21.converter.parse("2/4 " + note_seq, format='tinyNotation')

m.show("midi")
'''