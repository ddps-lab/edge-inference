import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('./model/rnn_imdb_model.h5')

negative_input = "This movie was just way too overrated. The fighting was not professional and in slow motion. I was expecting more from a 200 million budget movie. The little sister of T.Challa was just trying too hard to be funny. The story was really dumb as well. Don't watch this movie if you are going because others say its great unless you are a Black Panther fan or Marvels fan."

positive_input = "I was lucky enough to be included in the group to see the advanced screening in Melbourne on the 15th of April, 2012. And, firstly, I need to say a big thank-you to Disney and Marvel Studios. \
Now, the film... how can I even begin to explain how I feel about this film? It is, as the title of this review says a 'comic book triumph'. I went into the film with very, very high expectations and I was not disappointed. \
Seeing Joss Whedon's direction and envisioning of the film come to life on the big screen is perfect. The script is amazingly detailed and laced with sharp wit a humor. The special effects are literally mind-blowing and the action scenes are both hard-hitting and beautifully choreographed."

word_to_index = imdb.get_word_index()

# 알파벳과 숫자를 제외하고 모두 제거 및 알파벳 소문자화
new_sentence = re.sub('[^0-9a-zA-Z ]', '', positive_input).lower()
encoded = []

# 띄어쓰기 단위 토큰화 후 정수 인코딩
for word in new_sentence.split():
    try :
    # 단어 집합의 크기를 10,000으로 제한.
     if word_to_index[word] <= 10000:
        encoded.append(word_to_index[word]+3)
     else:
    # 10,000 이상의 숫자는 <unk> 토큰으로 변환.
        encoded.append(2)
    # 단어 집합에 없는 단어는 <unk> 토큰으로 변환.
    except KeyError:
      encoded.append(2)

pad_sequence = pad_sequences([encoded], maxlen=500)
score = float(model.predict(pad_sequence)) # 예측

if(score > 0.5):
    print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
else:
    print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))
