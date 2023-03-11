import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD # алгоритм будет проецировать данные в 300-мерное пространство
import numpy as np
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
import telebot
import os
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS
AudioSegment.converter = "ffmpeg-2023-03-05-git-912ac82a3c-full_build/bin/ffmpeg.exe"
AudioSegment.ffmpeg = "ffmpeg-2023-03-05-git-912ac82a3c-full_build/bin/ffmpeg.exe"
AudioSegment.ffprobe ="ffmpeg-2023-03-05-git-912ac82a3c-full_build/bin/ffprobe.exe"

good = pd.read_csv('good.tsv', sep='\t')
vectorizer = TfidfVectorizer()
vectorizer.fit(good.context_0)
matrix_big = vectorizer.transform(good.context_0)
print(matrix_big.shape)
svd = TruncatedSVD(n_components=300)
# коэффициенты этого преобразования выучиваются так, # чтобы сохранить максимум информации об исходной матрице
svd.fit(matrix_big)
matrix_small = svd.transform(matrix_big)
# в результате строк (наблюдений) столько же, столбцов меньше
print(matrix_small.shape)
# при этом сохранилось больше 40% исходной информации
print(svd.explained_variance_ratio_.sum())

def softmax(x):
    proba = np.exp(-x)
    return proba / sum(proba)
class NeighborSampler(BaseEstimator):
 def __init__ (self, k=5, temperature=1.0):
    self.k = k
    self.temperature = temperature
 def fit(self, x, y):
    self.tree_ = BallTree(x)
    self.y_ = np.array(y)
 def predict(self, X, random_state=None):
    distances, indices = self.tree_.query(X, return_distance=True, k=self.k)
    result = []
    for distance, index in zip(distances, indices):
      result.append(np.random.choice(index, p=softmax(distance * self.temperature)))
    return self.y_[result]


ns = NeighborSampler()
ns.fit(matrix_small, good.reply)
pipe = make_pipeline(vectorizer, svd, ns)
API_TOKEN = '5790101995:AAFO1_pIkgN2rB8pQ_GBWKptpd0_AjG9Fjc'
bot = telebot.TeleBot(API_TOKEN)

@bot.message_handler(content_types=['voice'])
def handle_voice_message(message):
    try:
        # Download the voice message file
        file_info = bot.get_file(message.voice.file_id)
        file = bot.download_file(file_info.file_path)

        # Save the voice message file locally as a .oga file
        file_path = 'voice_message.oga'
        with open(file_path, 'wb') as f:
            f.write(file)

        # Convert the .oga file to .wav format for speech recognition
        sound = AudioSegment.from_ogg(file_path)
        sound.export('voice_message.wav', format='wav')

        # Use speech recognition to convert the voice message to text
        r = sr.Recognizer()
        with sr.AudioFile('voice_message.wav') as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language='ru-RU')

        # Use the text to generate a response using the pipe object
        response_text = pipe.predict([text])[0]

        # Convert the response text to an audio file using text-to-speech synthesis
        tts = gTTS(text=response_text, lang='ru')
        tts.save('response_voice_message.mp3')

        # Send the response back to the user as a voice message
        with open('response_voice_message.mp3', 'rb') as f:
            bot.send_voice(message.chat.id, f)

        # Delete the temporary voice message and audio files
        os.remove(file_path)
        os.remove('voice_message.wav')
        os.remove('response_voice_message.mp3')

    except Exception as e:
        # If an exception occurs, send a voice message saying "Failed to recognize"
        tts = gTTS(text="Не удалось распознать", lang='ru')
        tts.save('failed_to_recognize.mp3')
        with open('failed_to_recognize.mp3', 'rb') as f:
            bot.send_voice(message.chat.id, f)
        os.remove('failed_to_recognize.mp3')

@bot.message_handler(func=lambda message: True)
def echo_message(message):
    bot.send_message(message.chat.id, pipe.predict([message.text]))
bot.polling()