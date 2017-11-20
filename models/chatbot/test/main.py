import speech_recognition as sr
import os
import subprocess

from gtts import gTTS
from chatterbot import ChatBot

dataset_path = "movie_lines.txt"

chatbot = ChatBot(
    'BotterChatter',
    trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
)
# trainer='chatterbot.trainers.ListTrainer'


def dataset_to_list(dataset_path):
    dialogue = []
    with open(dataset_path, "rb") as f:
        lines = f.readlines()

    for line in lines:
        line = str(line, "latin-1").strip()
        dialogue.append(line.split("+++$+++")[-1])

    return dialogue


if __name__ == "__main__":
    r = sr.Recognizer()
    dialogue = dataset_to_list(dataset_path)
    dialogue = dialogue[:(len(dialogue) // 100)]
    #chatbot.train(dialogue)
    chatbot.train("chatterbot.corpus.english")


    while 1:
        print("\nSay something!")
        with sr.Microphone() as source:
            audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You: {}".format(text))

            resp = str(chatbot.get_response(text))
            print("{}: {}".format(chatbot.name, resp))
        except:
            resp = "Say something that is not gibberish"

        tts = gTTS(text=resp, lang='en')
        tts.save("resp.mp3")
        subprocess.check_call(['mpg321', 'resp.mp3'], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
