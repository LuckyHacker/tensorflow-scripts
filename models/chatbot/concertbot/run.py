
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.interpreter import RegexInterpreter
from gtts import gTTS

def play_response(response):
    tts = gTTS(text=response, lang="en")
    tts.save("response.mp3")
    os.system("mpg321 response.mp3")

def run_concerts(serve_forever=True):
    agent = Agent.load("models/dialogue",
                       interpreter=RegexInterpreter())

    print("Say something!")
    while serve_forever:
        #agent.handle_channel(ConsoleInputChannel())
        text = input().strip()
        response = agent.handle_message(text)[0]
        print(response)
        play_response(response)
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    run_concerts()
