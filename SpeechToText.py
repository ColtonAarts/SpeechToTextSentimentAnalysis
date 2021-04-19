# import speech_recognition as sr
# from pysbd.utils import PySBDFactory
# import spacy
# import tkinter as tk
#
# # Initialize recognizer class (for recognizing the speech)
#
# r = sr.Recognizer()
#
# nlp = spacy.blank('en')
# nlp.add_pipe(PySBDFactory(nlp))
#
# doc = nlp("This is a sentence without punctuation The cat ran away")
# print(list(doc.sents))
#
# # Reading Microphone as source
# # listening the speech and store in audio_text variable
#
# with sr.Microphone() as source:
#     print("Talk")
#     audio_text = r.listen(source)
#     print("Time over, thanks")
#     # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
#
#     try:
#         # using google speech recognition
#         print("Text: " + r.recognize_google(audio_text))
#     except:
#         print("Sorry, I did not get that")