import tkinter as tk
import speech_recognition as sr
from EmotionDetection.Classification import NeuralNetwork
from EmotionDetection.Lexical import LexicalAnalysis
from nltk.stem import WordNetLemmatizer
import re
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pandas
import operator
import numpy as np


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.r = sr.Recognizer()
        self.lexical = LexicalAnalysis.LexicalAnalysis()
        self.master = master
        self.pack()
        self.create_widgets()
        self.running = False
        self.text = ""
        self.text_sequence = None

        self.stemmer = WordNetLemmatizer()

        df = pandas.read_csv("D:\\PycharmProjects\\ThesisWork\\Data\\EmotionDetection\\%_by_Emo_Full_Data_data (1).csv")

        df['Tweet'] = df['Tweet'].apply(self.clean)

        MAX_NB_WORDS = 50000
        # Max number of words in each tweet.
        self.MAX_SEQUENCE_LENGTH = 250
        self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        self.tokenizer.fit_on_texts(df['Tweet'].values)
        # Integer replacement
        X = self.tokenizer.texts_to_sequences(df['Tweet'].values)

        X = pad_sequences(X, maxlen=self.MAX_SEQUENCE_LENGTH)
        # Gets categorical values for the labels
        Y = pandas.get_dummies(df['Emotion']).values

        self.neuralNetwork = NeuralNetwork.NeuralNetwork(X.shape[1], 4)
        self.neuralNetwork.fit(X, Y)

    def clean(self, tweet):

        # Use this to remove hashtags since they can become nonsense words
        # trimmed_tweet = re.sub(r'(\s)#\w+', r'\1', tweet)

        # Remove all the special characters
        trimmed_tweet = re.sub(r'\W', ' ', tweet)

        # remove all single characters
        trimmed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', trimmed_tweet)

        # Remove single characters from the start
        trimmed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', trimmed_tweet)

        # Substituting multiple spaces with single space
        trimmed_tweet = re.sub(r'\s+', ' ', trimmed_tweet, flags=re.I)

        # Removes numbers
        trimmed_tweet = ''.join([i for i in trimmed_tweet if not i.isdigit()])

        # # Removing prefixed 'b'
        # trimmed_tweet = re.sub(r'^b\s+', '', trimmed_tweet)

        # Converting to Lowercase
        trimmed_tweet = trimmed_tweet.lower()

        # Lemmatization
        trimmed_tweet = trimmed_tweet.split()
        trimmed_tweet = [self.stemmer.lemmatize(word) for word in trimmed_tweet]
        trimmed_tweet = ' '.join(trimmed_tweet)
        return trimmed_tweet

    def create_widgets(self):
        self.text_field = tk.Text()
        self.text_field.tag_configure("red_tag", foreground="red")
        self.text_field.tag_configure("yellow_tag", foreground="yellow")
        self.text_field.tag_configure("black_tag", foreground="black")
        self.text_field.tag_configure("green_tag", foreground="green")
        self.text_field.tag_configure("blue_tag", foreground="blue")
        self.label = tk.Label()
        self.label.pack()
        self.text_field.pack()

        self.record = tk.Button(self, text="Push to Record")
        self.record.pack(side="left")
        self.record["command"] = self.start_capture

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")


    def start_capture(self):
        self.text_field.insert(tk.END, "Talk")
        print("Talk")
        with sr.Microphone() as source:


            audio_text = self.r.listen(source)
            self.text_field.insert(tk.END, "Time over, Thanks")
            # print("Time over, thanks")
            # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling

            try:
                # using google speech recognition
                self.text = self.r.recognize_google(audio_text)
                lst = list()
                lst.append(self.text)
                self.text_sequence = self.tokenizer.texts_to_sequences(lst)
                self.text_sequence = pad_sequences(self.text_sequence, self.MAX_SEQUENCE_LENGTH)
                results = self.neuralNetwork.predict(self.text_sequence)
                indexes = ""

                # results = model.predict(X_test)
                for prediction in results:
                    max_percent = max(prediction)
                    indexes = str(prediction.tolist().index(max_percent))
                if indexes == '0':
                    print("anger")
                    indexes = "anger"
                elif indexes == "1":
                    print("fear")
                    indexes = "fear"
                elif indexes == "2":
                    print("joy")
                    indexes = "joy"
                else:
                    print("sadness")
                    indexes = "sadness"
                print("Text: " + self.text)
                print(indexes)
                colours = self.lexical_analysis(self.text)
                words = self.text.split(" ")

                self.text_field.delete('1.0', tk.END)
                self.text_field.insert(tk.END, self.text)

                for num in range(len(words)):
                    word = words[num]
                    offset = "+%dc" % len(word)
                    pos_start = self.text_field.search(word, '1.0', tk.END)

                    while pos_start:
                        pos_end = pos_start + offset
                        self.text_field.tag_add(colours[num]+"_tag", pos_start, pos_end)
                        pos_start = self.text_field.search(word, pos_end, tk.END)

                self.text_field.insert(tk.END, "\n" + indexes)
            except:
                print("Sorry, I did not get that")


    def lexical_analysis(self, sentence):
        sentence = sentence.split(" ")
        lst = list()
        for word in sentence:
            values = self.lexical.find_sentiment(word)
            max_value = max(values.items(), key=operator.itemgetter(1))[0]
            print(values)
            print(max_value)
            if values[max_value] != 0:
                if max_value == "fear":
                    lst.append("blue")
                elif max_value == "anger":
                    lst.append("red")
                elif max_value == "sadness":
                    lst.append("yellow")
                elif max_value == "joy":
                    lst.append("green")
            else:
                lst.append("black")
            print(lst)
        return lst


    def end_capture(self):
        print(self.text)

root = tk.Tk()
app = Application(master=root)
app.mainloop()