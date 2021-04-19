class LexicalAnalysis:
    def __init__(self):
        self.words = self.create_words()

    def create_words(self):
        lexical_file = open("D:\\PycharmProjects\\SpeechToTextSentiemtnAnalysis\\EmotionDetection\\Lexical\\lexiconFile", "r").read().split("\n")
        my_dict = dict()
        for line in lexical_file:
            line = line.split(",")
            if line[0] not in my_dict:
                my_dict[line[0]] = dict()
                my_dict[line[0]]["anger"] = "0"
                my_dict[line[0]]["fear"] = "0"
                my_dict[line[0]]["joy"] = "0"
                my_dict[line[0]]["sadness"] = "0"
                my_dict[line[0]][line[-1]] = line[1]
            else:
                my_dict[line[0]][line[-1]] = (line[1])
        return my_dict

    def find_sentiment(self, word):
        if word in self.words:
            return self.words[word]
        else:
            my_dict = dict()
            my_dict["anger"] = 0
            my_dict["fear"] = 0
            my_dict["joy"] = 0
            my_dict["sadness"] = 0
            return my_dict
