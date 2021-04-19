from keras_preprocessing.text import Tokenizer
from tensorflow import keras
from tensorflow.keras import layers

from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.callbacks import EarlyStopping

STOPWORDS = set(stopwords.words('english'))
SYMBOLS = re.compile('[^0-9a-z #+_]')
stemmer = WordNetLemmatizer()


class NeuralNetwork:
    # shpe should be X.shape[1]
    def __init__(self, shape, num_lables, max_nb_words=50000, max_sequence_length=250, embedding_dim=100):
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words=max_nb_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        self.model = keras.Sequential()

        # Turns positive integers (indexes) into dense vectors of fixed size
        # NOTE: can only be used as the first layer in a model.
        self.model.add(layers.Embedding(max_nb_words, embedding_dim, input_length=shape))

        # convolutional layer: applies a filter in order to create a feature map that
        # summarizes the presence of detected features in the input.
        # we use relu because it was suggested
        # kernel_size needs to be odd -> (1,3,5,7 are common)
        self.model.add(layers.Conv1D(64, kernel_size=3, activation="relu"))

        # Downsamples the input representation by taking the maximum value over the window defined by pool_size.
        self.model.add(layers.MaxPooling1D())

        # The Dropout layer randomly sets input units to 0
        # with a frequency rate at each step during training time, which helps prevent overfitting.
        self.model.add(layers.SpatialDropout1D(0.2))

        # Adds a bidirectional LSTM layer. Allows us to read input twice (forwards and backwards)
        # We use concatenation as the merge mode because it was used in a paper
        self.model.add(layers.Bidirectional(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2), merge_mode='concat'))

        # performs activation calculation
        # model.add(layers.Dense(4, activation='softmax'))  # For detecting all emotions
        self.model.add(layers.Dense(num_lables, activation='softmax'))  # For detecting single emotions
        self.model.summary()

        # Configures the model for training.
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, x_train, y_train, epochs=5, batch_size=64):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                       callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    def predict(self, x_test):
        results = self.model.predict(x_test)
        return results
