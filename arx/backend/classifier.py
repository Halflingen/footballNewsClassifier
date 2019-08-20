from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

def predict_paragraphs(paragraphs, model_type):
    model = 0
    if model_tyope == 'cnn':
        tokenizer = 0
        with open('tokenizer_cnn.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        encoded_data = self.tokenizer.texts_to_sequences(paragraphs)
        padded_data = pad_sequences(encoded_data, maxlen=60, padding='post')
        model = load_model('./model_test.h5')
        score = model.predict(padded_data, batch_size=32)
        print(paragraphs)
        print(score)
        return score

