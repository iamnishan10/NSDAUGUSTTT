import librosa
import tensorflow as tf
import numpy as np
from keras.models import load_model


keras_model_path = './models'
SAVED_MODEL_PATH = keras.models.load_model(keras_model_path)

class _speaker_recog:
    :param model: Trained model
    

    model = None
    _mapping = [
        "59",
        "53",
        "54",
        "57",
        "60",
        "55",
        "58",
        "51",
        "52",
        "56",
        "49",
        "42",
        "46",
        "44",
        "50",
        "48",
        "47",
        "45",
        "43",
        "41",
        "35",
         "37",
        "31",
        "34",
        "38",
        "40",
        "32",
        "39",
        "33",
        "36",
        "30",
        "22",
        "25",
        "28",
        "23",
        "29",
        "24",
        "27",
        "21",
        "26",
         "19",
        "12",
        "16",
        "13",
        "14",
        "18",
        "11",
        "17",
        "20",
        "15",
        "05",
        "09",
        "07",
        "01",
        "08",
        "06",
        "04",
        "02",
        "03",
        "10"
    ]
    _instance = None


def predict(self, file_path):
        
        :param file_path (str): Path to audio file to predict
        :return predicted_keyword (str): Keyword predicted by the model
        

        # extract MFCC
        MFCCs = self.preprocess(file_path)

        # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        # get the predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return predicted_keyword


def preprocess(self, file_path, num_mfcc=13, n_fft=2048, hop_length=512):
            Extract MFCCs from audio file.
        :param file_path (str): Path of audio file
        :param num_mfcc (int): # of coefficients to extract
        :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
        :param hop_length (int): Sliding window for STFT. Measured in # of samples
        :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
        

        # load audio file
        signal, sample_rate = librosa.load(file_path)

        if len(signal) >= SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = signal[:SAMPLES_TO_CONSIDER]
             # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
            MFCCsDelta = librosa.feature.delta(MFCCs)
             MFCCsDoubleDelta = librosa.feature.delta(MFCCs , order =2)

            MFCC_CONC =numpy.concatenate((MFCCs, MFCCsDelta, MFCCsDoubleDelta ))
        return MFCC_CONC.T


def speaker_recog():
        Factory function for speaker_recog class.
    :return _speaker_recog._instance (_speaker_recog):


    # ensure an instance is created only the first time the factory function is called
    if _speaker_recog._instance is None:
        _speaker_recog._instance = _speaker_recog()
        _speaker_recog.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
    return _speaker_recog._instance




if __name__ == "__main__":

    # create instances of the speaker recognition system 
    srs = speaker_recog()
    srs1 = speaker_recog()
     # check that different instances point back to the same object (singleton)
    assert srs is srs1

    # make a prediction
    speaker = srs.predict("./DjangoRestApi/speaker_audio")
    speaker1 = srs.predict("./DjangoRestApi/speaker_audio")
    print(speaker)
    print(speaker1)
