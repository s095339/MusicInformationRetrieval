import librosa
import librosa.display as display
import numpy as np
import matplotlib.pyplot as plt
import soundfile

def Spectrogram_1_1(sound_flie_name):
    #load audio file-------------------------------------
    audio, sr = librosa.load(sound_flie_name)
    duration = librosa.get_duration(y=audio,sr=sr)
    print(f"load audiofile {sound_flie_name}")
    print(f"sampling rate = {sr}, duration = {duration}")
    #----------------------------------------------------

    #show plt----------------------------------------------------
    plt.figure()
    hop_len = 512
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=hop_len)),
                            ref=np.max)
    display.specshow(D, y_axis='log', sr=sr, hop_length=hop_len,
                         x_axis='time')
    plt.show()
    #----------------------------------------------------
    return
def main():
    #HW1_1=============================================#
    #(a)
    sound_flie_name = "mixkit-losing-piano-2024.wav"
    Spectrogram_1_1(sound_flie_name)
    #(b)
    #(c)
    #(d)

    #================================================#




    return

if __name__== "__main__":
    main()
