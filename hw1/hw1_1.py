import librosa
import librosa.display as display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

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
def WriteChirpToFile_1_c():
    sr = 8000
    t = np.arange(0,10,1/sr)
    x1 = np.sin(2000*t*t)
    x2 = np.sin(2000*t+10*np.sin(2.5*t*t))
    sf.write("x1.wav",x1,sr)
    sf.write("x2.wav",x2,sr)
    return
def main():
    #HW1_1=============================================#
    #(a)---------------------------------------------
    sound_flie_name = "mixkit-losing-piano-2024.wav"
    Spectrogram_1_1(sound_flie_name)
    #------------------------------------------------
    #(b)

    #(c)---------------------------------------------
    WriteChirpToFile_1_c()
    #------------------------------------------------
    
    #(d)---------------------------------------------
    #------------------------------------------------
    #================================================#




    return

if __name__== "__main__":
    main()
