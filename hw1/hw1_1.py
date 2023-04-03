import librosa
import librosa.display as display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

def main():
    
    #(a)---------------------------------------------
    sound_flie_name = "mixkit-losing-piano-2024.wav"
    print(f"load audiofile {sound_flie_name}...")
    audio, sr = librosa.load(sound_flie_name)
     #load audio file-------------------------------------
    duration = librosa.get_duration(y=audio,sr=sr)
    print(f"sampling rate = {sr}, duration = {duration}")
    #----------------------------------------------------

    #show plt----------------------------------------------------
    plt.figure()
    hop_len = 512
    window_len = 2048
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio, hop_length=hop_len,win_length=window_len)),
                            ref=np.max)
    display.specshow(D, y_axis='log', sr=sr, hop_length=hop_len,
                         x_axis='time')
    plt.show()
    #----------------------------------------------------
    
    #(b)calculate deriaive of x1 x2 for (D)------------
    t_x1 = np.arange(0,10,hop_len*(1/SR))
    IF_x1 = 2000*t_x1/np.pi

    t_x2 = np.arange(0,10,hop_len*(1/SR))
    IF_x2 = 2000+50*t_x2*np.cos(2.5*t*t)
    IF_x2 /= (2*np.pi)
    #------------------------------------------------

    #(c)---------------------------------------------
    SR = 8000
    t = np.arange(0,10,1/SR)
    x1 = np.sin(2000*t*t)
    x2 = np.sin(2000*t+10*np.sin(2.5*t*t))
    import os
    if not os.path.exists("x1.wav"):
        sf.write("x1.wav",x1,sr)
    if not os.path.exists("x2.wav"):
        sf.write("x2.wav",x2,sr)
    #------------------------------------------------
    
    #(d)---------------------------------------------
    plt.subplot(2,1,1)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(x1, hop_length=hop_len,win_length=window_len)),
                            ref=np.max)
    display.specshow(D, y_axis='log', sr=SR, hop_length=hop_len,
                         x_axis='time',fmax=8192)
    line1 = plt.plot(t_x1,IF_x1)
    plt.subplot(2,1,2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(x2, hop_length=hop_len,win_length=window_len)),
                            ref=np.max)
    display.specshow(D, y_axis='log', sr=SR, hop_length=hop_len,
                         x_axis='time',fmax=8192)
    line2 = plt.plot(t_x2,IF_x2)
    plt.show()
    #------------------------------------------------

    




    return

if __name__== "__main__":
    main()
