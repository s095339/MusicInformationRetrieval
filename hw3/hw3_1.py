from dataset import BeatDataset
import matplotlib.pyplot as plt
import numpy as np
import librosa
import  mir_eval

def validate(T1,T2,S1,G):
    #Tti--------
    Tt1 = 0
    if(abs((G-T1)/G)<=0.08):Tt1 = 1
    Tt2 = 0
    if(abs((G-T2)/G)<=0.08):Tt2 = 1
    #P score----
    P = S1*Tt1 + (1-S1)*Tt2
    #P ALOTC----
    P_ALOTC=0
    if(Tt1 or Tt2):P_ALOTC=1
    return P,Tt1, Tt2, P_ALOTC

def main(Dataset,win_length = 384,win_second = 2,win_length_unit = "npoints", early_fusion = False):
    

    #Dataset.get_bpm_Annotation(5)
    #tempo_estimate------------------------
    #Q1
    AC_P_avg = []
    AC_P_ALOTC_avg = []
    Fourier_P_avg = []
    Fourier_P_ALOTC_avg = []

    sample_interval = 100
    hop_length = 512
    win_length = 384
    
    for idx in range(len(Dataset)):
        #get audio and pbm
        info = Dataset.get_bpm_Annotation(idx)
        #get audio info-------
        audio = info["audio"]
        audio_name = info["name"]
        sr = info["sr"]
        G = info["bpm"]
        #print("bpm = ",G)
        #----------------------
        if(win_length_unit == "second"): #Q2 以秒為單位調整win_length大小
            win_length = int(sr*win_second/hop_length)
            #print("win_length = ",win_length," points")
        
        oenv = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length )
        plt.plot(oenv)
        plt.show()
        # ACF ========================================================================================================
        
        #ac_T1 = librosa.feature.tempo(onset_envelope=oenv, sr=sr,hop_length=hop_length)[0]
        
        #tempogram feature extraction
        ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                              hop_length=hop_length)
        
        #librosa.display.specshow(data = ac_tempogram, sr = sr, hop_length = hop_length, win_length=win_length)

        #plt.show() 

         
        if early_fusion:
            ac_temp_avg = np.average(abs(ac_tempogram), axis = 1)
        else:
            ac_temp_avg = np.average(abs(ac_tempogram[:,20:sample_interval]), axis = 1)
        #tempo_frequence
        ac_freq = librosa.tempo_frequencies(ac_tempogram.shape[0],sr = sr,hop_length = hop_length )
        ac_freq = list(ac_freq)
        
        start_index=0
        while(ac_freq[start_index]>=250): #去除不合理的bpm value
            start_index=start_index+1

        
        ac_T1_index = np.argmax(ac_temp_avg[start_index:])+start_index
        ac_T1 = ac_freq[ac_T1_index]
        
        
        #print(ac_freq)
        
        #ac_T1_index = ac_freq.index(ac_T1)
        ac_temp_avg[ac_T1_index-1:ac_T1_index+2] = 0 #刪除相鄰值
        ac_T2_index = np.argmax(ac_temp_avg[start_index:])+start_index 
        ac_T2 = ac_freq[ac_T2_index]
        #print(ac_T1)
        #print(ac_T2)
        S1 = ac_temp_avg[ac_T1_index]/(ac_temp_avg[ac_T1_index]+ac_temp_avg[ac_T2_index])
        P,Tt1, Tt2, P_ALOTC = validate(ac_T1,ac_T2,S1,G) #calculate P value and P_alotc
        
        AC_P_avg.append(P)
        AC_P_ALOTC_avg.append(P_ALOTC)
        #==========================================================================================

        #Fourier===================================================================================
        
        
        
        #tempogram feature extraction
        Fourier_tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr,
                                              hop_length=hop_length)
        #librosa.display.specshow(data = Fourier_tempogram, sr = sr, hop_length = hop_length, win_length=win_length)

        #plt.show() 
        
        #tempo_frequence
        Fourier_freq = librosa.fourier_tempo_frequencies(sr = sr,hop_length = hop_length,win_length=win_length )
        Fourier_freq = list(Fourier_freq)
        start_index = 5
        end_index=0
        while(Fourier_freq[end_index]<=250): #去除不合理的bpm value
            end_index=end_index+1
        #print(Fourier_freq)

        #print(ac_freq)
        if early_fusion:
            Fourier_temp_avg = np.average(abs(Fourier_tempogram), axis = 1)
        else:
            Fourier_temp_avg = np.average(abs(Fourier_tempogram[:,20:sample_interval]), axis = 1)
        #print(Fourier_temp_avg)
        #print(Fourier_temp_avg)
        Fourier_T1_index = np.argmax(Fourier_temp_avg[start_index:end_index])+start_index
        Fourier_T1 = Fourier_freq[Fourier_T1_index]
        Fourier_temp_avg[Fourier_T1_index-1:Fourier_T1_index+2] = 0
        Fourier_T2_index = np.argmax(Fourier_temp_avg[start_index:end_index])+start_index
        Fourier_T2 = Fourier_freq[Fourier_T2_index]
        #print(Fourier_T1)
        #print(Fourier_T2)
        S1 = Fourier_temp_avg[Fourier_T1_index]/(Fourier_temp_avg[Fourier_T1_index]+Fourier_temp_avg[Fourier_T2_index])
        P,Tt1, Tt2, P_ALOTC = validate(Fourier_T1,Fourier_T2,S1,G) #calculate P value and P_alotc
        
        Fourier_P_avg.append(P)
        Fourier_P_ALOTC_avg.append(P_ALOTC)
        #==========================================================================================
    print("AutoCorrelation tempogram-----------------------")
    print("Average P value = ",sum(AC_P_avg)/len(AC_P_avg))
    print("Average P_ALOTC = ",sum(AC_P_ALOTC_avg)/len(AC_P_ALOTC_avg))
    print("Fourier tempogram-------------------------------")
    print("Average P value = ",sum(Fourier_P_avg)/len(Fourier_P_avg))
    print("Average P_ALOTC = ",sum(Fourier_P_ALOTC_avg)/len(Fourier_P_ALOTC_avg))
if __name__ == '__main__':
    Dataset = BeatDataset(
        datapath = "./data/BallroomData/allwav",
        annotation_beat_path ="./data/BallroomAnnotations-master" ,
        annotation_bpm_path = "./data/BallroomAnnotations/ballroomGroundTruth",
    )
    #Q1
    print("------Q1--------")
    main(Dataset)
    print("------Q2--------")
    print("win length = 4s")
    main(Dataset,win_second = 4,win_length_unit = "second")
    print("win length = 8s")
    main(Dataset,win_second = 8,win_length_unit = "second")
    print("win length = 12s")
    main(Dataset,win_second = 12,win_length_unit = "second")
    print("------Q3-------")
    main(Dataset,early_fusion=True)