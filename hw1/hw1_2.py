import librosa
import librosa.display as display
import librosa.feature as feature
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import mir_eval.key as key    #evaluate the WA_global
import scipy.stats as stats # find the correlation coefficients

from tqdm import tqdm, trange
#----------------------
from lib.dataset import SWDDataset
from lib.define import KS_Cmajor,KS_Cminor,Key_dict,KEY_LIST

#KS scale template ----------
def Key2MirEvalkey(key:str):
    """
    transform the key representation of label to the mir_eval.key representation
    e.g.："C:maj"=>"C major"

    Parameter:
    --------------------
    key:str The key represented in the form of label


    return:
    --------------------
    key:str The key represented in the form of mir_eval.key
    """

    key = key.replace("maj","major")
    key = key.replace("min","minor")
    key = key.replace(":"," ")
    return key
def GenerateKStemplate(mode,shift):
    """
    Parameter
    ------------
    mode:major or minor
    shift:Tonic

    scale shift number for kstemplate(for both major and minor):
    
    Return
    ------------
    KS templates of the required scale
    """
    if mode.lower() == "major":
        init_template = KS_Cmajor
    if mode.lower() == "minor":
        init_template = KS_Cminor

    return np.roll(init_template,shift)
def GetChromaFeature(audio,sr,window_size = 2048,hop_size = 512,show = False):
    STFT = feature.chroma_stft
    CQT = feature.chroma_cqt
    CENS = feature.chroma_cens

    stft = STFT(y = audio,sr = sr,hop_length=hop_size)
    cqt = CQT(y = audio,sr = sr,hop_length=hop_size)
    cens = CENS(y = audio,sr = sr,hop_length=hop_size)
    #---
    if show:
        plt.subplot(3,1,1)
        display.specshow(stft , y_axis='log', sr=sr, win_length = window_size,hop_length=hop_size,
                            x_axis='time')
        plt.subplot(3,1,2)
        display.specshow(cqt , y_axis='log',win_length = window_size, sr=sr, hop_length=hop_size,
                            x_axis='time')
        plt.subplot(3,1,3)
        display.specshow(cens , y_axis='log', sr=sr, win_length = window_size,hop_length=hop_size,
                            x_axis='time')
        plt.show()
    #---
    return stft,cqt,cens

def CompareKStemplate(chroma,K_S_template,corr_func):
    """
    Compare the chromagram to the K_S_template
    
    Parameter:
    -----------
    chroma: the STFT,CQT,CENS chromagram
    K_S_template: K_S_template
    corr_func: correlation function 

    Return:
    ---------------
    max_id: the Key id with the biggest confidience
    """
    max_corr = 0
    max_id = 0
    #chroma_mean = chroma.mean()
    for i in range(K_S_template.shape[1]):   
        key_scale = K_S_template[i,:]  
        # I'm not sure ----------------------------------- 
        statistic = corr_func(chroma,key_scale).statistic
        if max_corr < statistic:
            max_corr = statistic
            max_id = i
        #---------------------------------------------------------
    return max_id
def Global_Key_detection():
    #prepare dataset:---------------------------------------
    pth = "./data/01_RawData/audio_wav/SC06"
    dataset = SWDDataset(data_pth = pth)
    #get global key annotation
    dataset.setGlobalAnotation(anotation_csv = "./data/02_Annotations/ann_audio_globalkey_2.csv",
                               start = 25,
                               end = 50
                               )
    WA = key.evaluate
    #------------------------------------------------------
    
    #i. Generate the K-s template:--------------------------
    K_S_list = []
    for i in range(12):
        K_S_list.append(GenerateKStemplate('major',i))
    for i in range(12):
        K_S_list.append(GenerateKStemplate('minor',i))
    K_S_template = np.array(K_S_list)
    #print(K_S_template)
    #display.specshow(K_S_template)
    #plt.show()
    print("K_S_template.shape = ",K_S_template.shape)
    #-------------------------------------------------------

    #ii Compute the chroma feature--------------------------
    #record--------------------------------------------
    #for RA
    pred_record_stft = []
    pred_record_cqt = []
    pred_record_cens = []
    answer_record = []
    #for WA
    stft_pred_WA = []
    cqt_pred_WA  = []
    cens_pred_WA  = []
    label_key_WA  = []
    #RA
    RA_stft = 0.0
    RA_cqt = 0.0
    RA_cens = 0.0
    #WA
    WA_stft = 0.0
    WA_cqt = 0.0
    WA_cens = 0.0

    progress = tqdm(total=len(dataset))

    # predict the global key for all audio 
    for idx in range(len(dataset)):
        #data prepare=======================================================
        #get audio and label
        audio,sr,name,label = dataset.getGlobalAnotation(idx)
        ans = Key_dict[label] #get the number representation of label key。
        label_key_WA.append(Key2MirEvalkey(label))
        answer_record.append(ans)
        #===================================================================

        #chromagram transfrorm==============================
        win_size = 4096
        hop_size = 1024
        stft,cqt,cens = GetChromaFeature(audio,sr,window_size=win_size,hop_size=hop_size)
        
        #first fusion
        stft_mean = np.mean(np.abs(stft),axis = 1 )
        #print(stft_mean.shape)
        cqt_mean = np.mean(np.abs(cqt),axis = 1 )
        cens_mean = np.mean(np.abs(cens),axis = 1 )
        #===================================================================


        #Predict Key==============================================================
        #stft
        pred_stft = CompareKStemplate(chroma = stft_mean,
                                K_S_template = K_S_template,
                                corr_func = stats.pearsonr
                                )
        #print(f"pred = {pred_stft},ans = {ans}\nkey_pred = {KEY_LIST[pred_stft]},key_ans = {label}")
        
        
        #cqt---
        pred_cqt = CompareKStemplate(chroma = cqt_mean,
                                K_S_template = K_S_template,
                                corr_func = stats.pearsonr
                                )    
        #print(f"pred = {pred_cqt},ans = {ans}\nkey_pred = {KEY_LIST[pred_cqt]},key_ans = {label}")
        
        #cens---
        pred_cens = CompareKStemplate(chroma = cens_mean,
                                K_S_template = K_S_template,
                                corr_func = stats.pearsonr
                                )
        #========================================================================
        

        #record the correct prediction===========================================
        #RA
        if pred_stft == ans:
            RA_stft+=1
        if pred_cqt == ans:
            RA_cqt+=1
        if pred_cens == ans:
            RA_cens+=1
        pred_record_stft.append(pred_stft)
        pred_record_cqt.append(pred_cqt)
        pred_record_cens.append(pred_cens)
        #WA
        ref_key = Key2MirEvalkey(label)
        #print(WA(ref_key,Key2MirEvalkey(KEY_LIST[pred_stft])))
        WA_stft += WA(ref_key,Key2MirEvalkey(KEY_LIST[pred_stft]))["Weighted Score"]
        WA_cqt += WA(ref_key,Key2MirEvalkey(KEY_LIST[pred_cqt]))["Weighted Score"]
        WA_cens += WA(ref_key,Key2MirEvalkey(KEY_LIST[pred_cens]))["Weighted Score"]
        #========================================================================
        progress.update(1)
    #calculate RAW accuracy-------------
    RA_stft /= len(dataset)
    RA_cqt /= len(dataset) 
    RA_cens /= len(dataset)     
    #calculate Weight Accuracy----------
    WA_stft /= len(dataset)
    WA_cqt /= len(dataset) 
    WA_cens /= len(dataset) 


    acc_dict = {
        "RA_stft":RA_stft,
        "RA_cqt":RA_cqt,
        "RA_cens":RA_cens,
        "WA_stft":WA_stft,
        "WA_cqt":WA_cqt,
        "WA_cens":WA_cens
    }
    return acc_dict 
        #print(f"WA:{WA(Key2MirEvalkey(label), Key2MirEvalkey( KEY_LIST[pred_stft]))}")
         


    """
    #late fusion
    for frame in range(stft.shape[0]):
        #print(frame)
        local_key_dect = np.zeros(shape = (12,))
        for i in range(K_S_template.shape[1]):   
            key_scale = K_S_template[i,:]  
            # I'm not sure ----------------------------------- 
            statistic = CORR(stft[:,frame],key_scale).statistic
            #--------------------------------------------------
            local_key_dect[i] = statistic
        key_record.append(np.argmax(local_key_dect))
    import statistics
    key_pre = statistics.mode(key_record)
    """
    #print(stft.shape)

    #-------------------------------------------------------

    return
def main():
    #HW1_2 (a)
    acc_dict = Global_Key_detection()
    print(acc_dict)

    return











if __name__ == '__main__':
    main()
    