import librosa
import librosa.display as display
import librosa.feature as feature
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import mir_eval
import mir_eval.key as key    #evaluate the WA_global
import scipy.stats as stats # find the correlation coefficients

from tqdm import tqdm, trange
#----------------------
from lib.dataset import SWDDataset
from lib.define import KS_Cmajor,KS_Cminor,Key_dict,KEY_LIST

#KS scale template ----------
def cossimarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

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
        try:
            statistic = corr_func(chroma,key_scale).statistic
        except:
            statistic = corr_func(chroma,key_scale)
        if max_corr < statistic:
            max_corr = statistic
            max_id = i
        #---------------------------------------------------------
    return max_id
def Global_Key_detection():
    #prepare dataset:---------------------------------------
    #準備資料集
    pth = "./data/01_RawData/audio_wav/SC06"
    dataset = SWDDataset(data_pth = pth)
    #get global key annotation
    dataset.setGlobalAnotation(anotation_csv = "./data/02_Annotations/ann_audio_globalkey_2.csv",
                               start = 25,
                               end = 50
                               )
    #WA要用的函數
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

    #progress = tqdm(total=len(dataset))

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
        #cqt---
        pred_cqt = CompareKStemplate(chroma = cqt_mean,
                                K_S_template = K_S_template,
                                corr_func = stats.pearsonr
                                )    
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
        #progress.update(1)
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

def Local_Key_detection(frame = 60):
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
    WA = key.evaluate
    #-------------------------------------------------------
    
    
    pth = "./data/01_RawData/audio_wav/SC06"
    dataset = SWDDataset(data_pth = pth)
    #get global key annotation
    dataset.setLocalAnotation(anotation_csv_dir = "./data/02_Annotations/ann_audio_localkey-ann1")
    dataset.getLocalAnotation(5)
    
   
    #總共的ACC-------------------
    stft_local_total_RA = 0
    cqt_local_total_RA = 0
    cens_local_total_RA = 0
    stft_local_total_WA = 0
    cqt_local_total_WA = 0
    cens_local_total_WA = 0
    total_piece = 0
    #--------------------------
    for idx in range(len(dataset)):#選一首歌
        
        audio,sr,name,localKey_anno = dataset.getLocalAnotation(idx)
        win_size = int(sr/10)
        hop_size = int(sr/10)
        #把0.1秒所包含的東西都平均起來：
        #hop_size = sr/10 win_size = sr/10
        duration = librosa.get_duration(y=audio,sr=sr)
        stft,cqt,cens = GetChromaFeature(audio,sr,window_size=win_size,hop_size=hop_size)
        #得到一個以0.1秒為單位的CHROMAGRAM
        #把local key的label表示方法做修改以方便做計算:--------
        #print(localKey_anno)
        #時間以0.1秒為單位，將label的時間做四捨五入
        first_key_time = int(round(float(localKey_anno[0][0]),ndigits = 1)*10)
        last_key_endtime = int(round(float(localKey_anno[-1][1]),ndigits = 1)*10)
        time_list = []
        local_key_list = []
        for key_anno in localKey_anno:
            time_list.append( int(round(float(key_anno[0]),ndigits = 1)*10))
            local_key_list.append(Key_dict[key_anno[2]])
        time_list.append(last_key_endtime) #把結束時間append進去，所以會比keylist多一個element
        
        keylabel = {
            "start":first_key_time,
            "end":last_key_endtime,
            "time_list":time_list,
            "key_list":local_key_list
        }

        #print(keylabel)
        #---------------------------------------------------
        #Local_Key_detection
        #產生對每一個0.1秒的調性判斷
        stft_local_key_list = []
        cqt_local_key_list = []
        cens_local_key_list = []
        for time in range(stft.shape[1]):
            start = max(0,time-int(frame/2))
            end = min(stft.shape[1],time+int(frame/2))
            st = stft[:,start:end].copy()
            cq =cqt[:,start:end].copy()
            ce = cens[:,start:end].copy()
            stft_local_frame_mean = np.mean(st,axis=1)
            cqt_local_frame_mean = np.mean(cq,axis=1)
            cens_local_frame_mean = np.mean(ce,axis=1)
            #print(stft_local_frame_mean.shape)
            pred_stft = CompareKStemplate(chroma = stft_local_frame_mean,
                                    K_S_template = K_S_template,
                                    corr_func = stats.pearsonr
                                    )

            #cqt--- 
            
            pred_cqt = CompareKStemplate(chroma = cqt_local_frame_mean,
                                    K_S_template = K_S_template,
                                    corr_func = stats.pearsonr
                                    )    
        
            #cens---
            pred_cens = CompareKStemplate(chroma = cens_local_frame_mean,
                                    K_S_template = K_S_template,
                                    corr_func = stats.pearsonr
                                    )
            #========================================================================
            #===================================
            stft_local_key_list.append(pred_stft)
            cqt_local_key_list.append(pred_cqt)
            cens_local_key_list.append(pred_cens)

        #----------------------------------------------------
        #stft_local_key_list 
        #cqt_local_key_list 
        #cens_local_key_list 
        #此時這三個LIST分別記錄了用不同方法預測的lOCAL KEY
        #------
        #acc -----------------------------------------------------
        stft_local_key_correct = 0
        cqt_local_key_correct  = 0
        cens_local_key_correct  = 0
        stft_local_key_WA_score = 0
        cqt_local_key_WA_score = 0
        cens_local_key_WA_score = 0
        stft_local_key_RA = 0
        cqt_local_key_RA  = 0
        cens_local_key_RA  = 0
        stft_local_key_WA = 0
        cqt_local_key_WA  = 0
        cens_local_key_WA  = 0
        current_key_id = 0
        #只從樂曲開始的地方判斷
        total = (keylabel["end"]-keylabel["start"]+1)  #只從樂曲開始的地方開始 為了算acc
        total_piece += total
        for key_id in range(len(stft_local_key_list)):
            stft_local_key = stft_local_key_list[key_id]
            cqt_local_key = cqt_local_key_list[key_id] 
            cens_local_key = cens_local_key_list[key_id]
            if key_id < keylabel["start"]:#只從樂曲開始的地方判斷
                continue
            elif key_id >= keylabel["end"]:#沒有聲音之後就結束
                break
            else: 
                if key_id>=keylabel["time_list"][current_key_id+1]:
                    current_key_id+=1

            current_key_label = keylabel["key_list"][current_key_id]
        
            if stft_local_key == current_key_label:
                stft_local_key_correct+=1
                stft_local_total_RA +=1
            if cqt_local_key == current_key_label:
                cqt_local_key_correct+=1
                cqt_local_total_RA += 1
            if cens_local_key == current_key_label:
                cens_local_key_correct+=1
                cens_local_total_RA += 1
            #計算WA-------------------------------------------
            current_key_label_mirEval = Key2MirEvalkey(KEY_LIST[current_key_label])
            stft_local_key_WA_score += WA(current_key_label_mirEval,Key2MirEvalkey(KEY_LIST[stft_local_key]))["Weighted Score"]
            cqt_local_key_WA_score  += WA(current_key_label_mirEval,Key2MirEvalkey(KEY_LIST[cqt_local_key]))["Weighted Score"]
            cens_local_key_WA_score += WA(current_key_label_mirEval,Key2MirEvalkey(KEY_LIST[cens_local_key]))["Weighted Score"]
            #計算所有24首歌的WA------------------------------------------------
            
            stft_local_total_WA += WA(current_key_label_mirEval,Key2MirEvalkey(KEY_LIST[stft_local_key]))["Weighted Score"]
            cqt_local_total_WA += WA(current_key_label_mirEval,Key2MirEvalkey(KEY_LIST[cqt_local_key]))["Weighted Score"]
            cens_local_total_WA+= WA(current_key_label_mirEval,Key2MirEvalkey(KEY_LIST[cens_local_key]))["Weighted Score"]
            #-------------------------------------------------------------------------------------------------------------
        #計算單首歌RA------------------------------------------
        stft_local_key_RA = stft_local_key_correct/total
        cqt_local_key_RA = cqt_local_key_correct/total
        cens_local_key_RA = cens_local_key_correct/total

        #計算單首歌WA-----------------------------------------
        stft_local_key_WA = stft_local_key_WA_score/total
        cqt_local_key_WA = cqt_local_key_WA_score/total
        cens_local_key_WA = cens_local_key_WA_score/total

        #---------------------------------------------------------
        print("audio name=",name)
        print("stft:")
        print(f"RA = {stft_local_key_RA } WA = {stft_local_key_WA }")
        print("cqt:")
        print(f"RA = {cqt_local_key_RA } WA = {cqt_local_key_WA }")
        print("cens:")
        print(f"RA = {cens_local_key_RA } WA = {cens_local_key_WA }")

    #計算整體RA WA
    stft_local_total_RA = stft_local_total_RA/total_piece
    cqt_local_total_RA = cqt_local_total_RA/total_piece
    cens_local_total_RA = cens_local_total_RA/total_piece

    stft_local_total_WA = stft_local_total_WA/total_piece
    cqt_local_total_WA = cqt_local_total_WA/total_piece
    cens_local_total_WA = cens_local_total_WA/total_piece

    print("total RA WA-------")
    print("stft:")
    print(f"RA = {stft_local_total_RA } WA = {stft_local_total_WA }")
    print("cqt:")
    print(f"RA = {cqt_local_total_RA } WA = {cqt_local_total_WA }")
    print("cens:")
    print(f"RA = {cens_local_total_RA } WA = {cens_local_total_WA }")
    
    return {
        "stft_RA":stft_local_total_RA,
        "cqt_RA":cqt_local_total_RA, 
        "cens_RA":cens_local_total_RA,
        "stft_WA":stft_local_total_WA,
        "cqt_WA":cqt_local_total_WA,
        "cens_WA":cens_local_total_WA
    }

def Segmentation(frame = 200):
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
    
    
    pth = "./data/01_RawData/audio_wav/SC06"
    dataset = SWDDataset(data_pth = pth)
    #get global key annotation
    dataset.setLocalAnotation(anotation_csv_dir = "./data/02_Annotations/ann_audio_localkey-ann1")
    #dataset.getSegmentAnotation(0)
    #return
   
    for idx in range(len(dataset)):#選一首歌
        audio,sr,name,key_label,interval = dataset.getSegmentAnotation(idx)
        win_size = int(sr/10)
        hop_size = int(sr/10)
        #把0.1秒所包含的東西都平均起來：
        #hop_size = sr/10 win_size = sr/10
        duration = librosa.get_duration(y=audio,sr=sr)
        stft,cqt,cens = GetChromaFeature(audio,sr,window_size=win_size,hop_size=hop_size)
        #得到一個以0.1秒為單位的CHROMAGRAM
        #把local key的label表示方法做修改以方便做計算:--------
        #print(localKey_anno)
        #時間以0.1秒為單位，將label的時間做四捨五入
        
    
        
        keylabel = {
            "start":interval[0,0],
            "end":interval[-1,1],
            "ref_intervals":interval,
            "ref_labels":key_label
        }

        #print(keylabel)
        #---------------------------------------------------
        #Local_Key_detection
        #產生對每一個0.1秒的調性判斷
        stft_local_key_list = []
        cqt_local_key_list = []
        cens_local_key_list = []
        for time in range(stft.shape[1]):
            start = max(0,time-int(frame/2))
            end = min(stft.shape[1],time+int(frame/2))
            st = stft[:,start:end].copy()
           
            stft_local_frame_mean = np.mean(st,axis=1)
           
            pred_stft = CompareKStemplate(chroma = stft_local_frame_mean,
                                    K_S_template = K_S_template,
                                    corr_func = stats.pearsonr
                                    )
            #========================================================================
            #===================================
            stft_local_key_list.append(pred_stft)
        start = int(keylabel["start"]*10)
        stop = int(keylabel["end"]*10)
        
        est_interval_temp = []
        est_key_list = []
        current_key = 0
        for i in range(len(stft_local_key_list[start:stop+1])):
            t  = i + start
            if t == start:
                est_interval_temp.append(start/10)
                last_key = stft_local_key_list[t]
                continue
            if t == stop:
                est_interval_temp.append((stop)/10)
                est_key_list.append(KEY_LIST[last_key])
                break
            
            if stft_local_key_list[t]!=last_key:
                est_interval_temp.append(t/10)
                est_key_list.append(KEY_LIST[last_key])
                last_key = stft_local_key_list[t]
        
        est_interval = np.zeros(shape = (len(est_interval_temp)-1,2))
    
        last = 0
        for i in range(len(est_interval_temp)):
            if i == 0:
                last = est_interval_temp[i]
                continue
            #print("last = ",last)
            est_interval[i-1,0] = last
            est_interval[i-1,1] = est_interval_temp[i]
            last = est_interval_temp[i]
        
        #print(est_interval_temp)
        print(est_interval)
        print(est_key_list)
        
        print(keylabel["ref_intervals"])
        print(keylabel["ref_labels"])

        results = mir_eval.chord.evaluate(keylabel["ref_intervals"], keylabel["ref_labels"], est_interval, est_key_list)
        print("audioname = ", name)
        underseg_score = results['underseg']
        overseg_score = results['overseg']
        avg_score = (underseg_score+overseg_score)/2

        print("Over-segmentation: {:.3f}".format(underseg_score))
        print("Under-segmentation: {:.3f}".format(overseg_score))
        print("Average segmentation: {:.3f}".format(avg_score ))
        
        



def main():
    #HW1_2 (a)
    acc_dict = Global_Key_detection()
    print(acc_dict)
    #HW1_2 (b)
    Local_Key_detection(200)
    #HW1_2 (c)
    Segmentation(200)

    


if __name__ == '__main__':
    main()
    