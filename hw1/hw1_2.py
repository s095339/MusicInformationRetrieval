import librosa
import librosa.display as display
import librosa.feature as feature
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import mir_eval   #evaluate the WA_global
import scipy.stats as stats # find the correlation coefficients

#----------------------
from dataset.dataset import SWDDataset


#KS scale template ----------
KS_Cmajor = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KS_Cminor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
   
def GenerateKStemplate(mode,shift):
    """
    Parameter
    ------------
    mode:major or minor
    shift:tone

    scale shift number for kstemplate(for both major and minor):
    C|C#|D|D#|E|F|F#|G|G#|A |A#|B 

    0|1-|2|3-|4|5-|6|7-|8|9-|10|11|
    
    Return
    ------------
    KS templates of the required scale
    """
    if mode.lower() == "major":
        init_template = KS_Cmajor
    if mode.lower() == "minor":
        init_template = KS_Cminor

    return np.roll(init_template,shift)
def Global_Key_detection():
    #prepare dataset:---------------------------------------
    pth = "./data/01_RawData/audio_wav/SC06"
    dataset = SWDDataset(data_pth = pth)
    #get global key annotation
    dataset.setGlobalAnotation(anotation_csv = "./data/02_Annotations/ann_audio_globalkey_2.csv",
                               start = 25,
                               end = 50
                               )
    #-------------------------------------------------------

    #i. Generate the K-s template:
    K_S_list = []
    for i in range(12):
        K_S_list.append(GenerateKStemplate('major',i))
    for i in range(12):
        K_S_list.append(GenerateKStemplate('minor',i))
    K_S_template = np.array(K_S_list)
    print(K_S_template)
    print("K_S_template.shape = ",K_S_template.shape)

    #for id in range(len(dataset)):
    #    audio,name,label = dataset.getGlobalAnotation(id)

    return
def main():

    #HW1_2 (a)
    Global_Key_detection()
    
    
    return





if __name__ == '__main__':
    main()
    