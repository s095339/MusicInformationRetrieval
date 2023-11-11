
from dataset import BeatDataset
import matplotlib.pyplot as plt
import numpy as np
import librosa
import  mir_eval


def main(Dataset):
    correct = 0
   
    correct_beat =  [0,0,0,0,0,0]
    correct_beat_len = [0,0,0,0,0,0]
    for idx in range(len(Dataset)):
        try:
            info = Dataset.get_meter_Annotation(idx)
        except:
            print(idx)
            continue
        audio = info["audio"]
        audio_name = info["name"]
        sr = info["sr"]
        meter_label = info["meter"]
        correct_beat_len[meter_label]+=1
        #print("meter = ",meter_label)
        # get onset envelope
        onset_env = librosa.onset.onset_strength(y =audio, sr=sr, aggregate=np.median)
        # get tempo and beats
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        # we assume 4/4 time
        meter_strength_list = []
          
        meter_strength_list.append(0)
        meter_strength_list.append(0)
        for meter in range(2,5):
            

            beat_strengths = onset_env[beats] #每個beat的能量

            
            measures = (len(beats) // meter) #總共的小節數
            measure_beat_strengths = beat_strengths[:measures * meter].reshape(-1, meter)
            # add up strengths per beat position
            length = measure_beat_strengths.shape[0]
            #print(measure_beat_strengths )
            beat_pos_strength = np.sum(measure_beat_strengths, axis=0)#
            
            downbeat_pos = np.argmax(beat_pos_strength) #最大的為down beat，可以偵測弱起拍
            max_beat_pos_strength = np.max(beat_pos_strength)
            # convert the beat positions to the same 2d measure format
            meter_strength_list.append(max_beat_pos_strength/length)

        max_strength = max(meter_strength_list)
        meter_est = meter_strength_list.index(max_strength)
        #print(meter_strength_list)
        #print("meter_est=",meter_est)
        if meter_est == meter:
            correct = correct + 1
            correct_beat[meter] = correct_beat[meter] + 1
    print("accruacy = ",correct/len(Dataset))
    for i in range(2,5):
        if correct_beat_len[i]>0:
            print(f"{i} beats acc = {correct_beat[i]/correct_beat_len[i]}")

if __name__ == '__main__':
    Dataset = BeatDataset(
        datapath = "./data/BallroomData/allwav",
        annotation_beat_path ="./data/BallroomAnnotations-master" ,
        annotation_bpm_path = "./data/BallroomAnnotations/ballroomGroundTruth",
    )
    #Q1
    
    main(Dataset)