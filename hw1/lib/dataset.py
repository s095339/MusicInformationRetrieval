import os
import csv
import librosa
import numpy as np
from .define import KS_Cmajor,KS_Cminor,Key_dict,KEY_LIST
class SWDDataset:
    """
    Build a SWDDataset for global key detection and local key detection
    """
    def __init__(self, data_pth:str, data_type = "wav"): 
        self.data_pth = data_pth
        self.data_type = data_type
    def __len__(self):
        return len(self.datalist)
    def setGlobalAnotation(self, anotation_csv:str,start:int,end:int):
        self.global_anotation_csv = anotation_csv
        self.global_anotation = []
        with open(self.global_anotation_csv, 'r') as file:
            csvreader = csv.reader(file, delimiter=':')
            
            
            for row in csvreader:
                self.global_anotation.append(row)
            self.global_anotation = self.global_anotation[start:end]
        print("get annotation...")
        # store annotation in dict form
        self.global_dict = {}
        for ano in self.global_anotation:
            temp = ano[0]
            temp =temp.replace("\"","")
            WorkID,PerformanceID,key = temp.split(';')
            self.global_dict[WorkID+"_"+PerformanceID] = key
        print(self.global_dict)
        self.datalist = list(self.global_dict.keys())
        print("self.datalist=",self.datalist)

    def setLocalAnotation(self, anotation_csv_dir:str):
        self.local_anotation_csv_dir = anotation_csv_dir
        self.local_anotation_csv = os.listdir(self.local_anotation_csv_dir)
        self.datalist = os.listdir(self.data_pth)
    def getLocalAnotation(self,idx:int):
        """
        產生2(B)需要的label
        """
        audio_name = self.local_anotation_csv[idx].split('.')[0]
        local_anotation_path = os.path.join(self.local_anotation_csv_dir, self.local_anotation_csv[idx])
        audio_file_name = audio_name+"."+self.data_type
        #print(audio_file_name)
        #print(self.local_anotation_csv[idx])
        audio_path = os.path.join(self.data_pth,audio_file_name)
        audio,sr = librosa.load(audio_path)
        localKey = []
        with open(local_anotation_path,'r') as file:
            csvreader = csv.reader(file, delimiter=';')
            for row in csvreader:
                localKey.append(row)
        #print(localKey)
        #self.datalist = os.listdir(self.data_pth)
        return np.array(audio),sr,audio_name,localKey[1:]
        
    def getGlobalAnotation(self,idx:int): 
        """
        產生2(A)需要的label
        """
        audio_ = self.datalist[idx]
        label = self.global_dict[audio_]
        audio_filename = audio_+"."+self.data_type
        audio_path = os.path.join(self.data_pth,audio_filename)
        audio,sr = librosa.load(audio_path)
        return np.array(audio),sr,audio_,label

    def getSegmentAnotation(self,idx:int):
        """
        產生2(C)需要的label
        """
        audio_name = self.local_anotation_csv[idx].split('.')[0]
        local_anotation_path = os.path.join(self.local_anotation_csv_dir, self.local_anotation_csv[idx])
        audio_file_name = audio_name+"."+self.data_type
        #print(audio_file_name)
        #print(self.local_anotation_csv[idx])
        audio_path = os.path.join(self.data_pth,audio_file_name)
        audio,sr = librosa.load(audio_path)
        localKey = []
        with open(local_anotation_path,'r') as file:
            csvreader = csv.reader(file, delimiter=';')
            for row in csvreader:
                localKey.append(row)
        #print(localKey)
        interval = np.zeros(shape=(len(localKey[1:]),2),dtype = float)
        key_label = []
        for i in range(len(localKey[1:])):
            interval[i,0] = round(float(localKey[i+1][0]), ndigits = 1)#start
            interval[i,1] = round(float(localKey[i+1][1]), ndigits = 1)#stop
            key_label.append(localKey[i+1][2])
        self.datalist = os.listdir(self.data_pth)
        #print(interval)
        #print(key_label)

        return np.array(audio),sr,audio_name,key_label,interval
    

if __name__ == '__main__':

    SWDDataset()