import os
import csv
import librosa

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
        self.anotation_csv = anotation_csv
        self.global_anotation = []
        with open(self.anotation_csv, 'r') as file:
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

    def setLocalAnotation(self, anotation_csv:str,start:int,end:int):
        pass

    def getGlobalAnotation(self,id:int): 
        """
        parameter
        -----------------
        id: the index of song

        return
        -----------------
        audio:np.ndarray readed by librosa.load(audio_)
        audio_:audio name
        label:global annotation 
        """
        audio_ = self.datalist[id]
        label = self.global_dict[audio_]
        audio_filename = audio_+"."+self.data_type
        audio_path = os.path.join(self.data_pth,audio_filename)
        audio = librosa.load(audio_path)
        return audio,audio_,label

if __name__ == '__main__':

    SWDDataset()