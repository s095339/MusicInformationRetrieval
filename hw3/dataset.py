import os
import librosa
class BeatDataset:
    def __init__(self,datapath,annotation_beat_path,annotation_bpm_path):
        # path
        self.datapath = datapath
        self.beat_anno_path = annotation_beat_path
        self.bpm_anno_path = annotation_bpm_path
        #read wav data
        self.datalist = os.listdir(datapath)
        #print(self.datalist)
    def __len__(self):
        return len(self.datalist)
    def get_bpm_Annotation(self, idx):
        wav_file_name = self.datalist[idx]
        wav_path = os.path.join(self.datapath, wav_file_name)
        audio, sr = librosa.load(wav_path)
        name = wav_file_name.replace(".wav","")

        #beat annotation
        #beat_anno_pth = os.path.join(self.beat_anno_path,name,".beat")
        #beat_anno = 

        #bpm annotation
        bpm_anno_pth = os.path.join(self.bpm_anno_path,name+".bpm")
        with open(bpm_anno_pth , "r") as bpm_f:
            bpm = int(bpm_f.read())
            #print("name = ",name,"bpm = ",bpm)
        
        return {
            "audio":audio,
            "sr":sr,
            "name":name,
            "bpm":bpm
        }

    def get_beat_Annotation(self, idx):
        wav_file_name = self.datalist[idx]
        wav_path = os.path.join(self.datapath, wav_file_name)
        audio, sr = librosa.load(wav_path)
        name = wav_file_name.replace(".wav","")

        #beat annotation
        #beat_anno_pth = os.path.join(self.beat_anno_path,name,".beat")
        #beat_anno = 
        beat_anno_pth = os.path.join(self.beat_anno_path,name+".beats")
        with open(beat_anno_pth , "r") as beat_f:
            beat = beat_f.readlines()
        beats = []
        for element in beat:
            beats.append(float(element.split(" ")[0]))

        return {    
            "audio":audio,
            "sr":sr,
            "name":name,
            "beats":beats
        }
    def get_meter_Annotation(self,idx):
        wav_file_name = self.datalist[idx]
        wav_path = os.path.join(self.datapath, wav_file_name)
        audio, sr = librosa.load(wav_path)
        name = wav_file_name.replace(".wav","")

        #beat annotation
        #beat_anno_pth = os.path.join(self.beat_anno_path,name,".beat")
        #beat_anno = 
        beat_anno_pth = os.path.join(self.beat_anno_path,name+".beats")
        with open(beat_anno_pth , "r") as beat_f:
            beat = beat_f.readlines()
        beats = []
      
        for element in beat:
            beats.append(int(element.split(" ")[1]))
        meter = max(beats)
        
        return {    
            "audio":audio,
            "sr":sr,
            "name":name,
            "meter":meter
        }