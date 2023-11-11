import matplotlib.pyplot as plt
import numpy as np
import librosa
from dataset import BeatDataset
import mir_eval
def main(Dataset):
    score = []

    for idx in range(len(Dataset)):
        #get audio and beat
        try:
            info = Dataset.get_beat_Annotation(idx)
        except:
            continue
        #get audio info-----
        audio = info["audio"]
        audio_name = info["name"]
        sr = info["sr"]
        beats = info["beats"]
        #----
        #print(beats)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr,
                                         aggregate=np.median)
        
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        est_beats = librosa.frames_to_time(beats, sr=sr)
        
        f_score = mir_eval.beat.f_measure(reference_beats=beats, estimated_beats= est_beats)
        #print(f_score)
        score.append(f_score)
    print(f"score = {sum(score)/len(score)}")
if __name__ == '__main__':
    Dataset = BeatDataset(
        datapath = "./data/BallroomData/allwav",
        annotation_beat_path ="./data/BallroomAnnotations-master" ,
        annotation_bpm_path = "./data/BallroomAnnotations/ballroomGroundTruth",
    )

    main(Dataset)