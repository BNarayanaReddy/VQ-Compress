from torch.utils.data import Dataset
import torchaudio
import os
from torchaudio.transforms import Resample

class WavDataset(Dataset):

    def __init__(self, wav_fldr):
        super().__init__()
        self.wavs = [os.path.join(wav_fldr, wav) for wav in os.listdir(wav_fldr)]
        self.sr = 16000
    
    def __getitem__(self, index):
        waveform, sr = torchaudio.load(self.wavs[index])
        if sr != self.sr:
            waveform = Resample(sr, self.sr)(waveform)
        return waveform
    
    def __len__(self):
        return len(self.wavs)
    
