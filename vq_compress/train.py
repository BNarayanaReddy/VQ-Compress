import torch
import torchaudio
from torch.utils.data import DataLoader
from wav_data import WavDataset
import os
from torchaudio.transforms import MFCC
from vq_init import ResidualVectorQuantize
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

mfcc_extractor = MFCC(
    sample_rate=16000,
    n_mfcc = 13,
    melkwargs={
        'n_fft': 200,
        "hop_length": 64
    }
)

def extract_feats(waveform: torch.Tensor) -> torch.Tensor:
    mfccs = mfcc_extractor(waveform)
    return mfccs

if __name__ == '__main__':
    n_epochs = 100
    dataset = WavDataset('wavs/')
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    rvq = ResidualVectorQuantize(
        input_dim=200,
        dim=64,
        n_q=4,
        bins=1024 
    )

    optimizer = Adam(
        rvq.parameters(), lr = 1e-3, betas=[0.9, 0.99], eps=1e-8
    )
    # lr_sch
    # Training
    optimizer.zero_grad()

    for epoch in range(n_epochs):
        for step, batch in enumerate(data_loader):
            quantized, commit_loss = rvq(batch)

            commit_loss.backward()  
            optimizer.step()
            print("Loss")
                 

        


