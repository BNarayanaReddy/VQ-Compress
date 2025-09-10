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
    n_mfcc = 64,
    melkwargs={
        'n_fft': 200,
        "hop_length": 64
    }
)

def extract_feats(waveform: torch.Tensor) -> torch.Tensor:
    mfccs = mfcc_extractor(waveform)
    return mfccs

if __name__ == '__main__':
    n_epochs = 200
    dataset = WavDataset('../wavs/')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    rvq = ResidualVectorQuantize(
        input_dim=64,
        codebook_dim=64,
        n_q=4,
        bins=1024 
    )
    ckpt_interval = 2
    optimizer = Adam(
        rvq.parameters(), lr = 1e-2, betas=[0.9, 0.99], eps=1e-8
    )
    print(len(list(rvq.parameters())))
    # lr_sch
    lr_sch = ExponentialLR(optimizer, 0.99)
    # Training
    optimizer.zero_grad()

    

    for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        for step, batch in enumerate(data_loader):
            features = []
            for wave in batch:
                features.append(mfcc_extractor(wave))
            features = torch.cat(features, dim=0)
            print("Features Shape", features.shape)
            quantized, commit_loss = rvq(features.permute(0, 2, 1))
            # print(commit_loss)
            commit_loss = commit_loss.mean()
            # print(commit_loss)
            commit_loss.backward()  
            optimizer.step()
            print("Loss", commit_loss)
        lr_sch.step()
        if epoch % 10:
            with open("debug_ckpt_statedictWithLearnableCodebook.txt", 'a')as ff:
                print(rvq.state_dict(), file = ff)
                print("Loss: ", commit_loss, file = ff)
        #     state_dict = {
        #         "codec_model": rvq.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "lr_sch": lr_sch.state_dict()
        #     }
        #     torch.save(
        #         state_dict, f"ckpt_{epoch}_{commit_loss}" 
        #     )

        


