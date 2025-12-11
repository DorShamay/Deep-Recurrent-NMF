import os
import torch
import torch.utils.data
import numpy as np
from scipy.io.wavfile import read
from stft_funcs import STFT, ISTFT

MAX_WAV_VALUE = 32768.0


class Dataset(torch.utils.data.Dataset):
    def __init__(self, len_dataset, audio_files, wavs_clean_dir, wavs_noisy_dir, audio_length, filenames):
        self.len_dataset = len_dataset
        self.audio_files = audio_files
        self.wavs_clean_dir = wavs_clean_dir
        self.wavs_noisy_dir = wavs_noisy_dir
        self.audio_length = audio_length
        self.filenames = filenames

        self.y, self.n = self.create_dataset()
        self.Y, self.Y_abs = self.process_data(noisy=False)
        self.N, self.N_abs = self.process_data(noisy=True)

    def process_data(self, noisy=True):
        stft = STFT()
        x = self.n if noisy else self.y
        x = x / MAX_WAV_VALUE
        x = torch.FloatTensor(x)
        X = stft(x)
        Fre = X.shape[1]
        X = X.permute(1, 0, 2).reshape(Fre, -1)
        X_abs = torch.abs(X)
        return X, X_abs


    def create_dataset(self):
        if len(self.audio_files) > 0.5 * self.len_dataset:
            if not os.path.exists("n_train.npy") or not os.path.exists("y_train.npy"):
                n_train = np.zeros((len(self.audio_files), self.audio_length)).astype(np.float32)
                y_train = np.zeros((len(self.audio_files), self.audio_length)).astype(np.float32)

                for ifile, audio_file in enumerate(self.audio_files):
                    print(
                        "Adding train file %d of %d total: %s" % (ifile + 1, len(self.audio_files), audio_file))

                    _, audio_clean = read(self.wavs_clean_dir + audio_file)
                    y_train[ifile, :] = audio_clean

                    _, audio_noisy = read(self.wavs_noisy_dir + audio_file)
                    n_train[ifile, :] = audio_noisy
                np.save("n_train.npy", n_train)
                np.save("y_train.npy", y_train)
            else:
                y_train = np.load("y_train.npy")
                n_train = np.load("n_train.npy")
            y = y_train
            n = n_train


        else:
            if not os.path.exists("n_valid.npy") or not os.path.exists("y_valid.npy"):
                n_valid = np.zeros((len(self.audio_files), self.audio_length)).astype(np.float32)
                y_valid = np.zeros((len(self.audio_files), self.audio_length)).astype(np.float32)

                for ifile, audio_file in enumerate(self.audio_files):
                    print("Adding valid file %d of %d total: %s" % (ifile + 1, len(self.audio_files), audio_file))

                    _, audio_clean = read(self.wavs_clean_dir + audio_file)
                    y_valid[ifile, :] = audio_clean

                    _, audio_noisy = read(self.wavs_noisy_dir + audio_file)
                    n_valid[ifile, :] = audio_noisy

                np.save("n_valid.npy", n_valid)
                np.save("y_valid.npy", y_valid)
            else:
                y_valid = np.load("y_valid.npy")
                n_valid = np.load("n_valid.npy")
            y = y_valid
            n = n_valid
        return y, n

    def __len__(self):
        return self.N.shape[1]

    def __getitem__(self, idx):
        N = self.N[:, idx]
        Y = self.Y[:, idx]

        return N, Y


