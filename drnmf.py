import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import copy
import numpy as np
from numpy import random
import scipy.io as sio
import scipy.io.wavfile
from numpy import random
np.random.seed(7654)
from sklearn.decomposition import NMF


import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Dataset
from torch.utils.data import DataLoader

from torchmetrics.functional.audio import \
    scale_invariant_signal_distortion_ratio as si_sdr

from scipy.linalg import eigvalsh
import numpy as np

import matplotlib.pyplot as plt

from stft_funcs import STFT, ISTFT

import sys

torch.manual_seed(0)
# torch.set_default_dtype(torch.float64)
torch.set_default_dtype(torch.float32)


MAX_WAV_VALUE = 32768.0
AUDIO_LENGTH = 32000
# AUDIO_LENGTH = 16000
SAMPLING_RATE = 16000
BATCH_SIZE = 126
# BATCH_SIZE = 18
CHECK_POINT_PATH = 'checkpoints/drnmf_net.pth'




def sparse_nmf_on_chunk(V, params, verbose=True):
    (m, n) = V.shape
    params_save = copy.deepcopy(params)
    params_save.update({'display': float(verbose)})
    r = int(params['r'])
    max_iter = int(params['max_iter'])

    if 'init_w' in params.keys():
        # W_init = params['init_w']
        W_init = params['init_w'].copy()
        H_init = random.rand(r, V.shape[1]).astype(np.float32)
        model = NMF(n_components=r, init='custom', solver='mu', beta_loss=2.0, tol=params['conv_eps'],
                    max_iter=max_iter, verbose=verbose)
        W = model.fit_transform(V, H=H_init, W=W_init).astype(np.float32)
        H = model.components_
    else:
        W_init = random.rand(V.shape[0], r).astype(np.float32)
        H_init = random.rand(r, V.shape[1]).astype(np.float32)
        model = NMF(n_components=r, init='custom', solver='mu', beta_loss=2.0, tol=params['conv_eps'],
                    max_iter=max_iter, verbose=verbose)
        W = model.fit_transform(V, H=H_init, W=W_init)
        H = model.components_

    return W, H


def sparse_nmf(V, params):
    # make a copy of the params dictionary, since we might modify it
    params_copy = copy.deepcopy(params)

    # get the shape of the data and determine the number of chunks
    (n_feats, n_frames) = V.shape
    r = int(params['r'])
    r_for_max_frame_batch_size = int(params['r'])
    max_frame_batch_size = BATCH_SIZE  # max number of frames
    frame_batch_size = int(float(max_frame_batch_size) * (float(r_for_max_frame_batch_size) / float(r)))
    n_chunks = int(np.ceil(float(n_frames) / float(frame_batch_size)))

    H = np.zeros((r, n_frames))

    # W dictionary update using batches of H
    for i in range(n_chunks):
        print("sparse NMF: processing chunk %d of %d..." % (i + 1, n_chunks))
        start_idx = i * frame_batch_size
        end_idx = (i + 1) * frame_batch_size
        W, H_tmp = sparse_nmf_on_chunk(V[:, start_idx:end_idx], params_copy)

        # update the current dictionary:
        if 'w_update_ind' in params_copy.keys():
            idx_update = np.where(params_copy['w_update_ind'])[0]
            params_copy['init_w'][:, idx_update] = W[:, idx_update]
        else:
            params_copy['init_w'] = W

        H[:, start_idx:end_idx] = H_tmp

    return W, H


def train_snmf(clean_frames, noise_frames, params_snmf):
    # train SNMF on clean speech
    W_y, _ = sparse_nmf(clean_frames, params_snmf)
    # train SNMF with noise and speech, fixing the clean speech dictionary
    print("Training SNMF with noise and speech...")
    W_init = np.concatenate((W_y, np.random.rand(*W_y.shape).astype(np.float32)), axis=1)
    idx_update = np.concatenate(
        (np.zeros(int(params_snmf['r']), dtype=bool), np.ones(int(params_snmf['r']), dtype=bool)))

    params_snmf_noise = copy.deepcopy(params_snmf)
    params_snmf_noise.update({'r': 2 * params_snmf['r'],
                              'init_w': W_init,
                              'w_update_ind': idx_update})

    W, H = sparse_nmf(noise_frames, params_snmf_noise)
    return W, H


def train(model, train_loader, valid_loader, num_epochs, optimizer, scheduler, start_epoch, loss_train, loss_valid, device):
    """Train a network.
    Returns:
        loss_valid {numpy} -- loss function values on valid set
    """
    # train_log_file = open(
    #     'train_logs.txt',
    #     'w')
    # sys.stdout = train_log_file


    # Main loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        for step, (b_x, b_y) in enumerate(train_loader):
            b_x = b_x.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)

            b_x_abs = torch.abs(b_x).to(torch.float32)


            h_hat = model(b_x_abs)

            ## Mask calculation
            W_clean, W_noise = torch.split(model.A_list[model.K-1].weight.t(), 200, dim=1)
            H_clean, H_noise = torch.split(h_hat, 200, dim=1)

            Y = W_clean @ H_clean.t()
            V = W_noise @ H_noise.t()

            Mask = Y / (1e-9 + Y + V)

            est_istft = istft(Mask.unsqueeze(0) * b_x.t().unsqueeze(0), AUDIO_LENGTH)
            clean_istft = istft(b_y.t().unsqueeze(0), AUDIO_LENGTH)

            loss = -si_sdr(preds=est_istft, target=clean_istft).mean() + 50

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            for k in range(K):
                model.A_list[k].weight.data.clamp_(0)
            model.h_0_K.data.clamp_(0)  # ensure non-negative vector
            train_loss += loss.data.item() - 50
            # if step == 0 or (step + 1) % 2000 == 0:
            #     print("Epoch %d, Step %d, Train loss %.8f" % (epoch, (step + 1), train_loss / (step + 1)))
            #     # train_log_file.flush()

        loss_train[epoch] = train_loss / len(train_loader)

        # validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(valid_loader):
                b_x = b_x.to(device, non_blocking=True)
                b_y = b_y.to(device, non_blocking=True)

                b_x_abs = torch.abs(b_x).to(torch.float32)

                h_hat = model(b_x_abs)

                W_clean, W_noise = torch.split(model.A_list[model.K - 1].weight.t(), 200, dim=1)
                H_clean, H_noise = torch.split(h_hat, 200, dim=1)

                Y = W_clean @ H_clean.t()
                V = W_noise @ H_noise.t()

                Mask = Y / (1e-9 + Y + V)

                est_istft = istft(Mask.unsqueeze(0) * b_x.t().unsqueeze(0), AUDIO_LENGTH)
                clean_istft = istft(b_y.t().unsqueeze(0), AUDIO_LENGTH)
                valid_loss += -si_sdr(preds=est_istft, target=clean_istft).mean().item()

                # if step == 0 or (step + 1) % 1000 == 0:
                #     print("Epoch %d, Step %d, Valid loss %.8f" % (epoch, (step + 1), valid_loss / (step + 1)))

            loss_valid[epoch] = valid_loss / len(valid_loader)
            last_lr = optimizer.param_groups[0]['lr']

            print(
                "Epoch %d, Train loss %.8f, Validation loss %.8f, with learning rate {0}".format(last_lr)
                % (epoch, loss_train[epoch], loss_valid[epoch])
            )
            scheduler.step(loss_valid[epoch])
            # train_log_file.flush()

            checkpoint_path = CHECK_POINT_PATH.split("/")[0] + "/epoch={}_".format(epoch) + CHECK_POINT_PATH.split("/")[1]
            print("Saving checkpoint to {}".format(checkpoint_path))
            dic = {
                'drnmf_net': model.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss_train': loss_train,
                'loss_valid': loss_valid
            }
            torch.save(dic, checkpoint_path)
            print("Complete.")
        torch.cuda.empty_cache()
    return loss_train, loss_valid


class LISTA_Model(nn.Module):
    def __init__(self, n, m, K=2, rho=0.0001, W=None, h0=None):
        super(LISTA_Model, self).__init__()
        self.n, self.m = n, m
        self.W = W
        self.K = K  # ISTA Iterations = K
        self.rho = rho  # Lagrangian Multiplier

        self.A_list = nn.ModuleList([nn.Linear(n, m, bias=False) for _ in range(K)])

        if W is not None:
            L = float(eigvalsh(W.t() @ W, eigvals=(m - 1, m - 1)))
            # L = float(eigvalsh(W.t() @ W, subset_by_index=[m - 1, m - 1]))
            self.alpha = nn.Parameter(torch.ones(K + 1, 1, 1)*L, requires_grad=True)
            self.h_0_K = nn.Parameter(h0, requires_grad=True)
        else:
            self.beta = nn.Parameter(torch.ones(K + 1, 1, 1), requires_grad=True)
            self.mu = nn.Parameter(torch.ones(K + 1, 1, 1), requires_grad=True)
            self.h_0_K = nn.Parameter(torch.zeros(m,1), requires_grad=True)
        # Initialization
        if W is not None:
            for k in range(K):
                self.A_list[k].weight.data = W.t()

    def _shrink(self, h, beta):
        h_shrink = beta * F.softshrink(h / beta, lambd=self.rho)
        return F.relu(h_shrink)

    def forward(self, x_batch):
        h_t_batch = torch.zeros(x_batch.size(0),self.m).to(x_batch.device)
        for t in range(x_batch.size()[0]):
            if t == 0:
                h_t = self.h_0_K.t().to(x_batch.device)
            else:
                h_t = h_t_minus_one
            for k in range(1,self.K+1):
                B = self.A_list[k-1].weight @ self.A_list[k-1].weight.T
                h_t = self._shrink(h_t - (1/self.alpha[k, :, :]) * h_t @ B.T + (1/self.alpha[k, :, :]) * self.A_list[k-1](x_batch[t,:].unsqueeze(0)), (1/self.alpha[k, :, :]))
            h_t_batch[t,:] = h_t
            h_t_minus_one = h_t	  #update h_t_minus_one to h_t_K
        return h_t_batch


def lista_apply(train_loader, valid_loader, K, W, h0, num_epochs, lr, weight_decay, device):
    n = W.shape[0]
    m = W.shape[1]

    lista = LISTA_Model(n=n, m=m, K=K, W=W, h0=h0).to(device)

    optimizer = torch.optim.Adam(
        lista.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Decrease LR if validation loss does not improve for 1 epoch
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # we want to minimize validation loss
        factor=0.1,  # multiply LR by this factor
        patience=0,  # wait 1 epoch before reducing
        verbose=True
    )
    os.makedirs("checkpoints/", exist_ok=True)
    if os.path.exists(CHECK_POINT_PATH):
        checkpoint = torch.load(CHECK_POINT_PATH, map_location=device)
        lista.load_state_dict(checkpoint['drnmf_net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        loss_train = checkpoint['loss_train']
        loss_valid = checkpoint['loss_valid']
    else:
        start_epoch = 0
        loss_train = np.zeros((num_epochs,))
        loss_valid = np.zeros((num_epochs,))


    loss_train, loss_valid = train(lista, train_loader, valid_loader, num_epochs, optimizer, scheduler, start_epoch, loss_train, loss_valid, device)
    train_err_lista = loss_train
    err_lista = loss_valid
    A = lista.A_list[K-1].weight.data.t()
    return lista, A, train_err_lista, err_lista


if __name__ == '__main__':

    # do_initial_nmf = True
    do_initial_nmf = False
    params_snmf = {'max_iter': 1000.,
                   'conv_eps': 1e-4,
                   'display': 0.,
                   'r': 200}

    stft = STFT(n_fft=1024, hop_length=256, win_length=1024, window="hanning")
    istft = ISTFT(n_fft=1024, hop_length=256, win_length=1024, window="hanning")


    wavs_clean_dir = "/Users/dor/dor/Datasets/VoiceBank_DEMAND_16k/clean_dataset_2sec/"
    wavs_noisy_dir = "/Users/dor/dor/Datasets/VoiceBank_DEMAND_16k/noisy_dataset_2sec/"

    if os.path.exists("train_valid_file_names.mat"):
        mat_data = scipy.io.loadmat('train_valid_file_names.mat')

        # Access the list of strings
        training_filelist = mat_data['training_filelist']
        validation_filelist = mat_data['validation_filelist']

        # Convert the numpy array to a Python list
        training_filelist = training_filelist.tolist()
        validation_filelist = validation_filelist.tolist()

        training_filelist = [file.replace(" ", "") for file in training_filelist]
        validation_filelist = [file.replace(" ", "") for file in validation_filelist]
        len_dataset = len(training_filelist) + len(validation_filelist)
    else:
        data_filenameslist = [wavfile for wavfile in os.listdir(wavs_clean_dir) if
                              wavfile.endswith('.wav')]  # dor
        random.shuffle(data_filenameslist)
        # data_filenameslist = data_filenameslist[:10000]
        # data_filenameslist = data_filenameslist[:200]
        # data_filenameslist = data_filenameslist[:2000]
        data_filenameslist = data_filenameslist[:5000]
        len_dataset = len(data_filenameslist)
        training_filelist, validation_filelist = data_filenameslist[
                                                 :int(0.8 * len_dataset)], data_filenameslist[
                                                                           int(0.8 * len_dataset):len_dataset]
        data = {'training_filelist': training_filelist, 'validation_filelist': validation_filelist}
        scipy.io.savemat('train_valid_file_names.mat', data)

    trainset = Dataset(len_dataset, training_filelist, wavs_clean_dir, wavs_noisy_dir, AUDIO_LENGTH, training_filelist)
    train_loader = DataLoader(trainset, num_workers=1, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True)

    validset = Dataset(len_dataset, validation_filelist, wavs_clean_dir, wavs_noisy_dir, AUDIO_LENGTH, validation_filelist)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False, batch_size=BATCH_SIZE, pin_memory=True)


    N_train_abs = trainset.N_abs.cpu().numpy().astype('float32')
    N_valid_abs = validset.N_abs.cpu().numpy().astype('float32')
    Y_train_abs = trainset.Y_abs.cpu().numpy().astype('float32')
    Y_valid_abs = validset.Y_abs.cpu().numpy().astype('float32')


    if do_initial_nmf:
        W, H = train_snmf(Y_train_abs, N_train_abs, params_snmf)
        sio.savemat(open("W.mat", "wb"), {"W": W})
        sio.savemat(open("H.mat", "wb"), {"H": H})
        sio.savemat(open("h0.mat", "wb"), {"h0": H[:, 0]})

    if do_initial_nmf:
        # Valid
        params_snmf_infer = copy.deepcopy(params_snmf)
        params_snmf_infer['r'] = 2 * params_snmf['r']
        idx_update = np.zeros(2 * int(params_snmf['r']), dtype=bool)
        params_snmf_infer.update({'init_w': W, 'w_update_ind': idx_update, 'conv_eps': 0.,
                                  'max_iter': 1000.
                                  # 'max_iter' : 25.
                                  })
        # W is fixed, apply snmf on the validtion set
        _, H_valid = sparse_nmf(N_valid_abs, params_snmf_infer)

        sio.savemat(open("H_valid.mat", "wb"), {"H_valid": H_valid})

        r = int(params_snmf['r'])  # number of basis vectors

        # extract subdictionaries for clean speech and noise
        W_y = W[:, :r]
        W_n = W[:, r:]

        # extract subactivation matrices for clean speech and noise
        H_y_valid = H_valid[:r, :]
        H_n_valid = H_valid[r:, :]

        # compute the ideal ratio mask
        clean_est = np.matmul(W_y, H_y_valid)
        noise_est = np.matmul(W_n, H_n_valid)
        irm_snmf = clean_est / (1e-9 + clean_est + noise_est)

        sio.savemat(open("irm_snmf.mat", "wb"), {"irm_snmf": irm_snmf})

        irm_snmf = torch.FloatTensor(irm_snmf).reshape(irm_snmf.shape[0], len(validset), irm_snmf.shape[1] // len(validset)).permute(1, 0, 2)
        validset_N = validset.N.reshape(validset.N.shape[0], len(validset), validset.N.shape[1] // len(validset)).permute(1, 0, 2)
        validset_Y = validset.Y.reshape(validset.Y.shape[0], len(validset), validset.Y.shape[1] // len(validset)).permute(1, 0, 2)
        # compute and save of validation loss
        est_istft = istft(irm_snmf * validset_N, AUDIO_LENGTH)
        clean_istft = istft(validset_Y, AUDIO_LENGTH)
        val_loss = -si_sdr(preds=est_istft, target=clean_istft).mean().item()

        print("Signal approximation SI-SDR loss for SNMF on validation set is " + str(format(val_loss, ".8f")))


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # dor
    print(device)


    L = sio.loadmat(open("W.mat", "rb"))
    W = np.asarray(L['W'], dtype='float32')
    print(W.shape)
    W = torch.from_numpy(W.astype(np.float32))

    L = sio.loadmat(open("h0.mat", "rb"))
    h0 = np.asarray(L['h0'], dtype='float32')
    h0 = torch.from_numpy(h0.astype(np.float32))
    h0 = h0.t()
    print(h0.shape)

    num_epochs = 30
    lr = 1e-5
    weight_decay = 0

    # Number of unfoldings
    K = 4

    lista, Wk, train_err_lista, err_lista = lista_apply(train_loader, validation_loader, K, W, h0, num_epochs, lr,
                                                        weight_decay, device)

    torch.save(lista.state_dict(), CHECK_POINT_PATH)

    epochs = range(0, num_epochs, 1)

    # plot the resutls
    fig = plt.figure()
    # plt.plot(epochs, train_err_lista, label='train loss', color='b',linewidth=0.5)
    plt.plot(epochs, err_lista, label='validation loss', color='r', linewidth=2)
    plt.xlabel('Number of epochs', fontsize=10)
    plt.ylabel('SI-SDR loss', fontsize=10)
    # plt.yscale("log")
    plt.legend()
    plt.savefig("training.png", dpi=300, bbox_inches="tight")
    plt.show()
