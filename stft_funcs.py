import torch
import torch.nn as nn

class STFT(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, window="hanning", center=True):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center

        if window is not None:
            if isinstance(window, str):
                if window.lower() == "hanning":
                    self.window_fn = torch.hann_window(self.win_length)
                elif window.lower() == "hamming":
                    self.window_fn = torch.hamming_window(self.win_length)
                elif window.lower() == "blackman":
                    self.window_fn = torch.blackman_window(self.win_length)
                else:
                    raise ValueError("Invalid window type. Supported types are 'hanning', 'hamming', and 'blackman'.")
            elif isinstance(window, torch.Tensor):
                self.window_fn = window
            else:
                raise ValueError("Invalid window type. It should be either a string or a torch.Tensor.")
        else:
            self.window_fn = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.window_fn = self.window_fn.to(device=self.device)

    def forward(self, input_signal):
        self.device = input_signal.device
        self.window_fn = self.window_fn.clone().detach().to(device=self.device)

        if input_signal.ndim == 3:
            batch_size, channels, time = input_signal.size()

            complex_spectrum = torch.stft(input_signal.view(batch_size * channels, time), n_fft=self.n_fft,
                                          hop_length=self.hop_length,
                                          win_length=self.win_length, window=self.window_fn, center=self.center,
                                          return_complex=True)

            _, Fre, Len = complex_spectrum.size()  # B*M , F= num freqs, L= num frame, 2= real imag
            complex_spectrum = complex_spectrum.view([batch_size, channels, Fre, Len])  # B*M, F, L -> B, M, F, L

        elif input_signal.ndim == 2:
            batch_size, time = input_signal.size()

            complex_spectrum = torch.stft(input_signal.view(batch_size, time), n_fft=self.n_fft,
                                          hop_length=self.hop_length,
                                          win_length=self.win_length, window=self.window_fn, center=self.center,
                                          return_complex=True)
        else:
            complex_spectrum = None

        return complex_spectrum


class ISTFT(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, window="hanning", center=True):
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center

        if window is not None:
            if isinstance(window, str):
                if window.lower() == "hanning":
                    self.window_fn = torch.hann_window(self.win_length)
                elif window.lower() == "hamming":
                    self.window_fn = torch.hamming_window(self.win_length)
                elif window.lower() == "blackman":
                    self.window_fn = torch.blackman_window(self.win_length)
                else:
                    raise ValueError("Invalid window type. Supported types are 'hanning', 'hamming', and 'blackman'.")
            elif isinstance(window, torch.Tensor):
                self.window_fn = window
            else:
                raise ValueError("Invalid window type. It should be either a string or a torch.Tensor.")
        else:
            self.window_fn = None

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.window_fn = self.window_fn.to(device=self.device)

    def forward(self, complex_spectrum, time):
        self.device = complex_spectrum.device

        if complex_spectrum.ndim == 4:
            batch_size, channels, Fre, Len = complex_spectrum.size()
            complex_spectrum = complex_spectrum.reshape([-1, Fre, Len])

            self.window_fn = self.window_fn.clone().detach().to(device=self.device)
            reconstructed_signal = torch.istft(complex_spectrum, n_fft=self.n_fft, hop_length=self.hop_length,
                                               win_length=self.win_length, window=self.window_fn, length=time,
                                               center=self.center)
            reconstructed_signal = reconstructed_signal.view([batch_size, channels, time])
        elif complex_spectrum.ndim == 3:
            self.window_fn = self.window_fn.clone().detach().to(device=self.device)
            reconstructed_signal = torch.istft(complex_spectrum, n_fft=self.n_fft, hop_length=self.hop_length,
                                               win_length=self.win_length, window=self.window_fn, length=time,
                                               center=self.center)
        else:
            reconstructed_signal = None

        return reconstructed_signal