import math
import torch
from .r2rfft import FFTCore

class CustomSTFT(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length=None,
        win_length=None,
        window=None,
        pad_mode="reflect",
        normalized=False,
        center=True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window if window is not None else torch.hann_window(n_fft)
        self.pad_mode = pad_mode
        self.normalized = normalized
        self.center = center

        # Initialize FFTCore
        self.fft_core = FFTCore(n_fft)

    def forward(self, x):
        # Step 1: Centering and padding
        if self.center:
            x = self._center_signal(x)

        # Step 2: Segment the signal into overlapping windows
        frames = self._frame_signal(x)

        # Step 3: Apply window function to each frame
        frames = frames * self.window.unsqueeze(1)

        # Step 4: Perform FFT on each frame
        parts = []
        for frame_idx in range(frames.shape[-1]):
            frame = frames[..., frame_idx]
            fft_result = self.fft_core(frame)
            parts.append(fft_result)

        out = torch.stack(parts, dim=-1)
        out = out[:, : self.n_fft // 2 + 1, :]

        # Normalize
        if self.normalized:
            out = out / math.sqrt(self.n_fft)

        return out

    def _center_signal(self, x):
        # Apply padding and centering if needed
        if self.pad_mode == "reflect":
            x = torch.nn.functional.pad(
                x, (self.n_fft // 2, self.n_fft // 2), mode="reflect"
            )
        return x

    def _frame_signal(self, x, n_fft=None, hop_length=None):
        n_fft = n_fft if n_fft is not None else self.n_fft
        hop_length = hop_length if hop_length is not None else self.hop_length

        frames = []
        for i in range(0, x.size(-1) - n_fft + 1, hop_length):
            frame = x[:, i : i + n_fft]
            frames.append(frame)
        return torch.stack(frames, dim=2)


class CustomISTFT(torch.nn.Module):
    def __init__(
        self,
        n_fft,
        hop_length=None,
        win_length=None,
        window=None,
        normalized=False,
        center=True,
        length=None,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.win_length = win_length if win_length is not None else n_fft
        self.normalized = normalized
        self.window = (window if window is not None else torch.hann_window(n_fft))
        self.center = center
        self.length = length
        self.fft_core = FFTCore(n_fft, inverse=True)  # Use inverse FFT

        self.edge_length = self.n_fft // 2

    def forward(self, z):
        # Unnormalize
        if self.normalized:
            z = z * math.sqrt(self.n_fft)
        frames = []
        for i in range(z.size(-1)):
            frame = z[..., i]
            frame = self._build_full_spectrum(frame)
            ifft_result = self.fft_core(frame)
            frames.append(ifft_result)

        frames = torch.stack(frames, dim=-1)

        # Apply overlap-add with window
        signal = self._overlap_add(frames)

        if self.center:
            # Trim the padding added during STFT
            signal = signal[..., self.edge_length : -self.edge_length]

        if self.length is not None:
            # Match the desired output length
            signal = signal[..., : self.length]

        # # Step 4: Apply window normalization if needed
        if self.normalized:
            signal = signal * (self.n_fft / (self.window.sum()*1.5))

        return signal

    def _build_full_spectrum(self, half_spectrum):
        real_half = half_spectrum[0, ...]
        imag_half = half_spectrum[1, ...]

        # Construct negative frequency components symmetrically
        real_full = torch.cat([real_half, real_half.flip(0)[1:-1]], dim=0)
        imag_full = torch.cat([imag_half, -imag_half.flip(0)[1:-1]], dim=0)

        full_spectrum = torch.stack((real_full, imag_full), dim=0)

        return full_spectrum

    def _overlap_add(self, frames, hop_length=None, win_length=None, window=None, edge_length=None):
        hop_length = hop_length if hop_length is not None else self.hop_length
        win_length = win_length if win_length is not None else self.win_length
        window = window if window is not None else self.window
        edge_length = edge_length if edge_length is not None else self.edge_length

        output_length = (frames.size(-1) - 1) * hop_length + win_length

        signal = torch.zeros((2, output_length), device=frames.device)
        window_sum = torch.zeros(output_length, device=frames.device)

        for i in range(frames.shape[-1]):
            start_idx = i * hop_length
            end_idx = start_idx + win_length

            frame = frames[..., i]
            signal[:, start_idx:end_idx] += frame * window
            window_sum[start_idx:end_idx] += window

        # Normalize by the overlapping window sum
        non_zero = window_sum >= 1e-6
        signal[..., non_zero] /= window_sum[non_zero]

        return signal