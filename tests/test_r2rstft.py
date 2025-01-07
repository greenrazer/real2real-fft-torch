import torch
import math
import unittest
from src.r2rstft import CustomSTFT, CustomISTFT

from tests.shared import visualize_signals

class TestSTFTOperations(unittest.TestCase):
    def setUp(self):
        # Common setup parameters used across tests
        self.n_fft = 512
        self.hop_length = self.n_fft // 4
        self.win_length = self.n_fft
        self.window = torch.hann_window(self.n_fft)
        
        # Common signal parameters
        self.signal_length = 2048
        self.sampling_rate = 8000
        self.t = torch.arange(self.signal_length) / self.sampling_rate
        self.signal = torch.sin(2 * torch.pi * 100 * self.t) + 0.1 * torch.randn_like(self.t)
        self.signal_2d = torch.stack((self.signal, torch.zeros_like(self.signal)), dim=0)

    def test_stft_computation(self):
        # Test STFT computation against PyTorch's implementation
        torch_stft = torch.stft(
            self.signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )

        custom_stft_mod = CustomSTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            pad_mode="reflect",
        )
        custom_stft = custom_stft_mod(self.signal_2d)

        torch_real = torch_stft.real
        torch_imag = torch_stft.imag
        custom_real = custom_stft[0, ...]
        custom_imag = custom_stft[1, ...]

        tolerance = 1e-6
        self.assertTrue(
            torch.allclose(torch_real, custom_real, atol=tolerance),
            "STFT Real Units Test Failed."
        )
        self.assertTrue(
            torch.allclose(torch_imag, custom_imag, atol=tolerance),
            "STFT Imaginary Units Test Failed."
        )

    def test_stft_reproducibility(self):
        # Test STFT reproducibility with reconstruction
        torch_stft = torch.stft(
            self.signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )

        custom_stft_mod = CustomSTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            pad_mode="reflect",
        )
        custom_stft = custom_stft_mod(self.signal_2d)

        torch_reconstructed = torch.istft(
            torch_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            normalized=True,
            center=True,
            length=self.signal_length,
        ).real

        custom_reconstructed = torch.istft(
            custom_stft[0] + 1j*custom_stft[1],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            normalized=True,
            center=True,
            length=self.signal_length,
        ).real

        tolerance = 1e-6
        self.assertTrue(
            torch.allclose(torch_reconstructed, custom_reconstructed, atol=tolerance),
            "STFT Reproducibility Test Failed."
        )

    def test_build_full_spectrum(self):
        signal = torch.ones_like(self.t)

        torch_stft = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )

        torch_stft2 = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            return_complex=True,
            onesided=False,
            pad_mode="reflect",
        )

        spectrum_full = torch_stft2.real[:, 0]
        spectrum_built = CustomISTFT._build_full_spectrum(None, 
            torch.stack((torch_stft.real, torch_stft.imag), dim=0)[..., 0])[0]

        tolerance = 1e-6
        self.assertTrue(
            torch.allclose(spectrum_full, spectrum_built, atol=tolerance),
            "Build Full Spectrum Test Failed."
        )

    def test_overlap_add(self):
        n_fft = 4
        hop_length = 4
        win_length = 4
        window = torch.rand((win_length))
        signal_length = 8
        signal = torch.stack([torch.rand((signal_length)), torch.zeros((signal_length))])

        frames = CustomSTFT._frame_signal(None, signal, n_fft=n_fft, hop_length=hop_length)
        result = CustomISTFT._overlap_add(
            None, frames, hop_length=hop_length, win_length=win_length, window=window, edge_length=n_fft//2
        )

        tolerance = 1e-6
        self.assertTrue(
            torch.allclose(result, signal, atol=tolerance),
            "Overlap Add Test Failed."
        )

    def test_istft(self):
        signal_length = int(math.pow(2, 10))
        t = torch.arange(signal_length) / self.sampling_rate
        signal = torch.sin(2 * torch.pi * 100 * t) + 0.1 * torch.randn_like(t)

        torch_stft = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )

        torch_reconstructed = torch.istft(
            torch_stft,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            normalized=True,
            center=True,
            length=signal_length,
        ).real

        custom_istft = CustomISTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            normalized=True,
            center=True,
            length=signal_length,
        )
        stft_signal = torch.stack((torch_stft.real, torch_stft.imag), dim=0)
        custom_reconstructed = custom_istft(stft_signal)[0, ...]

        visualize_signals(torch_reconstructed, custom_reconstructed)

        tolerance = 1e-6
        self.assertTrue(
            torch.allclose(torch_reconstructed, custom_reconstructed, atol=tolerance),
            "ISTFT Test Failed."
        )

    def test_istft_reproducibility(self):
        custom_stft_mod = CustomSTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            normalized=True,
            center=True,
            pad_mode="reflect",
        )
        custom_stft = custom_stft_mod(self.signal_2d)

        custom_istft = CustomISTFT(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            win_length=self.win_length,
            normalized=True,
            center=True,
            length=self.signal_length,
        )
        custom_reconstructed = custom_istft(custom_stft)[0, ...]

        tolerance = 1e-6
        self.assertTrue(
            torch.allclose(self.signal, custom_reconstructed, atol=tolerance),
            "ISTFT Reproducibility Test Failed."
        )
