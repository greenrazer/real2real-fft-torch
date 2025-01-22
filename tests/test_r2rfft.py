import torch
import math
import unittest
from src.r2rtorch.r2rfft import FFTCore


class TestFFT(unittest.TestCase):
    def test_fft(self):
        num_samples = int(math.pow(2, 9))

        torch.manual_seed(42)
        input_tensor = torch.rand(num_samples)
        input_tensor_2d = torch.stack(
            (input_tensor, torch.zeros_like(input_tensor)), dim=0
        )

        torch_fft = torch.fft.fft(input_tensor)

        fft_real = FFTCore(num_samples=num_samples)
        my_fft = fft_real(input_tensor_2d)

        tolerance = 1e-5
        self.assertTrue(
            torch.allclose(torch_fft.real, my_fft[0], atol=tolerance),
            "FFT Real Units Test Failed.",
        )
        self.assertTrue(
            torch.allclose(torch_fft.imag, my_fft[1], atol=tolerance),
            "FFT Imag Units Test Failed.",
        )

    def test_ifft(self):
        num_samples = int(math.pow(2, 9))

        torch.manual_seed(42)
        input_tensor = torch.rand(num_samples)

        freqs = torch.fft.fft(input_tensor)
        freqs_2d = torch.stack((freqs.real, freqs.imag), dim=0)

        torch_ifft = torch.fft.ifft(freqs)

        ifft_real = FFTCore(num_samples=num_samples, inverse=True)
        my_ifft = ifft_real(freqs_2d)

        tolerance = 1e-6
        self.assertTrue(
            torch.allclose(torch_ifft.real, my_ifft[0], atol=tolerance),
            "IFFT Real Units Test Failed",
        )
        self.assertTrue(
            torch.allclose(torch_ifft.imag, my_ifft[1], atol=tolerance),
            "IFFT Imag Units Test Failed",
        )

    def test_fft_reversible(self):
        num_samples = int(math.pow(2, 9))
        input_tensor = torch.rand(num_samples)

        fft_real = FFTCore(num_samples=num_samples)
        ifft_real = FFTCore(num_samples=num_samples, inverse=True)

        fft_result = fft_real(
            torch.stack((input_tensor, torch.zeros_like(input_tensor)), dim=-0)
        )
        ifft_result = ifft_real(fft_result)[0]

        tolerance = 1e-6
        self.assertTrue(
            torch.allclose(input_tensor, ifft_result, atol=tolerance),
            "Random input reproducibility test failed!",
        )
