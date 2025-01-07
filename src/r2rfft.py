import torch
import math

class FFTCore(torch.nn.Module):
    def __init__(self, num_samples, inverse=False):
        super().__init__()

        self.inverse = inverse
        self.one = torch.tensor(1.0)
        self.two = torch.tensor(2.0)
        self.angle_sign = self.one if self.inverse else -self.one

        self.num_samples = num_samples
        self.stages = int(math.log2(num_samples))
        self.bit_reversed_indicies = self.bit_reverse_permutation(num_samples)

    def forward(self, x):
        n = x.shape[1]

        real = x[0]
        imag = x[1]

        # Perform in-place bit-reversal permutation
        real = real[self.bit_reversed_indicies]
        imag = imag[self.bit_reversed_indicies]

        for stage in range(self.stages):
            group_size = 1 << (stage + 1)
            half_group = group_size >> 1

            # Reshape data into groups
            num_groups = n // group_size
            group_indices = torch.arange(num_groups, device=x.device) * group_size

            # Calculate twiddle factors for all groups
            j = torch.arange(half_group, device=x.device)  # Parallelize j loop
            angle = self.angle_sign * self.two * torch.pi * j / group_size
            # angle = self.angle_sign * self.two * math.pi * j / group_size
            wr = torch.cos(angle)
            wi = torch.sin(angle)

            # Expand the twiddle factors across all groups
            twiddle_real = wr.unsqueeze(0).expand(num_groups, half_group)
            twiddle_imag = wi.unsqueeze(0).expand(num_groups, half_group)

            # Indices for top and bottom elements in each group
            top_indices = group_indices[:, None] + j
            bottom_indices = top_indices + half_group

            # Perform butterfly operations for all groups in parallel
            tr_real = (
                twiddle_real * real[bottom_indices]
                - twiddle_imag * imag[bottom_indices]
            )
            tr_imag = (
                twiddle_imag * real[bottom_indices]
                + twiddle_real * imag[bottom_indices]
            )

            real[bottom_indices] = real[top_indices] - tr_real
            imag[bottom_indices] = imag[top_indices] - tr_imag

            real[top_indices] += tr_real
            imag[top_indices] += tr_imag

        out = torch.stack((real, imag), dim=0)

        if self.inverse:
            out = out / out.shape[1]

        return out

    def bit_reverse_permutation(self, n):
        indices = torch.arange(n)
        reversed_indices = torch.zeros_like(indices)

        num_bits = n.bit_length() - 1  # The number of bits for the indices
        for i in range(n):
            rev = 0
            temp = i
            for j in range(num_bits):
                rev = (rev << 1) | (temp & 1)
                temp >>= 1
            reversed_indices[i] = rev

        return reversed_indices


