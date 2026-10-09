import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

IMAGE_SIZE = 256
IMAGE_CHANNELS = 1
INITIAL_SIZE = 4
INITIAL_CHANNELS = 512

"""Baseline"""


def _generator_block(in_channels, out_channels):
    #  Double spatial resolution, then refine the upsampled feature map.
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _discriminator_block(in_channels, out_channels):
    # Learn features while halving the spatial resolution.
    convolution = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
    )
    # Initialize before spectral_norm creates its weight_orig/u/v state.
    _initialize_weights(convolution)
    return nn.Sequential(
        spectral_norm(convolution),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


def _dcgan_generator_block(
    in_channels, out_channels, kernel_size=4, stride=2, padding=1
):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _dcgan_discriminator_block(in_channels, out_channels, use_batchnorm=True):
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
    ]

    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))

    layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

    return nn.Sequential(*layers)


def _initialize_weights(module):
    # Shared initialization for ConvGAN and DCGAN.
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        weight = getattr(module, "weight_orig", module.weight)
        nn.init.normal_(weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        if module.weight is not None:
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ConvGANGenerator(nn.Module):
    # Generate one-channel ADNI MRI slices from latent noise.
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # Learn a small spatial seed containing the global brain layout.
        self.project = nn.Linear(
            latent_dim,
            INITIAL_CHANNELS * INITIAL_SIZE * INITIAL_SIZE,
        )
        self.seed = nn.Sequential(
            nn.Unflatten(1, (INITIAL_CHANNELS, INITIAL_SIZE, INITIAL_SIZE)),
            nn.BatchNorm2d(INITIAL_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # 4 -> 8 -> 16 -> 32 -> 64 -> 128 pixels.
        self.features = nn.Sequential(
            _generator_block(512, 512),
            _generator_block(512, 256),
            _generator_block(256, 128),
            _generator_block(128, 64),
            _generator_block(64, 32),
        )

        # Produce a single grayscale MRI channel at 256x256.
        self.to_image = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, IMAGE_CHANNELS, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.apply(_initialize_weights)

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent vectors with shape [batch, {self.latent_dim}], "
                f"but received {tuple(z.shape)}"
            )

        features = self.project(z)
        features = self.seed(features)
        features = self.features(features)
        return self.to_image(features)


class MinibatchStdDev(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, features):
        batch_size, _, height, width = features.shape
        if batch_size == 1:
            std_feature = features.new_zeros(batch_size, 1, height, width)
        else:
            feature_std = torch.sqrt(features.var(dim=0, unbiased=False) + self.eps)
            mean_std = feature_std.mean()
            std_feature = mean_std.reshape(1, 1, 1, 1).expand(
                batch_size, 1, height, width
            )

        # [B, C, H, W] → [B, C + 1, H, W]
        return torch.cat([features, std_feature], dim=1)


class ConvGANDiscriminator(nn.Module):
    #  Score one-channel 256x256 MRI slices using spectrally-normalized layers.

    def __init__(self):
        super().__init__()

        # 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 pixels.
        self.features = nn.Sequential(
            _discriminator_block(IMAGE_CHANNELS, 32),
            _discriminator_block(32, 64),
            _discriminator_block(64, 128),
            _discriminator_block(128, 256),
            _discriminator_block(256, 512),
            _discriminator_block(512, 512),
        )

        self.minibatch_stddev = MinibatchStdDev()
        # A 4x4 convolution sees the complete final feature map and emits one logit.
        # There is deliberately no Sigmoid: use BCEWithLogitsLoss or a hinge loss.
        classifier = nn.Conv2d(512 + 1, 1, kernel_size=4, stride=1, padding=0)
        _initialize_weights(classifier)
        self.classifier = nn.Sequential(
            spectral_norm(classifier),
            nn.Flatten(),
        )

    def forward(self, image):
        expected_shape = (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected images with shape [batch, {IMAGE_CHANNELS}, "
                f"{IMAGE_SIZE}, {IMAGE_SIZE}], but received {tuple(image.shape)}"
            )

        features = self.features(image)
        # [B, 512 + 1, 4, 4]
        # Add standard deviation as a new channel
        features = self.minibatch_stddev(features)
        return self.classifier(features)


class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        # [B, latent_dim, 1, 1] -> [B, 512, 4, 4]
        self.seed = _dcgan_generator_block(
            latent_dim,
            INITIAL_CHANNELS,
            kernel_size=INITIAL_SIZE,
            stride=1,
            padding=0,
        )

        # 4 -> 8 -> 16 -> 32 -> 64 -> 128
        self.features = nn.Sequential(
            _dcgan_generator_block(INITIAL_CHANNELS, 512),
            _dcgan_generator_block(512, 256),
            _dcgan_generator_block(256, 128),
            _dcgan_generator_block(128, 64),
            _dcgan_generator_block(64, 32),
        )

        # [B, 32, 128, 128] -> [B, 1, 256, 256]
        self.to_image = nn.Sequential(
            nn.ConvTranspose2d(
                32,
                IMAGE_CHANNELS,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

        self.apply(_initialize_weights)

    def forward(self, z):
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent vectors with shape "
                f"[batch, {self.latent_dim}], "
                f"but received {tuple(z.shape)}"
            )

        z = z.reshape(z.shape[0], self.latent_dim, 1, 1)

        features = self.seed(z)
        features = self.features(features)
        return self.to_image(features)


class DCGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.features = nn.Sequential(
            _dcgan_discriminator_block(IMAGE_CHANNELS, 32, use_batchnorm=False),
            _dcgan_discriminator_block(32, 64),
            _dcgan_discriminator_block(64, 128),
            _dcgan_discriminator_block(128, 256),
            _dcgan_discriminator_block(256, 512),
            _dcgan_discriminator_block(512, 512),
        )

        # [B, 512, 4, 4] -> [B, 1, 1, 1] -> [B, 1]
        self.classifier = nn.Sequential(
            nn.Conv2d(
                512, 1, kernel_size=INITIAL_SIZE, stride=1, padding=0, bias=False
            ),
            nn.Flatten(start_dim=1),
        )

        self.apply(_initialize_weights)

    def forward(self, image):
        expected_shape = (IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

        if image.ndim != 4 or tuple(image.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected images with shape "
                f"[batch, {IMAGE_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE}], "
                f"but received {tuple(image.shape)}"
            )

        features = self.features(image)
        return self.classifier(features)


# StyleGAN2
def normalize_second_moment(x, dim=-1, eps=1e-8):
    # Normalize each vector by its root mean square.
    rms = torch.sqrt(x.square().mean(dim=dim, keepdim=True) + eps)
    return x / rms


# Linear layer using equalized learning-rate parameterization.
# The stored weight is scaled by lr_multiplier / sqrt(fan_in) during each forward pass.
class EqualizedLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        activation="linear",
        lr_multiplier=1.0,
        bias_init=0.0,
    ):
        super().__init__()
        if in_features < 1 or out_features < 1 or lr_multiplier <= 0:
            raise ValueError("Dimensions and lr_multiplier must be positive.")

        if activation not in ("linear", "lrelu"):
            raise ValueError("activation must be 'linear' or 'lrelu'.")

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.lr_multiplier = float(lr_multiplier)

        self.weight_gain = self.lr_multiplier / math.sqrt(in_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        self.bias = nn.Parameter(torch.full((out_features,), float(bias_init)))

        nn.init.normal_(self.weight, mean=0.0, std=1.0 / self.lr_multiplier)

    def forward(self, x):
        effective_weight = self.weight * self.weight_gain
        effective_bias = self.bias * self.lr_multiplier

        y = F.linear(x, effective_weight, effective_bias)

        if self.activation == "lrelu":
            y = F.leaky_relu(y, negative_slope=0.2)
            # Leaky ReLU gain
            y = y * math.sqrt(2.0 / (1 + 0.2**2))

        return y


class MappingNetwork(nn.Module):
    def __init__(
        self,
        num_ws,
        z_dim=512,
        w_dim=512,
        num_layers=8,
        lr_multiplier=0.01,
        w_avg_beta=0.995,
    ):
        super().__init__()

        if min(num_ws, z_dim, w_dim, num_layers) < 1:
            raise ValueError("Dimensions and layer counts must be positive.")

        if not 0 <= w_avg_beta < 1:
            raise ValueError("w_avg_beta must be in [0, 1).")

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.w_avg_beta = w_avg_beta

        self.layers = nn.ModuleList(
            [
                EqualizedLinear(
                    in_features=z_dim if i == 0 else w_dim,
                    out_features=w_dim,
                    activation="lrelu",
                    lr_multiplier=lr_multiplier,
                )
                for i in range(num_layers)
            ]
        )

        self.register_buffer("w_avg", torch.zeros(w_dim))

    def forward(self, z, *, update_w_avg=True):
        if z.ndim != 2 or z.shape[1] != self.z_dim or z.shape[0] == 0:
            raise ValueError(
                f"Expected nonempty [B, {self.z_dim}], " f"got {tuple(z.shape)}."
            )

        w = normalize_second_moment(z.to(dtype=torch.float32))

        for layer in self.layers:
            w = layer(w)

        if self.training and update_w_avg:
            with torch.no_grad():
                batch_mean = w.detach().mean(dim=0)
                self.w_avg.mul_(self.w_avg_beta)
                self.w_avg.add_(batch_mean, alpha=1.0 - self.w_avg_beta)

        ws = w.unsqueeze(1).repeat(1, self.num_ws, 1)

        return ws


class EqualizedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        *,
        bias=True,
        activation="linear",
    ):
        super().__init__()
        if min(in_channels, out_channels, kernel_size) < 1:
            raise ValueError("Channels and kernel_size must be positive.")
        if kernel_size % 2 == 0:
            raise ValueError("Use an odd kernel size.")
        if activation not in ("linear", "lrelu"):
            raise ValueError("Activation must be 'linear' or 'lrelu'.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = kernel_size // 2
        self.activation = activation

        fan_in = in_channels * (kernel_size**2)
        self.weight_gain = 1.0 / math.sqrt(fan_in)

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B, {self.in_channels}, H, W], " f"got {tuple(x.shape)}."
            )

        effective_weight = self.weight * self.weight_gain

        y = F.conv2d(x, effective_weight, self.bias, stride=1, padding=self.padding)

        if self.activation == "lrelu":
            negative_slope = 0.2
            y = F.leaky_relu(y, negative_slope=negative_slope)
            # Leaky ReLU gain
            y = y * math.sqrt(2.0 / (1.0 + negative_slope**2))

        return y


def make_resample_filter():
    taps = torch.tensor([1.0, 3.0, 3.0, 1.0], dtype=torch.float32)

    kernel = torch.outer(taps, taps)
    # Normalization
    kernel = kernel / kernel.sum()

    return kernel.reshape(1, 1, 4, 4)


class FilteredUpsample2d(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("filter", make_resample_filter())

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected [B, C, H, W].")

        channels = x.shape[1]

        kernel = self.filter.to(dtype=x.dtype) * 4.0

        kernel = kernel.repeat(channels, 1, 1, 1)

        return F.conv_transpose2d(x, kernel, stride=2, padding=1, groups=channels)


class FilteredDownsample2d(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("filter", make_resample_filter())

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected [B, C, H, W].")

        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("Height and width must be even.")

        channels = x.shape[1]

        kernel = self.filter.to(dtype=x.dtype)

        kernel = kernel.repeat(channels, 1, 1, 1)

        return F.conv2d(x, kernel, stride=2, padding=1, groups=channels)
