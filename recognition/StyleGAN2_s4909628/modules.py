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
    def __init__(self, extra_padding=0):
        super().__init__()

        if not isinstance(extra_padding, int) or extra_padding < 0:
            raise ValueError("extra_padding must be a nonnegative integer.")

        self.extra_padding = extra_padding

        self.register_buffer("filter", make_resample_filter())

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected [B, C, H, W].")

        channels = x.shape[1]

        kernel = self.filter.to(dtype=x.dtype) * 4.0

        kernel = kernel.repeat(channels, 1, 1, 1)

        y = F.conv_transpose2d(x, kernel, stride=2, padding=0, groups=channels)

        if self.extra_padding == 0:
            return y[..., 1:-1, 1:-1]
        if self.extra_padding > 1:
            border = self.extra_padding - 1
            y = F.pad(y, (border, border, border, border))

        return y


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim=512,
        kernel_size=3,
        *,
        up=1,
        demodulate=True,
        eps=1e-8,
    ):
        super().__init__()
        if min(in_channels, out_channels, w_dim, kernel_size) < 1:
            raise ValueError("Dimensions must be positive.")

        if kernel_size % 2 == 0:
            raise ValueError("Use an odd kernel_size.")

        if up not in (1, 2):
            raise ValueError("up must be 1 or 2.")

        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_dim = w_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.up = up
        if up == 2:
            self.upsample = FilteredUpsample2d(extra_padding=self.padding)
        else:
            self.upsample = nn.Identity()

        self.demodulate = demodulate
        self.eps = eps

        self.affine = EqualizedLinear(
            w_dim, in_channels, activation="linear", lr_multiplier=1.0, bias_init=1.0
        )

        # Translate w into one modulation coefficient per input channel.
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )

        self.weight_gain = 1.0 / math.sqrt(in_channels * (kernel_size**2))

    def forward(self, x, w):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected [B, {self.in_channels}, H, W], " f"got {tuple(x.shape)}."
            )

        batch_size = x.shape[0]

        if batch_size == 0:
            raise ValueError("Batch must not be empty.")

        if w.ndim != 2 or tuple(w.shape) != (batch_size, self.w_dim):
            raise ValueError(
                f"Expected w shape [{batch_size}, {self.w_dim}], "
                f"got {tuple(w.shape)}."
            )

        # [B, in_channels]
        styles = self.affine(w)

        # [B, out_channels, in_channels, k, k]
        sample_weight = self.weight.unsqueeze(0) * styles[:, None, :, None, None]

        if self.demodulate:
            # [B, out_channels, 1, 1, 1]
            energy = sample_weight.square().sum(dim=(2, 3, 4), keepdim=True)

            sample_weight = sample_weight * torch.rsqrt(energy + self.eps)
        else:
            sample_weight = sample_weight * self.weight_gain

        if self.up == 2:
            x = self.upsample(x)
            conv_padding = 0
        else:
            conv_padding = self.padding

        grouped_input = x.reshape(
            1, batch_size * self.in_channels, x.shape[-2], x.shape[-1]
        )

        grouped_weight = sample_weight.reshape(
            batch_size * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        y = F.conv2d(
            grouped_input, grouped_weight, padding=conv_padding, groups=batch_size
        )

        return y.reshape(batch_size, self.out_channels, y.shape[-2], y.shape[-1])


# Style convolution, optional noise, bias, and activation.
class StyleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        resolution,
        w_dim=512,
        kernel_size=3,
        *,
        up=1,
        use_noise=True,
    ):
        super().__init__()

        if resolution < 1:
            raise ValueError("resolution must be positive.")
        if up not in (1, 2):
            raise ValueError("up must be 1 or 2.")

        if resolution % up != 0:
            raise ValueError("resolution must be divisible by up.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.up = up
        self.input_resolution = resolution // up
        self.use_noise = use_noise

        self.negative_slope = 0.2
        self.activation_gain = math.sqrt(2.0 / (1.0 + self.negative_slope**2))

        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            w_dim=w_dim,
            kernel_size=kernel_size,
            demodulate=True,
            up=up,
        )

        self.bias = nn.Parameter(torch.zeros(out_channels))

        if use_noise:
            self.noise_strength = nn.Parameter(torch.zeros(()))

            self.register_buffer(
                "noise_const", torch.randn(1, 1, resolution, resolution)
            )
        else:
            self.register_parameter("noise_strength", None)
            self.register_buffer("noise_const", None)

    def forward(self, x, w, *, noise_mode="random"):
        if noise_mode not in ("random", "const", "none"):
            raise ValueError("noise_mode must be 'random', 'const', or 'none'.")

        expected = (self.in_channels, self.input_resolution, self.input_resolution)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"Expected [B, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(x.shape)}."
            )

        y = self.conv(x, w)

        # Optional noise
        if self.use_noise and noise_mode != "none":
            if noise_mode == "random":
                noise = torch.randn(
                    y.shape[0],
                    1,
                    self.resolution,
                    self.resolution,
                    device=y.device,
                    dtype=y.dtype,
                )
            else:
                noise = self.noise_const.to(dtype=y.dtype)

            y = y + self.noise_strength * noise

        y = y + self.bias.reshape(1, -1, 1, 1)
        y = F.leaky_relu(y, negative_slope=self.negative_slope)
        return y * self.activation_gain


# Project features into image channels without demodulation.
class ToImage(nn.Module):
    def __init__(self, in_channels, w_dim=512, image_channels=1):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=image_channels,
            w_dim=w_dim,
            kernel_size=1,
            demodulate=False,
        )
        self.bias = nn.Parameter(torch.zeros(image_channels))

    def forward(self, x, w):
        image = self.conv(x, w)
        image = image + self.bias.reshape(1, -1, 1, 1)
        return image


class SynthesisBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, resolution, w_dim=512, image_channels=1
    ):
        super().__init__()
        if in_channels < 0 or min(out_channels, w_dim, image_channels) < 1:
            raise ValueError("Invalid channel count or w_dim.")

        if resolution < 4 or resolution & (resolution - 1):
            raise ValueError("resolution must be a power of two " "and at least 4.")

        self.is_first = resolution == 4
        if (in_channels == 0) != self.is_first:
            raise ValueError("in_channels must be 0 only for the 4x4 block.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.w_dim = w_dim
        self.image_channels = image_channels

        self.num_conv = 1 if self.is_first else 2
        self.num_toimage = 1

        self.num_ws = self.num_conv + self.num_toimage

        if self.is_first:
            self.const = nn.Parameter(torch.randn(1, out_channels, 4, 4))
            self.conv_up = None
            self.image_upsample = None
        else:
            self.register_parameter("const", None)

            self.conv_up = StyleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                resolution=resolution,
                w_dim=w_dim,
                up=2,
            )
            self.image_upsample = FilteredUpsample2d(extra_padding=0)

        self.conv = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            resolution=resolution,
            w_dim=w_dim,
            up=1,
        )

        self.to_image = ToImage(
            in_channels=out_channels, w_dim=w_dim, image_channels=image_channels
        )

    def forward(self, x, image, ws, *, noise_mode="random"):
        if (
            ws.ndim != 3
            or tuple(ws.shape[1:]) != (self.num_ws, self.w_dim)
            or ws.shape[0] == 0
        ):
            raise ValueError(
                f"Expected nonempty ws [B, {self.num_ws}, {self.w_dim}], "
                f"got {tuple(ws.shape)}."
            )

        batch_size = ws.shape[0]

        if self.is_first:
            if x is not None or image is not None:
                raise ValueError("The first block expects x=None and image=None.")

            x = self.const.expand(batch_size, -1, -1, -1)
        else:
            if x is None or image is None:
                raise ValueError("A later block requires both x and image.")

            previous_resolution = self.resolution // 2

            expected_x = (
                batch_size,
                self.in_channels,
                previous_resolution,
                previous_resolution,
            )
            expected_image = (
                batch_size,
                self.image_channels,
                previous_resolution,
                previous_resolution,
            )
            if tuple(x.shape) != expected_x:
                raise ValueError(f"Expected x {expected_x}, got {tuple(x.shape)}.")

            if tuple(image.shape) != expected_image:
                raise ValueError(
                    f"Expected image {expected_image}, got {tuple(image.shape)}."
                )

            x = self.conv_up(x, ws[:, 0, :], noise_mode=noise_mode)

        x = self.conv(x, ws[:, self.num_conv - 1, :], noise_mode=noise_mode)
        image_delta = self.to_image(x, ws[:, self.num_conv, :])

        if self.is_first:
            image = image_delta
        else:
            image = self.image_upsample(image) + image_delta

        return x, image


class SynthesisNetwork(nn.Module):
    def __init__(
        self,
        w_dim=512,
        image_resolution=256,
        image_channels=1,
        channel_base=16384,
        channel_max=512,
    ):
        super().__init__()
        if image_resolution < 4 or image_resolution & (image_resolution - 1):
            raise ValueError(
                "image_resolution must be a power of two " "and at least 4."
            )

        if min(w_dim, image_channels, channel_base, channel_max) < 1:
            raise ValueError("Dimensions and channel settings must be positive.")

        if channel_base < image_resolution:
            raise ValueError("channel_base must be at least image_resolution.")

        self.w_dim = w_dim
        self.image_resolution = image_resolution
        self.image_channels = image_channels
        self.channel_base = channel_base
        self.channel_max = channel_max

        # all synthesis resolutions
        self.block_resolutions = []
        resolution = 4
        while resolution <= image_resolution:
            self.block_resolutions.append(resolution)
            resolution *= 2
        # output channels at each resolution
        self.channels = {
            resolution: min(channel_base // resolution, channel_max)
            for resolution in self.block_resolutions
        }
        # blocks in increasing resolution order.
        self.blocks = nn.ModuleList()
        previous_channels = 0
        for resolution in self.block_resolutions:
            out_channels = self.channels[resolution]
            block = SynthesisBlock(
                in_channels=previous_channels,
                out_channels=out_channels,
                resolution=resolution,
                w_dim=w_dim,
                image_channels=image_channels,
            )

            self.blocks.append(block)
            previous_channels = out_channels

        # Adjacent blocks share one style slot.
        # --> previous ToIamge shares one style slot with current Conv Up
        self.num_ws = (
            sum(block.num_conv for block in self.blocks) + self.blocks[-1].num_toimage
        )

    def forward(self, ws, *, noise_mode="random"):
        if (
            ws.ndim != 3
            or tuple(ws.shape[1:]) != (self.num_ws, self.w_dim)
            or ws.shape[0] == 0
        ):
            raise ValueError(
                f"Expected nonempty ws [B, {self.num_ws}, {self.w_dim}], "
                f"got {tuple(ws.shape)}."
            )
        x = None
        image = None
        style_index = 0

        for block in self.blocks:
            block_ws = ws[:, style_index : style_index + block.num_ws, :]
            x, image = block(x, image, block_ws, noise_mode=noise_mode)

            style_index += block.num_conv

        return image


class StyleGAN2Generator(nn.Module):
    def __init__(
        self,
        z_dim=512,
        w_dim=512,
        image_resolution=256,
        image_channels=1,
        *,
        channel_base=16384,
        channel_max=512,
        mapping_layers=8,
        mapping_lr_multiplier=0.01,
        w_avg_beta=0.995,
    ):
        super().__init__()

        if z_dim < 1:
            raise ValueError("z_dim must be positive.")

        self.z_dim = z_dim
        self.w_dim = w_dim
        self.image_resolution = image_resolution
        self.image_channels = image_channels

        # determines how many style slots are needed
        self.synthesis = SynthesisNetwork(
            w_dim=w_dim,
            image_resolution=image_resolution,
            image_channels=image_channels,
            channel_base=channel_base,
            channel_max=channel_max,
        )

        self.num_ws = self.synthesis.num_ws

        self.mapping = MappingNetwork(
            num_ws=self.num_ws,
            z_dim=z_dim,
            w_dim=w_dim,
            num_layers=mapping_layers,
            lr_multiplier=mapping_lr_multiplier,
            w_avg_beta=w_avg_beta,
        )

    def forward(
        self,
        z,
        *,
        noise_mode="random",
        update_w_avg=True,
    ):
        if noise_mode not in ("random", "const", "none"):
            raise ValueError("noise_mode must be 'random', 'const', or 'none'.")

        ws = self.mapping(z, update_w_avg=update_w_avg)

        image = self.synthesis(ws, noise_mode=noise_mode)

        return image


class EqualizedDownsampleConv2d(EqualizedConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        *,
        bias=True,
        activation="linear",
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, bias=bias, activation=activation
        )

        self.register_buffer("filter", make_resample_filter())

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"Expected [B, {self.in_channels}, H, W].")

        if x.shape[0] == 0 or min(x.shape[-2:]) < 2:
            raise ValueError("Expected a nonempty batch and spatial size >= 2.")

        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("Height and width must be even.")

        kernel = self.filter.to(dtype=x.dtype).repeat(self.in_channels, 1, 1, 1)

        x = F.conv2d(x, kernel, padding=self.padding + 1, groups=self.in_channels)

        y = F.conv2d(x, self.weight * self.weight_gain, self.bias, stride=2, padding=0)

        if self.activation == "lrelu":
            y = F.leaky_relu(y, negative_slope=0.2)
            y = y * math.sqrt(2.0 / (1.0 + 0.2**2))

        return y


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, resolution):
        super().__init__()

        if min(in_channels, out_channels) < 1:
            raise ValueError("Channels must be positive.")

        if resolution < 8 or resolution & (resolution - 1):
            raise ValueError("resolution must be a power of two >= 8.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution

        self.conv = EqualizedConv2d(
            in_channels, in_channels, kernel_size=3, activation="lrelu"
        )

        self.conv_down = EqualizedDownsampleConv2d(
            in_channels, out_channels, kernel_size=3, activation="lrelu"
        )

        self.skip_down = EqualizedDownsampleConv2d(
            in_channels, out_channels, kernel_size=1, bias=False, activation="linear"
        )

    def forward(self, x):
        expected = (self.in_channels, self.resolution, self.resolution)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"Expected [B, {expected[0]}, {expected[1]}, {expected[2]}]."
            )

        main = self.conv_down(self.conv(x))
        skip = self.skip_down(x)

        return (main + skip) * math.sqrt(0.5)


class MinibatchStdDev(nn.Module):
    def __init__(self, group_size=4, eps=1e-8):
        super().__init__()

        if not isinstance(group_size, int) or group_size < 1:
            raise ValueError("group_size must be a positive integer.")

        if eps <= 0:
            raise ValueError("eps must be positive.")

        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        if x.ndim != 4 or min(x.shape) < 1:
            raise ValueError("Expected nonempty [B, C, H, W].")

        batch_size, channels, height, width = x.shape

        group_size = min(self.group_size, batch_size)

        while batch_size % group_size != 0:
            group_size -= 1

        grouped = x.reshape(
            batch_size // group_size, group_size, channels, height, width
        )

        centered = grouped - grouped.mean(dim=1, keepdim=True)
        std = torch.sqrt(centered.square().mean(dim=1) + self.eps)

        statistic = std.mean(dim=(1, 2, 3), keepdim=True)

        statistic = statistic.repeat_interleave(group_size, dim=0)

        statistic = statistic.expand(batch_size, 1, height, width)

        return torch.cat([x, statistic], dim=1)


class StyleGAN2Discriminator(nn.Module):
    def __init__(
        self,
        image_resolution=256,
        image_channels=1,
        *,
        channel_base=16384,
        channel_max=512,
        mbstd_group_size=4,
    ):
        super().__init__()

        if image_resolution < 4 or image_resolution & (image_resolution - 1):
            raise ValueError("image_resolution must be a power of two >= 4.")

        if min(image_channels, channel_base, channel_max) < 1:
            raise ValueError("Channels and channel settings must be positive.")

        if channel_base < image_resolution:
            raise ValueError("channel_base must be at least image_resolution.")

        self.image_resolution = image_resolution
        self.image_channels = image_channels
        self.channel_base = channel_base
        self.channel_max = channel_max

        self.block_resolutions = []
        resolution = image_resolution

        while resolution > 4:
            self.block_resolutions.append(resolution)
            resolution //= 2

        self.channels = {
            resolution: min(channel_base // resolution, channel_max)
            for resolution in self.block_resolutions + [4]
        }

        # Convert the input image into feature channels.
        self.from_image = EqualizedConv2d(
            image_channels,
            self.channels[image_resolution],
            kernel_size=1,
            activation="lrelu",
        )

        self.blocks = nn.ModuleList(
            [
                DiscriminatorBlock(
                    in_channels=self.channels[r],
                    out_channels=self.channels[r // 2],
                    resolution=r,
                )
                for r in self.block_resolutions
            ]
        )

        self.mbstd = MinibatchStdDev(group_size=mbstd_group_size)

        final_channels = self.channels[4]

        self.final_conv = EqualizedConv2d(
            final_channels + 1, final_channels, kernel_size=3, activation="lrelu"
        )
        self.final_dense = EqualizedLinear(
            final_channels * 4 * 4, final_channels, activation="lrelu"
        )
        self.out = EqualizedLinear(final_channels, 1, activation="linear")

    def forward(self, image):
        expected = (self.image_channels, self.image_resolution, self.image_resolution)

        if image.ndim != 4 or tuple(image.shape[1:]) != expected or image.shape[0] == 0:
            raise ValueError(
                f"Expected nonempty [B, {expected[0]}, {expected[1]}, {expected[2]}]."
            )

        x = self.from_image(image)
        for block in self.blocks:
            x = block(x)

        x = self.mbstd(x)
        x = self.final_conv(x)
        x = self.final_dense(x.flatten(1))

        return self.out(x)
