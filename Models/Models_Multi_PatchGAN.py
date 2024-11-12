import os
import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter


# Définition des constantes globales
PATCH_TYPES = {
    'small': (4, 30),
    'medium': (31, 80),
    'large': (81, 150)
}

class VariablePatchesNLayerDiscriminator(nn.Module):
    """
    Définit un discriminateur PatchGAN avec un nombre variable de couches et une taille de patch configurable.
    """
    def __init__(self, input_nc=3, ndf=64, norm="instance", tensorboard_logdir=None, global_step=None, patch_size=70, num_classes=10):
        super().__init__()
        self.tensorboard_logdir = tensorboard_logdir
        self.writer = SummaryWriter(log_dir=tensorboard_logdir) if tensorboard_logdir else None
        self.global_step = global_step
        self.num_classes = num_classes

        if norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        layers = []
        num_filters = ndf
        kernel_size = 4
        padding = 1
        stride = 2
        receptive_field_size = patch_size

        while receptive_field_size > 4 and num_filters <= 512:
            layers.append(nn.Conv2d(input_nc, num_filters, kernel_size, stride, padding))
            layers.append(norm_layer(num_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            input_nc = num_filters
            num_filters *= 2
            receptive_field_size /= stride

        final_conv = nn.Conv2d(input_nc, num_filters, kernel_size, 1, padding)
        layers.append(final_conv)
        layers.append(norm_layer(num_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(num_filters, self.num_classes, kernel_size, 1, padding))

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        x = input
        for layer in self.model:
            x = layer(x)
        x = torch.mean(x, dim=[2, 3])  # Moyenne sur les dimensions spatiales (H, W)
        return x

    def close_writer(self):
        if self.writer:
            self.writer.close()


class MultiScaleDiscriminator(nn.Module):
    """
    Définit un discriminateur multi-échelle PatchGAN avec trois types de patches : petit, moyen, grand.
    """
    def __init__(self, input_nc=3, ndf=64, norm='batch', tensorboard_logdir=None, global_step=None, patch_sizes={'small':70, 'medium':70, 'large':70}, num_classes=10):
        super(MultiScaleDiscriminator, self).__init__()
        self.patch_sizes = patch_sizes
        self.tensorboard_logdir = tensorboard_logdir
        self.global_step = global_step

        # Création des discriminateurs pour chaque type de patch
        self.scale_discriminators = nn.ModuleDict()
        for patch_type in PATCH_TYPES:
            patch_size = patch_sizes.get(patch_type, 70)  # Taille de patch par défaut si non spécifiée
            if tensorboard_logdir:
                scale_logdir = os.path.join(tensorboard_logdir, patch_type)
            else:
                scale_logdir = None
            discriminator = VariablePatchesNLayerDiscriminator(
                input_nc=input_nc,
                patch_size=patch_size,
                ndf=ndf,
                norm=norm,
                tensorboard_logdir=scale_logdir,
                global_step=global_step,
                num_classes=num_classes
            )
            self.scale_discriminators[patch_type] = discriminator

        # Downsampling pour les échelles suivantes
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, input):
        # Collecte des résultats de toutes les échelles
        results = []
        x = input
        for patch_type, discriminator in self.scale_discriminators.items():
            out = discriminator(x)
            results.append(out)
            x = self.downsample(x)  # Downsample pour l'échelle suivante

        # Combinaison des résultats de toutes les échelles (par exemple, en les moyennant)
        combined_output = torch.stack(results, dim=0).mean(dim=0)
        return combined_output