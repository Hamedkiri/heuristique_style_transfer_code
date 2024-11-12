import os
import torch
import torch.nn as nn

import functools
import torch.nn.functional as F



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

class VariablePatchesNLayerDiscriminator_test(nn.Module):
    """
    Définit un discriminateur PatchGAN avec un nombre variable de couches, une taille de patch configurable,
    des matrices de Gram pour la régularisation, et des mécanismes d'attention.
    """
    def __init__(self, input_nc=3, ndf=64, norm="instance", tensorboard_logdir=None, global_step=None,
                 patch_size=70, num_classes=10, gram_matrix_dim=64, pooling_type='avg'):
        super().__init__()
        self.tensorboard_logdir = tensorboard_logdir
        self.writer = None  # Pas besoin de SummaryWriter pour le test
        self.global_step = global_step
        self.num_classes = num_classes
        self.gram_matrix_dim = gram_matrix_dim
        self.pooling_type = pooling_type

        # Feature Extractor: Convolutional layers
        self.feature_extractor = nn.Sequential()
        if norm == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        else:
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

        num_filters = ndf
        input_channels = input_nc
        kernel_size = 4
        padding = 1
        stride = 2
        receptive_field_size = patch_size
        layer_idx = 0

        while receptive_field_size > 4 and num_filters <= 512:
            conv = nn.Conv2d(input_channels, num_filters, kernel_size, stride, padding)
            self.feature_extractor.add_module(f'conv{layer_idx}', conv)
            self.feature_extractor.add_module(f'norm{layer_idx}', norm_layer(num_filters))
            self.feature_extractor.add_module(f'relu{layer_idx}', nn.ReLU(inplace=True))
            input_channels = num_filters
            num_filters *= 2
            receptive_field_size /= stride
            layer_idx += 1

        final_conv = nn.Conv2d(input_channels, num_filters, kernel_size, 1, padding)
        self.feature_extractor.add_module(f'final_conv', final_conv)
        self.feature_extractor.add_module(f'final_norm', norm_layer(num_filters))
        self.feature_extractor.add_module(f'final_relu', nn.ReLU(inplace=True))
        self.feature_extractor.add_module(f'final_conv_ndf', nn.Conv2d(num_filters, ndf, kernel_size, 1, padding))

        # Projection Layers
        self.projection_layers = nn.ModuleList()
        for name, module in self.feature_extractor.named_modules():
            if isinstance(module, nn.Conv2d):
                num_channels = module.out_channels
                proj_layer = nn.Conv2d(num_channels, self.gram_matrix_dim, kernel_size=1)
                self.projection_layers.append(proj_layer)

        # Attention Mechanisms
        self.attention_per_layer = nn.MultiheadAttention(embed_dim=ndf, num_heads=8)
        self.attention_per_patch = nn.MultiheadAttention(embed_dim=ndf, num_heads=8)

        # Classifier
        self.classifier = nn.Linear(ndf, self.num_classes)

        # Feature Projection
        self.feature_projection = nn.Linear(self.gram_matrix_dim * self.gram_matrix_dim, ndf)

    def forward(self, input):
        assert input.ndim == 4, f"Input must be NCHW, got {input.shape}"
        x = input
        feature_maps = []
        proj_layer_idx = 0

        # Réinitialiser gram_norms pour chaque passage avant
        self.gram_norms = []

        for idx, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if torch.isnan(x).any():
                print(f"NaN detected after layer {idx}")
                x = torch.nan_to_num(x, nan=0.0)
            if isinstance(layer, nn.Conv2d):
                proj_layer = self.projection_layers[proj_layer_idx]
                x_proj = proj_layer(x)
                if torch.isnan(x_proj).any():
                    print(f"NaN detected in projected feature map at layer {idx}")
                    x_proj = torch.nan_to_num(x_proj, nan=0.0)
                # Normalisation
                x_proj = F.layer_norm(x_proj, x_proj.shape[1:])
                feature_maps.append(x_proj)
                proj_layer_idx += 1

        if not feature_maps:
            raise ValueError("No feature maps were collected. Check the model architecture and input data.")

        gram_matrices_per_layer = []
        epsilon = 1e-6  # Pour éviter la division par zéro

        for fm_idx, feature_map in enumerate(feature_maps):
            # Adaptive pooling
            pooled_feature_map = F.adaptive_avg_pool2d(feature_map, output_size=(4, 4))

            # Normalisation avant calcul de la matrice de Gram
            pooled_feature_map = F.layer_norm(pooled_feature_map, pooled_feature_map.shape[1:])

            num_channels = self.gram_matrix_dim
            # Aplatir les dimensions spatiales
            fm_flattened = pooled_feature_map.view(pooled_feature_map.size(0), num_channels, -1)

            # Calculer la matrice de Gram et la normaliser
            gram_matrices = torch.bmm(fm_flattened, fm_flattened.transpose(1, 2)) / (fm_flattened.size(-1) + epsilon)

            # Calculer la norme de la matrice de Gram pour la pénalisation
            gram_norm = torch.norm(gram_matrices, p='fro', dim=(1, 2))
            self.gram_norms.append(gram_norm)

            # Aplatir les matrices de Gram
            gram_matrices_flat = gram_matrices.view(gram_matrices.size(0), -1)

            # Projection des caractéristiques
            projected_features = self.feature_projection(gram_matrices_flat)

            if torch.isnan(projected_features).any():
                print(f"NaN detected in projected features at layer {fm_idx}, replacing NaNs with zeros.")
                projected_features = torch.nan_to_num(projected_features, nan=0.0)

            gram_matrices_per_layer.append(projected_features)

        if not gram_matrices_per_layer:
            raise ValueError("No valid Gram matrices computed. Check for NaNs in the computations.")

        # Empiler les matrices de Gram
        gram_matrices_stacked = torch.stack(gram_matrices_per_layer, dim=0)

        # Mécanismes d'attention
        attended_layer_matrix, _ = self.attention_per_layer(gram_matrices_stacked, gram_matrices_stacked, gram_matrices_stacked)
        attended_gram_matrices_per_patch, _ = self.attention_per_patch(attended_layer_matrix, attended_layer_matrix, attended_layer_matrix)

        # Agrégation des caractéristiques
        aggregated_features = torch.mean(attended_gram_matrices_per_patch, dim=0)

        # embeddings = aggregated_features
        # Suppression du pooling sur la dimension du batch
        # Cela garantit que les embeddings sont de taille [batch_size, ndf] et correspondent aux labels
        embeddings = aggregated_features

        # Classification finale
        output = self.classifier(aggregated_features)
        return embeddings, output

    def get_gram_norms(self):
        """
        Retourne les normes des matrices de Gram calculées lors du dernier passage avant.
        """
        return self.gram_norms

class MultiScaleDiscriminator_test(nn.Module):
    """
    Définit un discriminateur multi-échelle PatchGAN avec trois types de patches : petit, moyen, grand.
    """
    def __init__(self, input_nc=3, ndf=64, norm='batch', tensorboard_logdir=None, global_step=None,
                 patch_sizes={'small':10, 'medium':70, 'large':150}, num_classes=10,
                 gram_matrix_dim=64, pooling_type='avg'):
        super(MultiScaleDiscriminator_test, self).__init__()
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
            discriminator = VariablePatchesNLayerDiscriminator_test(
                input_nc=input_nc,
                patch_size=patch_size,
                ndf=ndf,
                norm=norm,
                tensorboard_logdir=scale_logdir,
                global_step=global_step,
                num_classes=num_classes,
                gram_matrix_dim=gram_matrix_dim,
                pooling_type=pooling_type
            )
            self.scale_discriminators[patch_type] = discriminator

    def forward(self, input):
        # Collecte des résultats de toutes les échelles
        outputs = []
        embeddings_list = []
        x = input
        for patch_type, discriminator in self.scale_discriminators.items():
            embeddings, out = discriminator(x)
            outputs.append(out)
            embeddings_list.append(embeddings)

        # Combinaison des résultats de toutes les échelles
        combined_output = torch.stack(outputs, dim=0).mean(dim=0)
        combined_embeddings = torch.stack(embeddings_list, dim=0).mean(dim=0)
        return combined_embeddings, combined_output

    def get_gram_norms(self):
        """
        Récupère les gram_norms de tous les discriminateurs internes.
        """
        gram_norms = []
        for discriminator in self.scale_discriminators.values():
            gram_norms.extend(discriminator.get_gram_norms())
        return gram_norms