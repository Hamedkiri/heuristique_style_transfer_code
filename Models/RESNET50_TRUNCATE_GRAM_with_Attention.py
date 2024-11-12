import torch
import torch.nn as nn






torch.autograd.set_detect_anomaly(True)  # Pour activer la détection d'anomalies

class TruncatedResNet50(nn.Module):
    def __init__(self, base_encoder, truncate_after_layer, num_classes, gram_matrix_size, device='cpu'):
        super(TruncatedResNet50, self).__init__()
        self.device = device

        # Supprime la dernière couche linéaire de ResNet50 avant la troncature
        layers = list(base_encoder.children())[:-1]  # Enlever la dernière couche linéaire
        self.truncated_encoder = nn.Sequential(*layers[:truncate_after_layer]).to(self.device)

        self.num_classes = num_classes
        self.gram_matrix_size = gram_matrix_size
        self.classifier = nn.Linear(self.gram_matrix_size ** 2, self.num_classes).to(self.device)
        self.attention = nn.MultiheadAttention(embed_dim=self.gram_matrix_size ** 2, num_heads=1).to(self.device)

    def gram_matrix(self, activations):
        (b, ch, h, w) = activations.size()
        features = activations.view(b, ch, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(h * w)

    def forward(self, x):
        x = x.to(self.device)
        gram_matrices = []
        for idx, layer in enumerate(self.truncated_encoder):
            x = layer(x)
            if isinstance(layer, nn.Sequential):  # Calculer la matrice de Gram pour les couches séquentielles
                gram_matrices.append(self.gram_matrix(x))

        if not gram_matrices:
            return torch.zeros((x.size(0), self.num_classes), requires_grad=True).to(
                self.device)  # Assurez-vous que la sortie a la bonne dimension

        gram_matrices = [torch.nn.functional.adaptive_avg_pool2d(gram, (self.gram_matrix_size, self.gram_matrix_size))
                         for gram in gram_matrices]

        gram_matrices = torch.stack(gram_matrices, dim=1)
        gram_matrices = gram_matrices.flatten(2)
        gram_matrices = gram_matrices.permute(1, 0, 2)

        attn_output, _ = self.attention(gram_matrices, gram_matrices, gram_matrices)
        attn_output = attn_output.mean(dim=0)

        return self.classifier(attn_output.view(attn_output.size(0), -1))
