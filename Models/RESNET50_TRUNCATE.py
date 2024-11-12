import torch
import torch.nn as nn



class TruncatedMoCoV3(nn.Module):
    def __init__(self, base_encoder, truncate_after_layer, num_classes, dim=256, device='cpu'):
        super(TruncatedMoCoV3, self).__init__()
        self.device = device

        # Remove the last layer (usually the classification layer) from the base encoder
        modules = list(base_encoder.children())[:-1]

        # Truncate the model up to the specified layer
        self.truncated_encoder = nn.Sequential(
            *modules[:truncate_after_layer]
        ).to(self.device)

        #self.truncated_encoder = nn.Sequential(
           # *list(base_encoder.children())[:truncate_after_layer]
       # ).to(self.device)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        if not self.truncated_encoder:
            raise ValueError("Truncated encoder is empty.")

        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output_shape = self.truncated_encoder(dummy_input)
            output_shape = self.pool(output_shape)
            output_shape = self.flatten(output_shape)
            num_features = output_shape.shape[1]

        self.fc = nn.Linear(num_features, dim).to(self.device)
        self.classifier = nn.Linear(dim, num_classes).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.truncated_encoder(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.classifier(x)
        return x


class TruncatedMoCoV3_best(nn.Module):
    def __init__(self, base_encoder, truncate_after_layer, dim=256, device='cuda'):
        super(TruncatedMoCoV3_best, self).__init__()
        self.device = device
        self.truncated_encoder = nn.Sequential(
            *list(base_encoder.children())[:truncate_after_layer]
        ).to(self.device)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output_shape = self.truncated_encoder(dummy_input)
            output_shape = self.pool(output_shape)
            output_shape = self.flatten(output_shape)
            num_features = output_shape.shape[1]

        self.fc = nn.Linear(num_features, dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.truncated_encoder(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)