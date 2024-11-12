import torch
import torch.nn as nn


class TruncatedMoCoV3(nn.Module):
    def __init__(self, base_encoder, truncate_after_layer, dim=256, device='cuda'):
        super(TruncatedMoCoV3, self).__init__()
        self.device = device
        modules = list(base_encoder.children())[:-1]

        # Truncate the model up to the specified layer
        self.truncated_encoder = nn.Sequential(
            *modules[:truncate_after_layer]
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

class CombinedModel(nn.Module):
    def __init__(self, moco_model, classifier):
        super(CombinedModel, self).__init__()
        self.moco_model = moco_model
        self.classifier = classifier

    def forward(self, x):
        features = self.moco_model(x)
        output = self.classifier(features)
        return output