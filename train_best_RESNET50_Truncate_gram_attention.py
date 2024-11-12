import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score
import numpy as np


class TruncatedResNet50(nn.Module):
    def __init__(self, base_encoder, truncate_after_layer, num_classes, gram_matrix_size, device='cpu'):
        super(TruncatedResNet50, self).__init__()
        self.device = device

        # Supprime la dernière couche linéaire de ResNet50 avant la troncature
        layers = list(base_encoder.children())[:-1]
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
            return torch.zeros((x.size(0), self.num_classes), requires_grad=True).to(self.device)

        gram_matrices = [torch.nn.functional.adaptive_avg_pool2d(gram, (self.gram_matrix_size, self.gram_matrix_size))
                         for gram in gram_matrices]

        gram_matrices = torch.stack(gram_matrices, dim=1)
        gram_matrices = gram_matrices.flatten(2)
        gram_matrices = gram_matrices.permute(1, 0, 2)

        attn_output, _ = self.attention(gram_matrices, gram_matrices, gram_matrices)
        attn_output = attn_output.mean(dim=0)

        return self.classifier(attn_output.view(attn_output.size(0), -1))


def load_model(model, model_path, device):
    if os.path.isfile(model_path):
        print(f"Loading pre-trained ResNet50 model from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"No model found at {model_path}")


def save_model_weights(model, save_path):
    state_dict = {
        'truncated_encoder': model.truncated_encoder.state_dict(),
        'classifier': model.classifier.state_dict(),
        'attention': model.attention.state_dict()  # Ajouter les poids de l'attention
    }
    torch.save(state_dict, save_path)
    print(f"Model weights saved to {save_path}")


def load_model_weights(model, load_path):
    if os.path.isfile(load_path):
        state_dict = torch.load(load_path, map_location=model.device)

        if 'truncated_encoder' in state_dict:
            model.truncated_encoder.load_state_dict(state_dict['truncated_encoder'], strict=True)
        if 'classifier' in state_dict:
            model.classifier.load_state_dict(state_dict['classifier'], strict=True)
        if 'attention' in state_dict:
            model.attention.load_state_dict(state_dict['attention'], strict=True)

        print(f"Model weights loaded from {load_path}.")
    else:
        print(f"No weights file found at {load_path}. Proceeding without loading weights.")


def train_model(model, train_loader, criterion, optimizer, num_epochs=25, writer=None, fold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            print(f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        if writer:
            writer.add_scalar(f"Fold_{fold}/Train/Loss", epoch_loss, epoch)
    return model


def evaluate_model(model, val_loader, criterion, writer=None, fold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    val_loss = 0.0
    corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    total_loss = val_loss / len(val_loader.dataset)
    accuracy = corrects.double() / len(val_loader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    print(f'Fold {fold}, Validation Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    if writer:
        writer.add_scalar(f"Fold_{fold}/Validation/Loss", total_loss)
        writer.add_scalar(f"Fold_{fold}/Validation/Accuracy", accuracy)
        writer.add_scalar(f"Fold_{fold}/Validation/Precision", precision)
        writer.add_scalar(f"Fold_{fold}/Validation/Recall", recall)
    return total_loss, accuracy.item(), precision, recall

def set_parameter_requires_grad(model, freeze_encoder):
    if freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name and "attention" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                print(f"Layer {name} is unfrozen.")
    else:
        for param in model.parameters():
            param.requires_grad = True



def main():
    parser = argparse.ArgumentParser(
        description='ResNet50 Fine-Tuning for Classification with Hyperparameter Loading')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained ResNet50 model')
    parser.add_argument('--epochs', default=25, type=int, help='Number of epochs to train')
    parser.add_argument('--save_dir', default='saved_models', type=str, help='Directory to save trained models')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--k_folds', default=5, type=int, help='Number of folds for cross-validation')
    parser.add_argument('--freeze_layers', action='store_true', help='Freeze the encoder layers')
    parser.add_argument('--load_hyperparams', type=str, help='Path to the hyperparameters JSON file to load')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=os.path.join(args.data, "train"), transform=transform)

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard')) if args.tensorboard else None

    if args.load_hyperparams:
        with open(args.load_hyperparams, 'r') as f:
            hyperparams = json.load(f)
        hidden_dims = hyperparams['hidden_dims']
        num_layers = hyperparams['num_layers']
        batch_size = hyperparams['batch_size']
        lr = hyperparams['lr']
        truncate_layer = hyperparams['truncate_layer']
        gram_matrix_size = hyperparams['gram_matrix_size']
    else:
        raise ValueError("Please provide a path to a hyperparameters JSON file using --load_hyperparams.")

    kfold = KFold(n_splits=args.k_folds, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        base_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        model = TruncatedResNet50(base_encoder, truncate_layer, num_classes=len(dataset.classes),
                                  gram_matrix_size=gram_matrix_size, device=device).to(device)

        load_model(model, args.model_path, device)
        set_parameter_requires_grad(model, args.freeze_layers)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)

        model = train_model(model, train_loader, criterion, optimizer, num_epochs=args.epochs, writer=writer, fold=fold)
        val_loss, val_accuracy, val_precision, val_recall = evaluate_model(model, val_loader, criterion, writer=writer,
                                                                           fold=fold)

        # Save model, performance and hyperparameters for each fold
        fold_best_path = os.path.join(args.save_dir, f"best_model_fold_{fold}.pth")
        fold_best_perf_path = os.path.join(args.save_dir, f"best_performance_fold_{fold}.json")
        fold_hyperparams_path = os.path.join(args.save_dir, f"best_hyperparameters_fold_{fold}.json")

        # Save the model and performance if it improves
        save_model_weights(model, fold_best_path)

        fold_best_perf = {'accuracy': val_accuracy, 'precision': val_precision, 'recall': val_recall}
        with open(fold_best_perf_path, 'w') as f:
            json.dump(fold_best_perf, f)

        fold_hyperparams = {
            'hidden_dims': hidden_dims,
            'num_layers': num_layers,
            'batch_size': batch_size,
            'lr': lr,
            'truncate_layer': truncate_layer,
            'gram_matrix_size': gram_matrix_size,
            'model_path': fold_best_path
        }
        with open(fold_hyperparams_path, 'w') as f:
            json.dump(fold_hyperparams, f, indent=4)


if __name__ == '__main__':
    main()
