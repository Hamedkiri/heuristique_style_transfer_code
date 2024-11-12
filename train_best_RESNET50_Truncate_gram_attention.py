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
from Models.Models_RESNET50_TRUNCATE_GRAM_with_Attention import TruncatedResNet50
from functions.functions_RESNET50_Truncate_Gram_Attention import load_model, set_parameter_requires_grad, train_model, save_model_weights, evaluate_model





def main():
    parser = argparse.ArgumentParser(
        description='ResNet50 Fine-Tuning for Classification with Hyperparameter Loading')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained ResNet50 model')
    parser.add_argument('--epochs', default=25, type=int, help='Number of epochs to train')
    parser.add_argument('--save_dir', default='saved_models_attention_gram_resnet50', type=str, help='Directory to save trained models')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--k_folds', default=2, type=int, help='Number of folds for cross-validation')
    parser.add_argument('--freeze_layers', action='store_true', help='Freeze the encoder layers')
    parser.add_argument('--config_path', type=str, help='Path to the hyperparameters JSON file to load')

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

    if args.config_path:
        with open(args.config_path, 'r') as f:
            hyperparams = json.load(f)
        hidden_dims = hyperparams['hidden_dims']
        num_layers = hyperparams['num_layers']
        batch_size = hyperparams['batch_size']
        lr = hyperparams['lr']
        truncate_layer = hyperparams['truncate_layer']
        gram_matrix_size = hyperparams['gram_matrix_size']
    else:
        raise ValueError("Please provide a path to a hyperparameters JSON file using --config_path.")

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

        # Save the model and performance
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

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
