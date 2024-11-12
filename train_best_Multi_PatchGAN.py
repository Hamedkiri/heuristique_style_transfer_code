import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from Models.Models_Multi_PatchGAN import MultiScaleDiscriminator
from functions.functions_Multi_PatchGAN import train_model, evaluate_model

# Définition des constantes globales

def main():
    parser = argparse.ArgumentParser(
        description='Fine-Tuning for Image Classification with Loaded Model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--epochs', default=25, type=int, help='Number of epochs to train')
    parser.add_argument('--save_dir', default='Model_Multi_scale_PatchGAN/best', type=str, help='Directory to save trained models')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--k_folds', default=2, type=int, help='Number of folds for cross-validation')
    parser.add_argument('--model_path', type=str, required=False, help='Path to model weights')
    parser.add_argument('--config_path', type=str, required=True, help='Path to model hyperparameters configuration')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Charger les hyperparamètres depuis le fichier de configuration
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    hidden_dims = config.get('hidden_dims', [64] * 8)
    batch_size = config.get('batch_size', 32)
    lr = config.get('lr', 0.01)
    patch_sizes = config.get('patch_sizes', {'small': 70, 'medium': 70, 'large': 70})
    # num_classes sera déterminé à partir du jeu de données

    # Préparer les transformations appliquées aux images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Charger le jeu de données d'entraînement
    dataset = datasets.ImageFolder(root=os.path.join(args.data, "train"), transform=transform)

    # Déterminer le nombre de classes à partir du jeu de données
    num_classes = len(dataset.classes)
    print(f"Nombre de classes détecté : {num_classes}")

    # Mettre à jour la configuration avec le nombre de classes détecté
    config['num_classes'] = num_classes

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard')) if args.tensorboard else None

    kfold = KFold(n_splits=args.k_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'========== FOLD {fold} ==========')
        # Créer un sous-ensemble pour l'entraînement et la validation
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Créer les DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Initialiser le modèle pour chaque fold
        model = MultiScaleDiscriminator(
            input_nc=3,
            ndf=64,
            norm='batch',
            patch_sizes=patch_sizes,
            num_classes=num_classes  # Utiliser le nombre de classes détecté
        ).to(device)

        # Charger les poids pré-entraînés si un chemin est fourni
        if args.model_path is not None:
            # Charger les poids pré-entraînés
            pretrained_dict = torch.load(args.model_path, map_location=device)
            model_dict = model.state_dict()

            # Filtrer les poids pour ignorer ceux qui ne correspondent pas en taille
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Fold {fold}: Poids du modèle chargés avec succès.")
        else:
            print(f"Fold {fold}: Aucun poids pré-entraîné chargé, entraînement à partir de zéro.")

        # Définir la fonction de perte et l'optimiseur
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # Entraîner le modèle
        model = train_model(model, train_loader, criterion, optimizer, num_epochs=args.epochs, writer=writer, fold=fold)

        # Évaluer le modèle
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(
            model, val_loader, criterion, writer=writer, fold=fold
        )
        fold_results.append({
            'fold': fold,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        })

        # Sauvegarder les poids du modèle pour ce fold
        fold_model_path = os.path.join(args.save_dir, f'model_fold_{fold}.pth')
        torch.save(model.state_dict(), fold_model_path)
        print(f"Fold {fold}: Poids du modèle sauvegardés à {fold_model_path}")

    # Sauvegarder les hyperparamètres mis à jour
    with open(os.path.join(args.save_dir, 'retrained_hyperparameters.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Enregistrer les performances du modèle réentraîné dans un fichier JSON
    with open(os.path.join(args.save_dir, 'retrained_performance.json'), 'w') as f:
        json.dump(fold_results, f, indent=4)

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
