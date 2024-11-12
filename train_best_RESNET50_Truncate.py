import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import numpy as np
import random

from functions.RESNET50_Truncate import save_training_info, save_model_and_hyperparameters, load_hyperparameters, train_model,evaluate_model, generate_transform_combinations, load_training_info, load_best_model, AugmentedDataset
from Models.RESNET50_TRUNCATE import TruncatedMoCoV3, Classifier




def main():
    parser = argparse.ArgumentParser(description='Fine-tuning MoCo v3 for Weather Classification')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the best pre-trained MoCo v3 model')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the best hyperparameters configuration')
    parser.add_argument('--epochs', default=25, type=int, help='Number of epochs to train')
    parser.add_argument('--save_dir', default='saved_models', type=str, help='Directory to save trained models')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--k_folds', default=5, type=int, help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--color_aug', action='store_true', help='Enable color augmentation')
    parser.add_argument('--geom_aug', action='store_true', help='Enable geometric augmentation')
    parser.add_argument('--num_color_transforms', type=int, default=0,
                        help='Number of color transformations to apply (randomly selected)')
    parser.add_argument('--num_geom_transforms', type=int, default=0,
                        help='Number of geometric transformations to apply (randomly selected)')
    parser.add_argument('--geom_transforms', nargs='+', type=str, default=None,
                        help='List of geometric transformations to apply')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Charger les hyperparamètres pour assurer la répétabilité
    hyperparameters = load_hyperparameters(args.config_path)
    batch_size = hyperparameters['batch_size']
    lr = hyperparameters['lr']
    truncate_layer = hyperparameters['truncate_layer']
    seed = hyperparameters.get('seed', args.seed)

    # Fixer la graine aléatoire pour la répétabilité
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        seed = random.randint(0, 1000000)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        hyperparameters['seed'] = seed

    # Définir les transformations disponibles
    available_color_transforms = [
        ('brightness', transforms.ColorJitter(brightness=0.5)),
        ('contrast', transforms.ColorJitter(contrast=0.5)),
        ('saturation', transforms.ColorJitter(saturation=0.5)),
        ('hue', transforms.ColorJitter(hue=0.1)),
        ('grayscale', transforms.RandomGrayscale(p=1.0))
    ]

    available_geom_transforms_dict = {
        'horizontal_flip': transforms.RandomHorizontalFlip(p=1.0),
        'vertical_flip': transforms.RandomVerticalFlip(p=1.0),
        'rotation': transforms.RandomRotation(degrees=15),
        'affine': transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        'resized_crop': transforms.RandomResizedCrop(224, scale=(0.8, 1.0))
    }
    available_geom_transforms = list(available_geom_transforms_dict.items())

    # Récupérer les transformations sélectionnées pour les augmentations de données
    selected_color_transforms = []
    selected_geom_transforms = []
    selected_color_names = []
    selected_geom_names = []

    if args.color_aug and args.num_color_transforms > 0:
        num_color_transforms = min(args.num_color_transforms, len(available_color_transforms))
        selected_color_transforms = random.sample(available_color_transforms, num_color_transforms)
        selected_color_names = [name for name, _ in selected_color_transforms]
        selected_color_transforms_transforms = [transform for _, transform in selected_color_transforms]
    else:
        selected_color_transforms_transforms = [transforms.Lambda(lambda x: x)]

    if args.geom_aug:
        if args.geom_transforms:
            selected_geom_transforms = []
            selected_geom_names = []
            for t in args.geom_transforms:
                if t in available_geom_transforms_dict:
                    selected_geom_transforms.append((t, available_geom_transforms_dict[t]))
                    selected_geom_names.append(t)
                else:
                    print(f"Warning: Geometric transform '{t}' is not recognized.")
            selected_geom_transforms_transforms = [transform for _, transform in selected_geom_transforms]
        elif args.num_geom_transforms > 0:
            num_geom_transforms = min(args.num_geom_transforms, len(available_geom_transforms))
            selected_geom_transforms = random.sample(available_geom_transforms, num_geom_transforms)
            selected_geom_names = [name for name, _ in selected_geom_transforms]
            selected_geom_transforms_transforms = [transform for _, transform in selected_geom_transforms]
        else:
            selected_geom_transforms_transforms = [transforms.Lambda(lambda x: x)]
    else:
        selected_geom_transforms_transforms = [transforms.Lambda(lambda x: x)]

    # Base transform
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Générer les combinaisons de transformations
    transform_combinations = generate_transform_combinations(
        selected_geom_transforms_transforms,
        selected_color_transforms_transforms,
        base_transform
    )

    # Créer le dataset sans transformations initiales
    dataset = datasets.ImageFolder(root=os.path.join(args.data, "train"), transform=None)

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard')) if args.tensorboard else None

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=seed)

    fold_results = []
    best_model_results = load_training_info(args.save_dir, 'best_model_results.json') or []
    best_model_performance = float('inf')
    best_global_model_path = None

    training_info = load_training_info(args.save_dir, 'training_info.json') or {
        "num_classes": len(dataset.classes),
        "class_names": dataset.classes,
        "num_samples_per_class": {cls: len([item for item in dataset.imgs if dataset.classes[item[1]] == cls]) for cls in dataset.classes},
        "total_num_samples": len(dataset),
        "num_epochs": args.epochs,
        "num_folds": args.k_folds,
        "fold_results": []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'FOLD {fold}')

        # Créer le dataset de validation avec la transformation de base
        val_dataset = datasets.ImageFolder(root=dataset.root, transform=base_transform)
        val_subset = Subset(val_dataset, val_idx)

        # Créer le dataset d'entraînement augmenté
        train_subset = Subset(dataset, train_idx)
        augmented_dataset = AugmentedDataset(train_subset, transform_combinations)

        # Création des DataLoaders
        train_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        base_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        moco_model = TruncatedMoCoV3(base_encoder, truncate_layer, dim=256, device=device).to(device)
        classifier = Classifier(input_dim=256, num_classes=len(dataset.classes)).to(device)

        load_best_model(classifier, moco_model, args.model_path)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(moco_model.parameters()) + list(classifier.parameters())), lr=lr, momentum=0.9)

        moco_model, classifier = train_model(moco_model, classifier, train_loader, criterion, optimizer, num_epochs=args.epochs, writer=writer, fold=fold)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(moco_model, classifier, val_loader, criterion, writer=writer, fold=fold)
        fold_results.append((val_loss, val_accuracy, val_precision, val_recall, val_f1))

        fold_result = {
            "fold": fold,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1
        }

        training_info["fold_results"].append(fold_result)

        # Enregistrer les performances du fold dans un fichier JSON
        fold_performance_path = os.path.join(args.save_dir, f"fold_{fold}_performance.json")
        with open(fold_performance_path, 'w') as f:
            json.dump(fold_result, f, indent=4)
        print(f"Fold {fold} performance saved to {fold_performance_path}")

        if val_loss < best_model_performance:
            best_model_performance = val_loss
            best_global_model_path = os.path.join(args.save_dir, f"best_global_model.pth")
            save_model_and_hyperparameters(moco_model, classifier, hyperparameters, args.save_dir, "best_global_model")

        best_fold_model_path = os.path.join(args.save_dir, f"best_model_fold_{fold}.pth")
        best_fold_model_info = next((model for model in best_model_results if model["fold"] == fold), None)

        if best_fold_model_info is None or val_loss < best_fold_model_info["val_loss"]:
            save_model_and_hyperparameters(moco_model, classifier, hyperparameters, args.save_dir, f"best_model_fold_{fold}")
            best_model_results = [model for model in best_model_results if model["fold"] != fold]
            best_model_results.append({
                "fold": fold,
                "model_path": best_fold_model_path,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1
            })

    avg_results = np.mean(fold_results, axis=0)
    print(f"Average Validation Loss: {avg_results[0]:.4f}, Accuracy: {avg_results[1]:.4f}, Precision: {avg_results[2]:.4f}, Recall: {avg_results[3]:.4f}, F1 Score: {avg_results[4]:.4f}")

    training_info["average_results"] = {
        "avg_val_loss": avg_results[0],
        "avg_accuracy": avg_results[1],
        "avg_precision": avg_results[2],
        "avg_recall": avg_results[3],
        "avg_f1": avg_results[4]
    }

    save_training_info(training_info, args.save_dir, 'training_info.json')
    save_training_info(best_model_results, args.save_dir, 'best_model_results.json')

    if writer:
        writer.close()

    print(f"Best global model saved at {best_global_model_path} with validation loss: {best_model_performance:.4f}")

if __name__ == '__main__':
    main()
