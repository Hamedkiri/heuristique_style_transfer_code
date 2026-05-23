import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import numpy as np
import random

from functions.functions_RESNET50_Truncate import (
    save_training_info,
    save_model_and_hyperparameters,
    load_hyperparameters,
    train_model,
    evaluate_model,
    generate_transform_combinations,
    load_training_info,
    AugmentedDataset
)
from Models.Models_RESNET50_TRUNCATE import TruncatedMoCoV3, Classifier


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning MoCo v3 for Weather Classification')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--model_path', type=str, help='Path to the best pre-trained MoCo v3 model')
    parser.add_argument('--config_path', type=str, required=True, help='Path to hyperparameters configuration')
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs to train')
    parser.add_argument('--save_dir', default='saved_models', type=str, help='Directory to save trained models')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--k_folds', default=2, type=int, help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--color_aug', action='store_true', help='Enable color augmentation')
    parser.add_argument('--geom_aug', action='store_true', help='Enable geometric augmentation')
    parser.add_argument('--num_color_transforms', type=int, default=0,
                        help='Number of color transformations to apply randomly')
    parser.add_argument('--num_geom_transforms', type=int, default=0,
                        help='Number of geometric transformations to apply randomly')
    parser.add_argument('--geom_transforms', nargs='+', type=str, default=None,
                        help='List of geometric transformations to apply')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze all encoder layers and train only the classifier')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Chargement des hyperparamètres et graine de réplication ---
    hyperparameters = load_hyperparameters(args.config_path)
    batch_size     = hyperparameters['batch_size']
    lr             = hyperparameters['lr']
    truncate_layer = hyperparameters['truncate_layer']
    seed           = hyperparameters.get('seed', args.seed)

    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    else:
        seed = random.randint(0, 1_000_000)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        hyperparameters['seed'] = seed

    # --- Préparation des augmentations ---
    available_color = [
        ('brightness', transforms.ColorJitter(brightness=0.5)),
        ('contrast',   transforms.ColorJitter(contrast=0.5)),
        ('saturation', transforms.ColorJitter(saturation=0.5)),
        ('hue',        transforms.ColorJitter(hue=0.1)),
        ('grayscale',  transforms.RandomGrayscale(p=1.0))
    ]
    available_geom = {
        'horizontal_flip': transforms.RandomHorizontalFlip(p=1.0),
        'vertical_flip':   transforms.RandomVerticalFlip(p=1.0),
        'rotation':        transforms.RandomRotation(degrees=15),
        'affine':          transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1)),
        'resized_crop':    transforms.RandomResizedCrop(224, scale=(0.8,1.0))
    }

    if args.color_aug and args.num_color_transforms > 0:
        chosen_c = random.sample(available_color, min(args.num_color_transforms, len(available_color)))
        color_transforms = [t for _, t in chosen_c]
    else:
        color_transforms = [transforms.Lambda(lambda x: x)]

    if args.geom_aug:
        if args.geom_transforms:
            geom_transforms = [available_geom[t] for t in args.geom_transforms if t in available_geom]
        elif args.num_geom_transforms > 0:
            sampled = random.sample(list(available_geom.items()), min(args.num_geom_transforms, len(available_geom)))
            geom_transforms = [t for _, t in sampled]
        else:
            geom_transforms = [transforms.Lambda(lambda x: x)]
    else:
        geom_transforms = [transforms.Lambda(lambda x: x)]

    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    transform_combinations = generate_transform_combinations(
        geom_transforms, color_transforms, base_transform
    )

    # --- Chargement du jeu de données ---
    dataset = datasets.ImageFolder(root=os.path.join(args.data, "train"), transform=None)
    writer  = SummaryWriter(log_dir=os.path.join(args.save_dir,'tensorboard')) if args.tensorboard else None
    kf      = KFold(n_splits=args.k_folds, shuffle=True, random_state=seed)

    fold_results         = []
    best_model_results   = load_training_info(args.save_dir, 'best_model_results.json') or []
    best_model_performance = float('inf')
    best_global_model_path = None

    training_info = load_training_info(args.save_dir, 'training_info.json') or {
        "num_classes": len(dataset.classes),
        "class_names": dataset.classes,
        "num_samples_per_class": {
            cls: len([i for i in dataset.imgs if dataset.classes[i[1]] == cls])
            for cls in dataset.classes
        },
        "total_num_samples": len(dataset),
        "num_epochs": args.epochs,
        "num_folds": args.k_folds,
        "fold_results": []
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'=== FOLD {fold} ===')

        # --- Préparation des sous-ensembles train / val ---
        val_ds     = datasets.ImageFolder(root=dataset.root, transform=base_transform)
        val_subset = Subset(val_ds, val_idx)

        train_subset     = Subset(dataset, train_idx)
        augmented_dataset = AugmentedDataset(train_subset, transform_combinations)

        train_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader   = DataLoader(val_subset,      batch_size=batch_size, shuffle=False, num_workers=4)

        # --- Initialisation des modèles ---
        base_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        moco_model   = TruncatedMoCoV3(base_encoder, truncate_layer, dim=256, device=device).to(device)
        classifier   = Classifier(input_dim=256, num_classes=len(dataset.classes)).to(device)

        # --- Chargement du checkpoint si fourni ---
        classifier_mismatch = False
        if args.model_path:
            ckpt = torch.load(args.model_path, map_location=device)

            # MoCo
            moco_sd = ckpt.get('moco_state_dict', ckpt.get('truncated_moco_state_dict', {}))
            moco_model.load_state_dict(moco_sd, strict=False)

            # Classifier
            cls_sd = ckpt.get('classifier_state_dict', ckpt.get('classifier_state_dict', {}))
            if 'fc.weight' in cls_sd and 'fc.bias' in cls_sd:
                w_ckpt = cls_sd['fc.weight']
                w_cur  = classifier.fc.weight.data
                if w_ckpt.shape == w_cur.shape:
                    classifier.load_state_dict(cls_sd, strict=True)
                    print(f"Fold {fold}: Classifier chargé depuis le checkpoint.")
                else:
                    classifier_mismatch = True
                    print(f"Fold {fold}: mismatch classifier → checkpoint a {tuple(w_ckpt.shape)}, "
                          f"attendu {tuple(w_cur.shape)}. Nouveau classifieur non chargé.")
            else:
                classifier_mismatch = True
                print(f"Fold {fold}: pas de poids de classifieur dans le checkpoint → nouveau classifieur.")

        else:
            print(f"Fold {fold}: pas de checkpoint fourni, entraînement from scratch.")

        # --- Gel de l'encodeur si demandé (le classifieur reste entraînable) ---
        if args.freeze_encoder:
            for name, param in moco_model.named_parameters():
                param.requires_grad = False
            print(f"Fold {fold}: encodeur gelé (seul le classifieur sera entraîné).")

        # --- Critère et optimiseur sur les params.requires_grad=True uniquement ---
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad,
                   list(moco_model.parameters()) + list(classifier.parameters())),
            lr=lr, momentum=0.9
        )

        # --- Boucle d'entraînement ---
        moco_model, classifier = train_model(
            moco_model, classifier, train_loader, criterion, optimizer,
            num_epochs=args.epochs, writer=writer, fold=fold
        )

        # --- Évaluation et enregistrement des métriques ---
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
            moco_model, classifier, val_loader, criterion, writer=writer, fold=fold
        )
        fold_results.append((val_loss, val_acc, val_prec, val_rec, val_f1))

        result = {
            "fold": fold,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1
        }
        training_info["fold_results"].append(result)
        with open(os.path.join(args.save_dir, f"fold_{fold}_performance.json"), 'w') as f:
            json.dump(result, f, indent=4)

        # --- Sauvegarde du meilleur modèle global et par fold ---
        if val_loss < best_model_performance:
            best_model_performance = val_loss
            save_model_and_hyperparameters(
                moco_model, classifier, hyperparameters,
                args.save_dir, "best_global_model"
            )
            best_global_model_path = os.path.join(args.save_dir, "best_global_model.pth")

        fold_info = next((m for m in best_model_results if m["fold"] == fold), None)
        fold_model_path = os.path.join(args.save_dir, f"best_model_fold_{fold}.pth")
        if fold_info is None or val_loss < fold_info["val_loss"]:
            save_model_and_hyperparameters(
                moco_model, classifier, hyperparameters,
                args.save_dir, f"best_model_fold_{fold}"
            )
            best_model_results = [m for m in best_model_results if m["fold"] != fold]
            best_model_results.append({
                "fold": fold,
                "model_path": fold_model_path,
                **{k: result[k] for k in ("val_loss","val_accuracy","val_precision","val_recall","val_f1")}
            })

    # --- Résultats moyens et sauvegarde finale ---
    avg = np.mean(fold_results, axis=0)
    print(f"Avg Loss: {avg[0]:.4f}, Acc: {avg[1]:.4f}, Prec: {avg[2]:.4f}, Rec: {avg[3]:.4f}, F1: {avg[4]:.4f}")
    training_info["average_results"] = {
        "avg_val_loss": avg[0],
        "avg_accuracy": avg[1],
        "avg_precision": avg[2],
        "avg_recall": avg[3],
        "avg_f1": avg[4]
    }
    save_training_info(training_info, args.save_dir, 'training_info.json')
    save_training_info(best_model_results, args.save_dir, 'best_model_results.json')

    if writer:
        writer.close()
    print(f"Best global model: {best_global_model_path}, loss {best_model_performance:.4f}")


if __name__ == '__main__':
    main()
