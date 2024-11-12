import argparse
import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import  confusion_matrix
import numpy as np

from Models.Models_Multi_PatchGAN import MultiScaleDiscriminator_test
from functions.functions_Multi_PatchGAN import convert_to_serializable, run_camera, evaluate_model_test, plot_tsne, plot_tsne_interactive, plot_confusion_matrix, style_transfer_patches, evaluate_classification




def main():
    """
    Script de test du modèle avec options et fonctionnalités supplémentaires.
    """
    parser = argparse.ArgumentParser(
        description='Évaluation des performances du modèle sur des données de test.')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers les poids du modèle entraîné')
    parser.add_argument('--config_path', type=str, required=True, help='Chemin vers le fichier de configuration du modèle')
    parser.add_argument('--data', type=str, required=True, help='Chemin vers le répertoire racine du jeu de données')
    parser.add_argument('--num_samples', default=None, type=int, help='Nombre d\'images à tester')
    parser.add_argument('--save_dir', default='results', type=str, help='Répertoire pour sauvegarder les résultats')
    parser.add_argument('--measure_time', action='store_true', help='Mesurer le temps pris pour chaque image')
    parser.add_argument('--mode', type=str, choices=['tsne', 'tsne_interactive', 'camera', 'style_transfer', 'classification'],
                        required=True,
                        help='Mode: tsne, tsne_interactive, camera, style_transfer, or classification')
    parser.add_argument('--pooling_type', type=str, default='avg', choices=['avg', 'max'],
                        help='Choisir le type de pooling (avg ou max)')
    parser.add_argument('--save_camera_video', action='store_true', help='Sauvegarder la sortie vidéo de la caméra')
    parser.add_argument('--prob_threshold', default=0.5, type=float, help='Seuil de probabilité pour la classification')
    parser.add_argument('--classes', nargs='+', default=None, help='Classes utilisées pour le mode caméra')
    parser.add_argument('--colors', nargs='+', help='Liste de couleurs pour la visualisation t-SNE')
    parser.add_argument('--layers', type=int, default=5,
                        help='Nombre de couches convolutionnelles à utiliser pour le style transfer')
    parser.add_argument('--threshold', default=1e-4, type=float, help='Seuil d\'erreur pour le style transfer')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Taux d\'apprentissage pour le style transfer')
    parser.add_argument('--num_iterations', default=500, type=int, help='Nombre d\'itérations pour le style transfer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Chargement des hyperparamètres du modèle
    with open(args.config_path, 'r') as f:
        config = json.load(f)

    hidden_dims = config.get('hidden_dims', [])
    batch_size = config.get('batch_size', 32)
    lr = config.get('lr', 0.01)
    patch_sizes = config.get('patch_sizes', {'small': 10, 'medium': 70, 'large': 150})
    num_classes = config.get('num_classes', 10)
    gram_matrix_dim = config.get('gram_matrix_dim', 64)
    lambda_reg = config.get('lambda_reg', 1e-3)

    # Préparation des transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialisation du modèle
    model = MultiScaleDiscriminator_test(
        input_nc=3,
        ndf=64,
        norm='batch',
        tensorboard_logdir=None,
        global_step=None,
        patch_sizes=patch_sizes,
        num_classes=num_classes,
        gram_matrix_dim=gram_matrix_dim,
        pooling_type=args.pooling_type
    ).to(device)

    # Chargement des poids du modèle
    model_state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(model_state)

    if args.mode == 'camera':
        if args.classes is None:
            raise ValueError("Vous devez spécifier les classes avec l'option --classes en mode caméra.")
        run_camera(model, transform, args.classes, args.save_camera_video, args.save_dir, args.prob_threshold,
                   args.measure_time)

    elif args.mode in ['tsne', 'tsne_interactive']:
        test_dataset = datasets.ImageFolder(root=os.path.join(args.data, 'test'), transform=transform)

        if args.num_samples:
            indices = list(range(len(test_dataset)))
            np.random.shuffle(indices)
            indices = indices[:args.num_samples]
            test_dataset = Subset(test_dataset, indices)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        criterion = nn.CrossEntropyLoss().to(device)
        _, _, _, embeddings, labels = evaluate_model_test(model, test_loader, criterion)

        # Gestion des chemins d'images
        if isinstance(test_dataset, Subset):
            img_paths = [test_dataset.dataset.samples[i][0] for i in test_dataset.indices]
        else:
            img_paths = [sample[0] for sample in test_dataset.samples]

        if args.mode == 'tsne':
            plot_tsne(embeddings, labels, test_dataset.dataset.classes if isinstance(test_dataset, Subset) else test_dataset.classes, colors=args.colors, save_dir=args.save_dir)
        else:
            plot_tsne_interactive(embeddings, labels, test_dataset.dataset.classes if isinstance(test_dataset, Subset) else test_dataset.classes, img_paths, test_dataset,
                                  colors=args.colors)

    elif args.mode == 'style_transfer':
        test_dataset = datasets.ImageFolder(root=os.path.join(args.data, 'test'), transform=transform)

        if args.num_samples:
            indices = list(range(len(test_dataset)))
            np.random.shuffle(indices)
            indices = indices[:args.num_samples]
            test_dataset = Subset(test_dataset, indices)

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
        style_transfer_patches(model, test_loader, device, save_dir=args.save_dir, layers=args.layers,
                               threshold=args.threshold,
                               num_iterations=args.num_iterations, learning_rate=args.learning_rate)

    elif args.mode == 'classification':
        test_dataset = datasets.ImageFolder(root=os.path.join(args.data, 'test'), transform=transform)

        if args.num_samples:
            indices = list(range(len(test_dataset)))
            np.random.shuffle(indices)
            indices = indices[:args.num_samples]
            test_dataset = Subset(test_dataset, indices)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        precision, recall, f1, all_preds, all_labels = evaluate_classification(model, test_loader, device)

        # Calcul de l'accuracy
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

        # Création du répertoire de sortie si nécessaire
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # Génération de la matrice de confusion
        cm = confusion_matrix(all_labels, all_preds)
        classes = test_dataset.dataset.classes if isinstance(test_dataset, Subset) else test_dataset.classes
        plot_confusion_matrix(cm, classes, args.save_dir)

        # Sauvegarder les résultats de classification
        results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "predictions": [int(p) for p in all_preds],
            "labels": [int(l) for l in all_labels]
        }

        results_path = os.path.join(args.save_dir, "classification_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=convert_to_serializable)

        print(f"Résultats de classification sauvegardés dans {results_path}")



if __name__ == '__main__':
    main()
