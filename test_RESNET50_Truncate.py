import argparse
import os
import json
import cv2
import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

import random

import hdbscan
from Models.Models_RESNET50_TRUNCATE import TruncatedMoCoV3, Classifier, CombinedModel
from functions.functions_RESNET50_Truncate import load_best_model, compute_embeddings_with_paths, test, perform_tsne, run_camera, plot_tsne_interactive, list_available_cameras


def main():
    parser = argparse.ArgumentParser(description='MoCo pour une tâche de classification spécifique')
    parser.add_argument('--data', type=str, help='Chemin vers le dataset')
    parser.add_argument('--config_path', type=str, required=True, help='Chemin vers le fichier JSON avec les meilleurs hyperparamètres')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le modèle pré-entraîné')
    parser.add_argument('--batch_size', default=32, type=int, help='Taille de lot pour les tests')
    parser.add_argument('--num_samples', type=int, default=None, help='Nombre d\'échantillons à tester')
    parser.add_argument('--save_dir', default='results', type=str, help='Répertoire pour enregistrer les résultats')
    parser.add_argument('--tensorboard', action='store_true', help='Activer la journalisation TensorBoard')
    parser.add_argument('--save_camera_video', action='store_true', help='Enregistrer les vidéos de la caméra')
    parser.add_argument('--prob_threshold', default=0.5, type=float, help='Seuil de probabilité pour considérer une classe comme inconnue')
    parser.add_argument('--visualize_gradcam', action='store_true', help='Visualiser Grad-CAM et l\'image avant transformation')
    parser.add_argument('--save_gradcam_images', action='store_true', help='Enregistrer les images Grad-CAM')
    parser.add_argument('--measure_time', action='store_true', help='Mesurer et enregistrer le temps moyen de traitement par image')
    parser.add_argument('--mode', choices=['classifier', 'tsne', 'tsne_interactive', 'camera', 'inference', 'clustering'], default='classifier', help='Mode d\'opération')
    parser.add_argument('--colors', nargs='+', default=None, metavar='COLORS', help='Liste des couleurs pour t-SNE ou clustering')
    parser.add_argument('--clustering_class', type=str, help='Nom de la classe pour le clustering HDBSCAN')
    parser.add_argument('--min_cluster_size', type=int, nargs='+', default=[10, 15, 20], metavar='MIN_CLUSTER_SIZE', help='Liste des valeurs min_cluster_size pour HDBSCAN')
    parser.add_argument('--min_samples', type=int, nargs='+', default=[5, 10], metavar='MIN_SAMPLES', help='Liste des valeurs min_samples pour HDBSCAN')
    parser.add_argument('--kalman_filter', action='store_true', help='Appliquer un filtre de Kalman pour lisser les prédictions de la caméra')
    parser.add_argument('--save_test_images', action='store_true', help='Sauvegarder les images d\'évaluation et de test avec prédictions et probabilités')  # Nouvelle option
    parser.add_argument('--test_data', type=str, help="Chemin vers les données de test d'origine")
    parser.add_argument('--list_cameras', action='store_true', help='Lister les caméras disponibles')
    parser.add_argument('--camera_index', type=int, default=0, help='Index de la caméra à utiliser')
    parser.add_argument('--colormap', type=str, default='hot', help='Colormap pour les visualisations Grad-CAM (par exemple, hot, autumn, afmhot)')
    parser.add_argument('--compute_auc', action='store_true', help='Calculer le score AUC pour le modèle')  # Nouvel argument


    args = parser.parse_args()

    colormap_dict = {
        'autumn': cv2.COLORMAP_AUTUMN,
        'bone': cv2.COLORMAP_BONE,
        'jet': cv2.COLORMAP_JET,
        'winter': cv2.COLORMAP_WINTER,
        'rainbow': cv2.COLORMAP_RAINBOW,
        'ocean': cv2.COLORMAP_OCEAN,
        'summer': cv2.COLORMAP_SUMMER,
        'spring': cv2.COLORMAP_SPRING,
        'cool': cv2.COLORMAP_COOL,
        'hsv': cv2.COLORMAP_HSV,
        'pink': cv2.COLORMAP_PINK,
        'hot': cv2.COLORMAP_HOT,
        'inferno': cv2.COLORMAP_INFERNO,
        'magma': cv2.COLORMAP_MAGMA,
        'plasma': cv2.COLORMAP_PLASMA,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'cividis': cv2.COLORMAP_CIVIDIS,
        'turbo': cv2.COLORMAP_TURBO,
    }
    chosen_colormap = colormap_dict.get(args.colormap.lower(), 'hot')

    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'TensorBoard'))

    with open(args.config_path, 'r') as f:
        best_config = json.load(f)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if 'classes' in best_config:
        class_names = best_config['classes']
    elif args.data:
        dataset = datasets.ImageFolder(root=os.path.join(args.data, 'test'), transform=transform)
        class_names = dataset.classes
    else:
        class_names = ["fog", "rain", "snow", "sun"]

    truncate_layer = best_config['truncate_layer']
    moco_base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    moco_model = TruncatedMoCoV3(moco_base, truncate_layer, dim=256, device='cuda')
    classifier = Classifier(input_dim=256, num_classes=len(class_names))
    load_best_model(classifier, moco_model, args.model_path)

    if args.data:
        dataset = datasets.ImageFolder(root=os.path.join(args.data, 'test'), transform=transform)

        if args.num_samples is not None:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[:args.num_samples]
            dataset = Subset(dataset, indices)

        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        if args.mode == 'classifier':
            test_accuracy, test_loss, precision, recall, f1, cm, auc = test(
                classifier,
                moco_model,
                test_loader,
                nn.CrossEntropyLoss().cuda(),
                writer if args.tensorboard else None,
                class_names,
                args.save_dir,
                transform,
                args.prob_threshold,
                args.visualize_gradcam,
                args.save_gradcam_images,
                args.measure_time,
                args.save_test_images,
                colormap=chosen_colormap,
                compute_auc=args.compute_auc
            )
            print(
                f"Précision du test: {test_accuracy}, Perte du test: {test_loss}, Précision: {precision}, Rappel: {recall}, Score F1: {f1}")
            if auc is not None:
                print(f"Score AUC: {auc}")
        elif args.mode in ['tsne', 'tsne_interactive']:
            all_embeddings, all_labels, img_paths = compute_embeddings_with_paths(moco_model, test_loader, 'cuda')
            results = {'embeddings': all_embeddings.tolist(), 'labels': all_labels.tolist()}
            output_path = os.path.join(args.save_dir, os.path.basename(args.model_path).replace('.pth', '_embeddings.json'))
            os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Assurez-vous que le répertoire existe
            with open(output_path, 'w') as f:
                json.dump(results, f)
            if args.mode == 'tsne':
                perform_tsne(all_embeddings, all_labels, class_names, args.colors, args.save_dir)
            else:
                plot_tsne_interactive(all_embeddings, all_labels, class_names, img_paths, args.colors, save_dir=args.save_dir)
        elif args.mode == 'clustering':
            all_embeddings, all_labels, img_paths = compute_embeddings_with_paths(moco_model, test_loader, 'cuda')
            if not args.clustering_class:
                raise ValueError("L'option --clustering_class doit être spécifiée pour le mode clustering")
            class_index = class_names.index(args.clustering_class)
            class_embeddings = all_embeddings[all_labels == class_index]
            class_img_paths = [img_paths[i] for i in range(len(all_labels)) if all_labels[i] == class_index]

            best_num_clusters = 0
            best_cluster_labels = None
            best_params = {}

            for min_cluster_size in args.min_cluster_size:
                for min_samples in args.min_samples:
                    print(f"Testing HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
                    clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(class_embeddings)
                    cluster_labels = clustering.labels_

                    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                    print(f"Number of clusters found: {num_clusters}")

                    if num_clusters > best_num_clusters:
                        best_num_clusters = num_clusters
                        best_cluster_labels = cluster_labels
                        best_params = {'min_cluster_size': min_cluster_size, 'min_samples': min_samples}

            if best_cluster_labels is None:
                raise ValueError("No clusters found with the provided HDBSCAN parameters.")

            cluster_labels = best_cluster_labels

            # Préparer les résultats du clustering
            cluster_info = {}
            unique_labels = set(cluster_labels)
            for label in unique_labels:
                indices = [i for i, lbl in enumerate(cluster_labels) if lbl == label]
                cluster_info[str(label)] = {
                    'num_images': len(indices),
                    'img_paths': [class_img_paths[i] for i in indices]
                }

            clustering_results = {
                'num_clusters': best_num_clusters,
                'clusters': cluster_info,
                'best_params': best_params
            }

            clustering_output_path = os.path.join(args.save_dir, f'{args.clustering_class}_clustering_results.json')
            with open(clustering_output_path, 'w') as f:
                json.dump(clustering_results, f)

            print(f"Clustering results saved in '{clustering_output_path}' with parameters {best_params}")

            plot_tsne_interactive(class_embeddings, cluster_labels, [f'Cluster {i}' for i in range(best_num_clusters)] + ['Noise'], class_img_paths, colors=args.colors, num_clusters=best_num_clusters, save_dir=args.save_dir)


    elif args.mode == 'camera':

        run_camera(classifier, moco_model, transform, class_names, args.save_camera_video, args.save_dir,
                   args.prob_threshold, args.measure_time, args.kalman_filter, args.camera_index)

    if args.tensorboard:
        writer.close()

    if args.list_cameras:
        available_cameras = list_available_cameras()
        print(f"Caméras disponibles : {available_cameras}")
        return
if __name__ == '__main__':
    main()