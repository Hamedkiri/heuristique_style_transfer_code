import argparse
import os
import json
import cv2
import time
import random
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

import hdbscan

from Models.Models_RESNET50_TRUNCATE import TruncatedMoCoV3, Classifier, CombinedModel
from functions.functions_RESNET50_Truncate import (
    load_best_model,
    compute_embeddings_with_paths,
    test,
    perform_tsne,
    run_camera,
    plot_tsne_interactive,
    list_available_cameras
)


VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class InferenceImageDataset(Dataset):
    """
    Dataset pour le mode inference.
    Permet de charger :
      - un dossier contenant des images (récursif)
      - un fichier image unique
    """
    def __init__(self, input_path: str, transform=None):
        self.transform = transform
        self.image_paths = self._gather_images(input_path)

        if len(self.image_paths) == 0:
            raise ValueError(f"Aucune image trouvée dans : {input_path}")

    def _gather_images(self, input_path: str) -> List[str]:
        if os.path.isfile(input_path):
            if input_path.lower().endswith(VALID_IMAGE_EXTENSIONS):
                return [input_path]
            raise ValueError(f"Le fichier fourni n'est pas une image supportée : {input_path}")

        if os.path.isdir(input_path):
            image_paths = []
            for root, _, files in os.walk(input_path):
                for fname in files:
                    if fname.lower().endswith(VALID_IMAGE_EXTENSIONS):
                        image_paths.append(os.path.join(root, fname))
            image_paths.sort()
            return image_paths

        raise ValueError(f"Chemin invalide pour l'inférence : {input_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, img_path


def inference_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    paths = [item[1] for item in batch]
    return images, paths


@torch.no_grad()
def run_inference(
    classifier,
    moco_model,
    inference_loader,
    class_names,
    device,
    save_dir,
    json_name="inference_predictions.json",
    measure_time=False
):
    classifier.eval()
    moco_model.eval()

    os.makedirs(save_dir, exist_ok=True)

    results = []
    total_images = 0
    total_time = 0.0

    softmax = nn.Softmax(dim=1)

    for images, img_paths in inference_loader:
        images = images.to(device, non_blocking=True)

        if measure_time and device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        embeddings = moco_model(images)
        logits = classifier(embeddings)
        probs = softmax(logits)

        if measure_time and device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start_time

        pred_indices = torch.argmax(probs, dim=1)
        pred_scores = probs[torch.arange(probs.size(0)), pred_indices]

        total_images += images.size(0)
        total_time += elapsed

        probs_cpu = probs.detach().cpu().tolist()
        pred_indices_cpu = pred_indices.detach().cpu().tolist()
        pred_scores_cpu = pred_scores.detach().cpu().tolist()

        for path, pred_idx, pred_score, prob_vector in zip(
            img_paths, pred_indices_cpu, pred_scores_cpu, probs_cpu
        ):
            result = {
                "image_path": path,
                "predicted_class_index": int(pred_idx),
                "predicted_class": class_names[pred_idx],
                "predicted_probability": float(pred_score),
                "class_probabilities": {
                    class_names[i]: float(prob_vector[i]) for i in range(len(class_names))
                }
            }
            results.append(result)

    output_json = {
        "num_images": len(results),
        "classes": class_names,
        "predictions": results
    }

    if measure_time and total_images > 0:
        output_json["timing"] = {
            "total_inference_time_seconds": float(total_time),
            "average_time_per_image_seconds": float(total_time / total_images)
        }

    output_path = os.path.join(save_dir, json_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"Résultats d'inférence sauvegardés dans : {output_path}")
    print(f"Nombre d'images traitées : {len(results)}")

    if measure_time and total_images > 0:
        print(f"Temps total : {total_time:.4f} s")
        print(f"Temps moyen par image : {total_time / total_images:.6f} s")

    return output_json


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def resolve_class_names(args, best_config, transform):
    """
    Détermine les noms de classes dans cet ordre :
      1. --classes
      2. classes dans le fichier config
      3. classes détectées depuis args.data/test
      4. fallback par défaut
    """
    if args.classes:
        return args.classes

    if 'classes' in best_config and isinstance(best_config['classes'], list):
        return best_config['classes']

    if args.data:
        test_dir = os.path.join(args.data, 'test')
        if os.path.isdir(test_dir):
            dataset = datasets.ImageFolder(root=test_dir, transform=transform)
            return dataset.classes

    return ["fog", "rain", "snow", "sun"]


def main():
    parser = argparse.ArgumentParser(description='MoCo pour une tâche de classification spécifique')

    parser.add_argument('--data', type=str, help='Chemin vers le dataset')
    parser.add_argument('--config_path', type=str, required=True, help='Chemin vers le fichier JSON avec les meilleurs hyperparamètres')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin vers le modèle pré-entraîné')
    parser.add_argument('--batch_size', default=32, type=int, help='Taille de lot')
    parser.add_argument('--num_samples', type=int, default=None, help='Nombre d’échantillons à tester')
    parser.add_argument('--save_dir', default='results', type=str, help='Répertoire pour enregistrer les résultats')
    parser.add_argument('--tensorboard', action='store_true', help='Activer la journalisation TensorBoard')
    parser.add_argument('--save_camera_video', action='store_true', help='Enregistrer les vidéos de la caméra')
    parser.add_argument('--prob_threshold', default=0.5, type=float, help='Seuil de probabilité pour considérer une classe comme inconnue')
    parser.add_argument('--visualize_gradcam', action='store_true', help='Visualiser Grad-CAM et l’image avant transformation')
    parser.add_argument('--save_gradcam_images', action='store_true', help='Enregistrer les images Grad-CAM')
    parser.add_argument('--measure_time', action='store_true', help='Mesurer et enregistrer le temps moyen de traitement par image')
    parser.add_argument(
        '--mode',
        choices=['classification', 'tsne', 'tsne_interactive', 'camera', 'inference', 'clustering'],
        default='classification',
        help='Mode d’opération'
    )
    parser.add_argument('--colors', nargs='+', default=None, metavar='COLORS', help='Liste des couleurs pour t-SNE ou clustering')
    parser.add_argument('--clustering_class', type=str, help='Nom de la classe pour le clustering HDBSCAN')
    parser.add_argument('--min_cluster_size', type=int, nargs='+', default=[10, 15, 20], metavar='MIN_CLUSTER_SIZE', help='Liste des valeurs min_cluster_size pour HDBSCAN')
    parser.add_argument('--min_samples', type=int, nargs='+', default=[5, 10], metavar='MIN_SAMPLES', help='Liste des valeurs min_samples pour HDBSCAN')
    parser.add_argument('--kalman_filter', action='store_true', help='Appliquer un filtre de Kalman pour lisser les prédictions de la caméra')
    parser.add_argument('--save_test_images', action='store_true', help='Sauvegarder les images d’évaluation et de test avec prédictions et probabilités')
    parser.add_argument('--test_data', type=str, help="Chemin vers les données de test d'origine")
    parser.add_argument('--list_cameras', action='store_true', help='Lister les caméras disponibles')
    parser.add_argument('--camera_index', type=int, default=0, help='Index de la caméra à utiliser')
    parser.add_argument('--colormap', type=str, default='hot', help='Colormap pour les visualisations Grad-CAM')
    parser.add_argument('--compute_auc', action='store_true', help='Calculer le score AUC pour le modèle')
    parser.add_argument('--classes', nargs='+', type=str, help='Liste des classes à utiliser')
    parser.add_argument('--afficher_params', action='store_true', help='Afficher le nombre de paramètres du modèle')

    # Arguments spécifiques au mode inference
    parser.add_argument('--inference_input', type=str, help='Chemin vers un dossier d’images ou une image unique pour le mode inference')
    parser.add_argument('--inference_json_name', type=str, default='inference_predictions.json', help='Nom du fichier JSON de sortie pour le mode inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Nombre de workers pour les DataLoader')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

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
    chosen_colormap = colormap_dict.get(args.colormap.lower(), cv2.COLORMAP_HOT)

    writer = None
    if args.tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'TensorBoard'))

    if args.list_cameras:
        available_cameras = list_available_cameras()
        print(f"Caméras disponibles : {available_cameras}")
        return

    with open(args.config_path, 'r', encoding='utf-8') as f:
        best_config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    transform = build_transform()
    class_names = resolve_class_names(args, best_config, transform)

    truncate_layer = best_config['truncate_layer']
    moco_base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    moco_model = TruncatedMoCoV3(moco_base, truncate_layer, dim=256, device=str(device)).to(device)
    classifier = Classifier(input_dim=256, num_classes=len(class_names)).to(device)

    load_best_model(classifier, moco_model, args.model_path)

    if args.afficher_params:
        total_params = (
            sum(p.numel() for p in moco_model.parameters()) +
            sum(p.numel() for p in classifier.parameters())
        )
        print(f"Nombre total de paramètres du modèle (MoCo + Classifier) : {total_params}")

    if args.mode == 'classification':
        if not args.data:
            raise ValueError("Le chemin du dataset doit être spécifié pour le mode classification")

        dataset = datasets.ImageFolder(root=os.path.join(args.data, 'test'), transform=transform)

        if args.num_samples is not None:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            indices = indices[:args.num_samples]
            dataset = Subset(dataset, indices)

        test_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda")
        )

        test_accuracy, test_loss, precision, recall, f1, cm, auc = test(
            classifier,
            moco_model,
            test_loader,
            nn.CrossEntropyLoss().to(device),
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
            f"Précision du test: {test_accuracy}, "
            f"Perte du test: {test_loss}, "
            f"Précision: {precision}, "
            f"Rappel: {recall}, "
            f"Score F1: {f1}"
        )
        if auc is not None:
            print(f"Score AUC: {auc}")

    elif args.mode in ['tsne', 'tsne_interactive']:
        if not args.data:
            raise ValueError("Le chemin du dataset doit être spécifié pour le mode t-SNE")

        dataset = datasets.ImageFolder(root=os.path.join(args.data, 'test'), transform=transform)

        test_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda")
        )

        all_embeddings, all_labels, img_paths = compute_embeddings_with_paths(
            moco_model, test_loader, str(device)
        )

        results = {
            'embeddings': all_embeddings.tolist(),
            'labels': all_labels.tolist()
        }

        output_path = os.path.join(
            args.save_dir,
            os.path.basename(args.model_path).replace('.pth', '_embeddings.json')
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if args.mode == 'tsne':
            perform_tsne(all_embeddings, all_labels, class_names, args.colors, args.save_dir)
        else:
            plot_tsne_interactive(
                all_embeddings,
                all_labels,
                class_names,
                img_paths,
                args.colors,
                save_dir=args.save_dir
            )

    elif args.mode == 'clustering':
        if not args.data:
            raise ValueError("Le chemin du dataset doit être spécifié pour le mode clustering")

        dataset = datasets.ImageFolder(root=os.path.join(args.data, 'test'), transform=transform)

        test_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda")
        )

        all_embeddings, all_labels, img_paths = compute_embeddings_with_paths(
            moco_model, test_loader, str(device)
        )

        if not args.clustering_class:
            raise ValueError("L'option --clustering_class doit être spécifiée pour le mode clustering")

        if args.clustering_class not in class_names:
            raise ValueError(
                f"Classe '{args.clustering_class}' introuvable dans {class_names}"
            )

        class_index = class_names.index(args.clustering_class)
        class_embeddings = all_embeddings[all_labels == class_index]
        class_img_paths = [img_paths[i] for i in range(len(all_labels)) if all_labels[i] == class_index]

        best_num_clusters = 0
        best_cluster_labels = None
        best_params = {}

        for min_cluster_size in args.min_cluster_size:
            for min_samples in args.min_samples:
                print(f"Testing HDBSCAN with min_cluster_size={min_cluster_size}, min_samples={min_samples}")
                clustering = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples
                ).fit(class_embeddings)

                cluster_labels = clustering.labels_
                num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                print(f"Number of clusters found: {num_clusters}")

                if num_clusters > best_num_clusters:
                    best_num_clusters = num_clusters
                    best_cluster_labels = cluster_labels
                    best_params = {
                        'min_cluster_size': min_cluster_size,
                        'min_samples': min_samples
                    }

        if best_cluster_labels is None:
            raise ValueError("Aucun cluster trouvé avec les paramètres HDBSCAN fournis.")

        cluster_info = {}
        unique_labels = set(best_cluster_labels)

        for label in unique_labels:
            indices = [i for i, lbl in enumerate(best_cluster_labels) if lbl == label]
            cluster_info[str(label)] = {
                'num_images': len(indices),
                'img_paths': [class_img_paths[i] for i in indices]
            }

        clustering_results = {
            'num_clusters': best_num_clusters,
            'clusters': cluster_info,
            'best_params': best_params
        }

        clustering_output_path = os.path.join(
            args.save_dir,
            f'{args.clustering_class}_clustering_results.json'
        )

        with open(clustering_output_path, 'w', encoding='utf-8') as f:
            json.dump(clustering_results, f, indent=2, ensure_ascii=False)

        print(f"Clustering results saved in '{clustering_output_path}' with parameters {best_params}")

        cluster_display_names = [f'Cluster {i}' for i in range(best_num_clusters)] + ['Noise']

        plot_tsne_interactive(
            class_embeddings,
            best_cluster_labels,
            cluster_display_names,
            class_img_paths,
            colors=args.colors,
            num_clusters=best_num_clusters,
            save_dir=args.save_dir
        )

    elif args.mode == 'camera':
        run_camera(
            classifier,
            moco_model,
            transform,
            class_names,
            args.save_camera_video,
            args.save_dir,
            args.prob_threshold,
            args.measure_time,
            args.kalman_filter,
            args.camera_index
        )

    elif args.mode == 'inference':
        if not args.inference_input:
            raise ValueError("Le chemin --inference_input doit être spécifié pour le mode inference")

        inference_dataset = InferenceImageDataset(args.inference_input, transform=transform)
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=inference_collate_fn
        )

        run_inference(
            classifier=classifier,
            moco_model=moco_model,
            inference_loader=inference_loader,
            class_names=class_names,
            device=device,
            save_dir=args.save_dir,
            json_name=args.inference_json_name,
            measure_time=args.measure_time
        )

    else:
        raise ValueError(f"Mode non reconnu: {args.mode}")

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()