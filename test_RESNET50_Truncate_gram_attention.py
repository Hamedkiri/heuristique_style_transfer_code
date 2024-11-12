import argparse
import os
import json
import torch

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import precision_score, recall_score, f1_score

import random
from Models.Models_RESNET50_TRUNCATE_GRAM_with_Attention import  TruncatedResNet50_for_test
from functions.functions_RESNET50_Truncate_Gram_Attention import run_camera, style_transfer, load_hyperparameters, load_model_weights, evaluate_model_test, perform_tsne, plot_tsne_interactive



def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ResNet50 model with options for t-SNE, classification, camera or style transfer')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model weights')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the hyperparameters configuration file')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of images to sample for testing')
    parser.add_argument('--mode', type=str,
                        choices=['tsne', 'tsne_interactive', 'classification', 'camera', 'style_transfer'],
                        required=True,
                        help='Mode: tsne, tsne_interactive, classification, camera or style_transfer')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--colors', nargs='+', help='List of colors for t-SNE')
    parser.add_argument('--classes', nargs='+', help='Classes used for prediction in camera mode')
    parser.add_argument('--save_camera_video', action='store_true', help='Save camera video')
    parser.add_argument('--prob_threshold', default=0.5, type=float,
                        help='Probability threshold for classifying as unknown')
    parser.add_argument('--measure_time', action='store_true',
                        help='Measure and record the average processing time per image')
    parser.add_argument('--layers', type=int, default=4,
                        help='Number of convolutional layers to use for style transfer')
    parser.add_argument('--threshold', default=1e-7, type=float, help='Error threshold for style transfer')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Learning rate for style transfer')
    parser.add_argument('--num_iterations', default=500, type=int, help='Number of iterations for style transfer')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    hyperparams = load_hyperparameters(args.config_path)
    if not hyperparams:
        raise ValueError("Hyperparameters configuration is required")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root=os.path.join(args.data, "test"), transform=transform)
    num_classes = len(dataset.classes)

    if args.num_samples:
        dataset = Subset(dataset, random.sample(range(len(dataset)), args.num_samples))

    data_loader = DataLoader(dataset, batch_size=hyperparams.get('batch_size', 32), shuffle=False, num_workers=4)

    base_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    model = TruncatedResNet50_for_test (base_encoder, truncate_after_layer=hyperparams.get('truncate_layer', 7),
                              num_classes=num_classes,
                              gram_matrix_size=hyperparams.get('gram_matrix_size', 32),
                              device=device).to(device)

    load_model_weights(model, args.model_path)

    if args.mode == 'classification':
        embeddings, preds, labels, img_paths = evaluate_model_test(model, data_loader, device)
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')

        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        results_path = os.path.join(args.save_dir, 'classification_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Classification results saved to {results_path}")

    elif args.mode == 'tsne':
        embeddings, _, labels, _ = evaluate_model_test(model, data_loader, device)
        tsne_path = os.path.join(args.save_dir, 'tsne_visualization.png')
        perform_tsne(embeddings, labels, tsne_path, colors=args.colors)

    elif args.mode == 'tsne_interactive':
        embeddings, _, labels, img_paths = evaluate_model_test(model, data_loader, device)
        plot_tsne_interactive(embeddings, labels,
                              dataset.dataset.classes if isinstance(dataset, Subset) else dataset.classes, img_paths,
                              dataset, colors=args.colors)

    elif args.mode == 'camera':
        if args.classes is None:
            raise ValueError("You must specify classes with the --classes option for camera mode.")
        run_camera(model, transform, args.classes, args.save_camera_video, args.save_dir, args.prob_threshold,
                   args.measure_time)

    elif args.mode == 'style_transfer':
        style_transfer(model, data_loader, device, save_dir=args.save_dir, layers=args.layers, threshold=args.threshold,
                       num_iterations=args.num_iterations, learning_rate=args.learning_rate)


if __name__ == '__main__':
    main()