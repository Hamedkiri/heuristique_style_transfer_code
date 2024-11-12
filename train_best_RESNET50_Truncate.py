import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import numpy as np
from pytorch_grad_cam import GradCAM
from functions.RESNET50_Truncate import evaluate_model_best, train_model_best, load_hyperparameters, load_training_info, load_best_model, save_model_and_hyperparameters, generate_heatmap, denormalize, show_images_side_by_side, save_training_info
from Models.RESNET50_TRUNCATE import TruncatedMoCoV3_best, Classifier





def main():
    parser = argparse.ArgumentParser(description='Fine-tuning MoCo v3 for Weather Classification')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the best pre-trained MoCo v3 model')
    parser.add_argument('--config_path', type=str, required=True, help='Path to the best hyperparameters configuration')
    parser.add_argument('--epochs', default=25, type=int, help='Number of epochs to train')
    parser.add_argument('--save_dir', default='saved_models', type=str, help='Directory to save trained models')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--k_folds', default=5, type=int, help='Number of folds for cross-validation')
    parser.add_argument('--visualize_gradcam', action='store_true', help='Visualize Grad-CAM and original image')
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

    hyperparameters = load_hyperparameters(args.config_path)
    batch_size = hyperparameters['batch_size']
    lr = hyperparameters['lr']
    truncate_layer = hyperparameters['truncate_layer']

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard')) if args.tensorboard else None

    kf = KFold(n_splits=args.k_folds, shuffle=True)

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
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        base_encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
        moco_model = TruncatedMoCoV3_best(base_encoder, truncate_layer, dim=256, device=device).to(device)
        classifier = Classifier(input_dim=256, num_classes=len(dataset.classes)).to(device)

        load_best_model(classifier, moco_model, args.model_path)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(moco_model.parameters()) + list(classifier.parameters())), lr=lr, momentum=0.9)

        moco_model, classifier = train_model_best(moco_model, classifier, train_loader, criterion, optimizer, num_epochs=args.epochs, writer=writer, fold=fold)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model_best(moco_model, classifier, val_loader, criterion, writer=writer, fold=fold)
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

        if args.visualize_gradcam:
            # Generate and save Grad-CAM heatmaps for each fold
            target_layer = moco_model.truncated_encoder[-1]
            grad_cam = GradCAM(model=moco_model, target_layers=[target_layer])

            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                for j in range(inputs.size(0)):
                    input_img = inputs[j]
                    label = labels[j].item()
                    visualization, grayscale_cam = generate_heatmap(grad_cam, input_img, label)

                    # Denormalize the original image for display
                    orig_img = denormalize(input_img.cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    orig_img = orig_img.permute(1, 2, 0).numpy()
                    orig_img = (orig_img * 255).astype(np.uint8)

                    grayscale_cam_img = np.uint8(255 * grayscale_cam)

                    combined_image = show_images_side_by_side(orig_img, visualization, grayscale_cam_img)

                    writer.add_image(f"GradCAM/Fold_{fold}_Val_{i}_{j}", combined_image, dataformats='HWC')

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
