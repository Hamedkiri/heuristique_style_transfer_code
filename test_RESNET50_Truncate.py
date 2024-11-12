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
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from PIL import Image, ImageTk
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import hdbscan
from matplotlib.path import Path
from pykalman import KalmanFilter

class TruncatedMoCoV3(nn.Module):
    def __init__(self, base_encoder, truncate_after_layer, dim=256, device='cuda'):
        super(TruncatedMoCoV3, self).__init__()
        self.device = device
        self.truncated_encoder = nn.Sequential(
            *list(base_encoder.children())[:truncate_after_layer]
        ).to(self.device)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        dummy_input = torch.zeros(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            output_shape = self.truncated_encoder(dummy_input)
            output_shape = self.pool(output_shape)
            output_shape = self.flatten(output_shape)
            num_features = output_shape.shape[1]

        self.fc = nn.Linear(num_features, dim).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.truncated_encoder(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def load_best_model(classifier, moco_model, filepath):
    checkpoint = torch.load(filepath)
    moco_state_dict = checkpoint.get('moco_model_state_dict', checkpoint)
    classifier_state_dict = checkpoint.get('classifier_state_dict', checkpoint)

    moco_model.load_state_dict(moco_state_dict, strict=False)
    classifier.load_state_dict(classifier_state_dict, strict=False)
    moco_model.cuda()
    classifier.cuda()

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
    args = parser.parse_args()

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
            test_accuracy, test_loss, precision, recall, f1, cm = test(classifier, moco_model, test_loader, nn.CrossEntropyLoss().cuda(), writer if args.tensorboard else None, class_names, args.save_dir, transform, args.prob_threshold, args.visualize_gradcam, args.save_gradcam_images, args.measure_time)
            print(f"Précision du test: {test_accuracy}, Perte du test: {test_loss}, Précision: {precision}, Rappel: {recall}, Score F1: {f1}")
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
        run_camera(classifier, moco_model, transform, class_names, args.save_camera_video, args.save_dir, args.prob_threshold, args.measure_time, args.kalman_filter)

    if args.tensorboard:
        writer.close()

def run_camera(classifier, moco_model, transform, class_names, save_video, save_dir, prob_threshold, measure_time, kalman_filter):
    moco_model.eval()
    classifier.eval()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return

    if save_video:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        video_save_path = os.path.join(save_dir, "camera_output.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_save_path, fourcc, 20.0, (640, 480))

    times = []

    if kalman_filter:
        # Initialisation du filtre de Kalman
        kf = KalmanFilter(initial_state_mean=np.zeros(len(class_names)),
                          initial_state_covariance=np.eye(len(class_names)),
                          n_dim_obs=len(class_names))
        state_means = np.zeros(len(class_names))
        state_covariance = np.eye(len(class_names))

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur: Impossible de lire l'image de la caméra")
                break

            start_time = time.time()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            img_tensor = transform(pil_img).unsqueeze(0).cuda()

            features = moco_model(img_tensor)
            output = classifier(features)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]

            if kalman_filter:
                # Ajustement de la covariance de l'observation en fonction de l'incertitude du modèle
                observation_covariance = np.diag(1.0 - probabilities)
                state_means, state_covariance = kf.filter_update(
                    filtered_state_mean=state_means,
                    filtered_state_covariance=state_covariance,
                    observation=probabilities,
                    observation_covariance=observation_covariance
                )
                kalman_probabilities = state_means

                # Prédictions avec et sans Kalman
                pred_label_no_kalman = np.argmax(probabilities)
                prob_no_kalman = probabilities[pred_label_no_kalman]
                pred_label_with_kalman = np.argmax(kalman_probabilities)
                prob_with_kalman = kalman_probabilities[pred_label_with_kalman]

                label_no_kalman = class_names[pred_label_no_kalman]
                label_with_kalman = class_names[pred_label_with_kalman]

                text = (f"Original: {label_no_kalman}, Prob: {prob_no_kalman:.4f} | "
                        f"Kalman: {label_with_kalman}, Prob: {prob_with_kalman:.4f}")
            else:
                pred_label_no_kalman = np.argmax(probabilities)
                prob_no_kalman = probabilities[pred_label_no_kalman]
                label_no_kalman = class_names[pred_label_no_kalman]

                text = f"Label: {label_no_kalman}, Prob: {prob_no_kalman:.4f}"

            end_time = time.time()
            times.append(end_time - start_time)

            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Camera', frame)

            if save_video:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if measure_time:
        with open(os.path.join(save_dir, "times_camera.json"), "w") as f:
            json.dump(times, f, indent=4)
        print(f"Temps moyen de traitement par image: {np.mean(times)} secondes")
        print(f"Temps total de traitement: {np.sum(times)} secondes")

    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()

def test(classifier, moco_model, loader, criterion, writer, class_names, save_dir, transform, prob_threshold, visualize_gradcam, save_gradcam_images, measure_time):
    moco_model.eval()
    classifier.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds = []
    all_targets = []
    all_features = []
    times = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for class_name in class_names:
        class_dir = os.path.join(save_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            if isinstance(loader.dataset, Subset):
                original_images = [
                    loader.dataset.dataset.imgs[idx][0]
                    for idx in loader.dataset.indices[i * loader.batch_size:(i + 1) * loader.batch_size]
                ]
            else:
                original_images = [
                    loader.dataset.imgs[idx][0]
                    for idx in range(i * loader.batch_size, (i + 1) * loader.batch_size)
                ]

            start_time = time.time()

            data, target = data.cuda(), target.cuda()
            features = moco_model(data)
            output = classifier(features)
            loss = criterion(output, target)
            total_loss += loss.item()
            probabilities = torch.nn.functional.softmax(output, dim=1)
            max_probs, predicted = torch.max(probabilities, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_features.append(features.cpu().numpy())

            end_time = time.time()
            times.append(end_time - start_time)

            for j in range(data.size(0)):
                img_path = original_images[j]
                img = Image.open(img_path)
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                label = class_names[target[j].cpu().item()]
                pred = class_names[predicted[j].cpu().item()] if max_probs[j].item() >= prob_threshold else "Unknown"
                prob = probabilities[j][predicted[j]].cpu().item()

                text = f"Label: {label}, Pred: {pred}, Prob: {prob:.4f}"
                cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                class_dir = os.path.join(save_dir, label)
                img_save_path = os.path.join(class_dir, f"{i}_{j}.png")
                cv2.imwrite(img_save_path, img)

                if writer:
                    img_tensor = torch.tensor(img).permute(2, 0, 1)
                    writer.add_image(f'Test/Images/{i}_{j}', img_tensor, global_step=i)
                    writer.add_text(f'Test/Prédictions/{i}_{j}', text, global_step=i)

                if visualize_gradcam:
                    target_layer = moco_model.truncated_encoder[-1]
                    grad_cam = GradCAM(model=moco_model, target_layers=[target_layer])
                    input_img = data[j]
                    label_idx = target[j].item()
                    with torch.set_grad_enabled(True):
                        visualization, grayscale_cam = generate_heatmap(grad_cam, input_img, label_idx)

                    orig_img = Image.open(img_path)
                    orig_img = orig_img.resize((224, 224))
                    orig_img = np.array(orig_img)

                    grayscale_cam_img = np.uint8(255 * grayscale_cam)
                    grayscale_cam_img_rgb = cv2.cvtColor(grayscale_cam_img, cv2.COLOR_GRAY2RGB)

                    combined_image = np.hstack((orig_img, visualization, grayscale_cam_img_rgb))
                    if save_gradcam_images:
                        gradcam_class_dir = os.path.join(save_dir, f"GradCAM_{label}")
                        if not os.path.exists(gradcam_class_dir):
                            os.makedirs(gradcam_class_dir)
                        combined_image_save_path = os.path.join(gradcam_class_dir, f"gradcam_{i}_{j}.png")
                        cv2.imwrite(combined_image_save_path, combined_image)

                    if writer:
                        combined_image_tensor = torch.tensor(combined_image).permute(2, 0, 1)
                        writer.add_image(f'GradCAM/Images/{i}_{j}', combined_image_tensor, global_step=i)

    accuracy = 100 * correct / total
    average_loss = total_loss / len(loader)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    cm = confusion_matrix(all_targets, all_preds)

    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Précision du test: {accuracy}\n")
        f.write(f"Perte du test: {average_loss}\n")
        f.write(f"Précision: {precision}\n")
        f.write(f"Rappel: {recall}\n")
        f.write(f"Score F1: {f1}\n")
        f.write(f"Matrice de confusion:\n{cm}\n")

    if measure_time:
        with open(os.path.join(save_dir, "times_test.json"), "w") as f:
            json.dump(times, f, indent=4)
        print(f"Temps moyen de traitement par image: {np.mean(times)} secondes")
        print(f"Temps total de traitement: {np.sum(times)} secondes")

    return accuracy, f"{average_loss:.4f}", precision, recall, f1, cm

def compute_embeddings_with_paths(model, loader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    img_paths = []

    original_dataset = loader.dataset.dataset if isinstance(loader.dataset, Subset) else loader.dataset

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            embeddings = model(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

            if isinstance(loader.dataset, Subset):
                img_paths.extend([
                    original_dataset.imgs[idx][0]
                    for idx in loader.dataset.indices[i * loader.batch_size:(i + 1) * loader.batch_size]
                ])
            else:
                img_paths.extend([
                    original_dataset.imgs[idx][0]
                    for idx in range(i * loader.batch_size, (i + 1) * loader.batch_size)
                ])

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    return all_embeddings, all_labels, img_paths

def perform_tsne(embeddings, labels, class_names, colors=None, results_dir='results'):
    print("Réalisation de t-SNE...")
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    if colors and len(colors) >= num_classes:
        color_map = {label.item(): colors[i] for i, label in enumerate(unique_labels)}
    else:
        color_map = {label.item(): plt.cm.tab20(i / num_classes) for i, label in enumerate(unique_labels)}

    for label in unique_labels:
        indices = labels == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                    label=class_names[label.item()], color=color_map[int(label)])
    plt.legend()
    tsne_plot_path = os.path.join(results_dir, 'tsne_plot.png')
    plt.savefig(tsne_plot_path)
    plt.show()
    print(f"t-SNE plot saved to '{tsne_plot_path}'")

def plot_tsne_interactive(embeddings, labels, classes, img_paths, colors=None, num_clusters=None, save_dir='results'):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
    tsne_results = tsne.fit_transform(embeddings_flat)

    # Création de la fenêtre Tkinter
    root = tk.Tk()
    root.title("Interactive t-SNE with Images")

    # Création de la figure matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    if colors and len(colors) >= num_classes:
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    else:
        color_palette = plt.cm.get_cmap("tab20", num_classes)
        color_map = {label: color_palette(label) for label in unique_labels}

    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[color_map[int(label)] for label in labels], picker=True)
    legend = ax.legend(handles=scatter.legend_elements()[0], labels=[classes[int(label)] for label in unique_labels])

    def onpick(event):
        ind = event.ind[0]
        img_path = img_paths[ind]
        img = Image.open(img_path)
        img = img.resize((400, 400), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
        label_text.set(f"Label: {classes[int(labels[ind])]}")

    def on_key(event):
        if event.key == 'z':
            zoom(event.xdata, event.ydata, 0.9)
        elif event.key == 'a':
            zoom(event.xdata, event.ydata, 1.1)

    def zoom(x, y, factor):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim([x - (x - xlim[0]) * factor, x + (xlim[1] - x) * factor])
        ax.set_ylim([y - (y - ylim[0]) * factor, y + (ylim[1] - y) * factor])
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', onpick)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Ajout de la figure matplotlib à la fenêtre Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, sticky='nsew')

    # Ajout du label pour afficher l'image sélectionnée et le texte du label
    img_label = tk.Label(root)
    img_label.grid(row=0, column=1, sticky='nsew')
    label_text = tk.StringVar()
    label_label = tk.Label(root, textvariable=label_text)
    label_label.grid(row=1, column=1, sticky='nsew')
    inside_points_label = tk.StringVar()
    inside_points_count_label = tk.Label(root, textvariable=inside_points_label)
    inside_points_count_label.grid(row=2, column=0, columnspan=2, sticky='nsew')

    if num_clusters is not None:
        cluster_label_text = tk.StringVar(value=f"Number of clusters: {num_clusters}")
        cluster_label = tk.Label(root, textvariable=cluster_label_text)
        cluster_label.grid(row=3, column=0, columnspan=2, sticky='nsew')

    # Configuration des poids des colonnes et des lignes
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_rowconfigure(2, weight=1)
    root.grid_rowconfigure(3, weight=1)

    global polygon_selector  # Declare the global variable
    polygon = []
    polygon_selector = None
    polygon_cleared = True  # Variable témoin pour savoir si le polygone a été effacé

    def enable_polygon_selector(event):
        global polygon_selector
        global polygon_cleared
        if event.button == 3:  # Right-click
            if polygon_selector is None or polygon_cleared:
                polygon_selector = PolygonSelector(ax, onselect=onselect, useblit=True)
                polygon_cleared = False
                print("Polygon selector enabled.")

    def onselect(verts):
        polygon.clear()
        polygon.extend(verts)
        print("Polygon vertices:", verts)

    def analyze_polygon():
        if len(polygon) < 3:
            print("Polygon not closed. Select at least 3 points.")
            return

        inside_points = []
        outside_points = []
        polygon_path = Path(polygon)  # Use Path from matplotlib.path

        for i, (x, y) in enumerate(tsne_results):
            point = (x, y)
            if polygon_path.contains_point(point):
                inside_points.append({"path": img_paths[i], "class": classes[int(labels[i])], "position": point})
            else:
                outside_points.append({"path": img_paths[i], "class": classes[int(labels[i])], "position": point})

        save_json(inside_points, os.path.join(save_dir, "inside_polygon.json"))
        save_json(outside_points, os.path.join(save_dir, "outside_polygon.json"))

        inside_points_label.set(f"Points inside polygon: {len(inside_points)}")
        # Display points inside the polygon at the top
        update_top_display(inside_points)

    def update_top_display(inside_points):
        dropdown_values = [f"{point['path']} ({point['class']})" for point in inside_points]
        dropdown['values'] = dropdown_values
        if dropdown_values:
            dropdown.current(0)  # Select the first item

    def save_json(data, filename):
        with open(filename, "w") as f:
            json.dump(data, f, default=convert_to_serializable)

    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return obj

    def clear_polygon():
        global polygon_selector
        global polygon_cleared
        polygon.clear()
        if polygon_selector:
            polygon_selector.disconnect_events()  # Completely disconnect the polygon selector events
            polygon_selector.set_visible(False)  # Hide the selector's line
            del polygon_selector  # Delete the polygon selector
            polygon_selector = None
        while ax.patches:
            ax.patches.pop().remove()  # Remove all patches from the axis
        fig.canvas.draw()
        inside_points_label.set("")  # Clear the inside points label
        label_text.set("")  # Clear the top display text
        polygon_cleared = True  # Set the variable témoin to indicate that the polygon has been cleared

    fig.canvas.mpl_connect('button_press_event', enable_polygon_selector)

    close_button = tk.Button(root, text="Close Polygon", command=analyze_polygon)
    close_button.grid(row=4, column=0, sticky='ew')

    clear_button = tk.Button(root, text="Clear Polygon", command=clear_polygon)
    clear_button.grid(row=4, column=1, sticky='ew')

    # Add a label at the top to display points inside the polygon
    top_display_text = tk.StringVar()
    top_display_label = tk.Label(root, textvariable=top_display_text, justify='left')
    top_display_label.grid(row=5, column=0, columnspan=2, sticky='nsew')

    # Add a dropdown list to display points inside the polygon
    dropdown = ttk.Combobox(root)
    dropdown.grid(row=6, column=0, columnspan=2, sticky='ew')

    root.mainloop()

def generate_heatmap(grad_cam, img, target_category):
    grad_cam.model.eval()
    input_tensor = img.unsqueeze(0).requires_grad_(True).cuda()
    targets = [ClassifierOutputTarget(target_category)]
    grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize to [0, 1]
    visualization = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam

if __name__ == '__main__':
    main()