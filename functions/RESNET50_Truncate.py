
import cv2
import time

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from tkinter import filedialog, messagebox, ttk
from matplotlib.path import Path
from pykalman import KalmanFilter
import os
import json
import torch

from torchvision import datasets, transforms, models
from torch.utils.data import Subset, Dataset

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, transform_combinations):
        self.transform_combinations = transform_combinations
        self.loader = datasets.folder.default_loader

        if isinstance(original_dataset, Subset):
            self.dataset = original_dataset.dataset
            self.indices = original_dataset.indices
        else:
            self.dataset = original_dataset
            self.indices = range(len(original_dataset))

        self.samples = [self.dataset.samples[i] for i in self.indices]
        self.labels = [sample[1] for sample in self.samples]
        self.total_images = len(self.samples) * len(self.transform_combinations)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx

    def __len__(self):
        return self.total_images

    def __getitem__(self, idx):
        sample_idx = idx // len(self.transform_combinations)
        transform_idx = idx % len(self.transform_combinations)
        path, target = self.samples[sample_idx]
        image = self.loader(path)
        transform = self.transform_combinations[transform_idx]
        if transform is not None:
            image = transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image, target



def train_model(moco_model, classifier, train_loader, criterion, optimizer, num_epochs=25, writer=None, fold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moco_model.to(device)
    classifier.to(device)
    moco_model.train()
    classifier.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            features = moco_model(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            print(f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        if writer:
            writer.add_scalar(f"Fold_{fold}/Train/Loss", epoch_loss, epoch)
    return moco_model, classifier

def evaluate_model(moco_model, classifier, val_loader, criterion, writer=None, fold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moco_model.to(device)
    classifier.to(device)
    moco_model.eval()
    classifier.eval()
    val_loss = 0.0
    corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = moco_model(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    total_loss = val_loss / len(val_loader.dataset)
    accuracy = corrects.double() / len(val_loader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f'Fold {fold}, Validation Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    if writer:
        writer.add_scalar(f"Fold_{fold}/Validation/Loss", total_loss)
        writer.add_scalar(f"Fold_{fold}/Validation/Accuracy", accuracy)
        writer.add_scalar(f"Fold_{fold}/Validation/Precision", precision)
        writer.add_scalar(f"Fold_{fold}/Validation/Recall", recall)
        writer.add_scalar(f"Fold_{fold}/Validation/F1", f1)
    return total_loss, accuracy.item(), precision, recall, f1

def load_hyperparameters(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_model_and_hyperparameters(moco_model, classifier, hyperparameters, save_dir, filename):
    model_path = os.path.join(save_dir, f"{filename}.pth")
    hyperparams_path = os.path.join(save_dir, f"{filename}_hyperparameters.json")

    torch.save({
        'moco_model_state_dict': moco_model.state_dict(),
        'classifier_state_dict': classifier.state_dict()
    }, model_path)
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparameters, f, indent=4)

    print(f"Model saved to {model_path}")
    print(f"Hyperparameters saved to {hyperparams_path}")

def save_training_info(info, save_dir, filename):
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'w') as f:
        json.dump(info, f, indent=4)
    print(f"Training information saved to {file_path}")

def load_training_info(save_dir, filename):
    file_path = os.path.join(save_dir, filename)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            info = json.load(f)
        return info
    return None

def generate_transform_combinations(selected_geom_transforms, selected_color_transforms, base_transform):
    # Gérer les cas où aucune transformation n'est sélectionnée
    if not selected_geom_transforms:
        selected_geom_transforms = [transforms.Lambda(lambda x: x)]
    if not selected_color_transforms:
        selected_color_transforms = [transforms.Lambda(lambda x: x)]

    # Créer toutes les combinaisons possibles
    transform_combinations = []
    for geom_transform in selected_geom_transforms:
        for color_transform in selected_color_transforms:
            transform_combinations.append(transforms.Compose([
                geom_transform,
                color_transform,
                base_transform
            ]))
    return transform_combinations



def load_best_model(classifier, moco_model, filepath):
    checkpoint = torch.load(filepath)
    print("Checkpoint keys:", checkpoint.keys())

    # Vérifier si les clés 'moco_model_state_dict' et 'classifier_state_dict' existent
    if 'moco_model_state_dict' in checkpoint and 'classifier_state_dict' in checkpoint:
        # Le checkpoint contient des state_dict séparés
        moco_state_dict = checkpoint['moco_model_state_dict']
        classifier_state_dict = checkpoint['classifier_state_dict']
    else:
        # Le checkpoint est un state_dict plat, nous devons le séparer
        # Inclure toutes les clés sauf celles commençant par 'classifier.'
        moco_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('classifier.')}
        classifier_state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('classifier.'):
                # Garder les clés telles quelles pour le classifier
                classifier_state_dict[k] = v
            elif k.startswith('fc.'):
                # Inclure également les clés commençant par 'fc.' dans le classifier_state_dict
                classifier_state_dict[k] = v
        print("MoCo state_dict keys:", moco_state_dict.keys())
        print("Classifier state_dict keys:", classifier_state_dict.keys())

    # Vérifier et ajuster les clés pour correspondre aux modèles
    # Pour moco_model, les clés doivent correspondre exactement
    missing_keys_moco = moco_model.load_state_dict(moco_state_dict, strict=False)
    print("Missing keys in moco_model:", missing_keys_moco.missing_keys)

    # Pour classifier, il faut s'assurer que les noms de couches correspondent
    # Ajuster les clés si nécessaire
    adjusted_classifier_state_dict = {}
    for k, v in classifier_state_dict.items():
        if k.startswith('classifier.'):
            # Si le modèle 'Classifier' a une couche nommée 'fc', renommer
            new_k = k.replace('classifier.', 'fc.')
            adjusted_classifier_state_dict[new_k] = v
        else:
            adjusted_classifier_state_dict[k] = v
    missing_keys_classifier = classifier.load_state_dict(adjusted_classifier_state_dict, strict=False)
    print("Missing keys in classifier:", missing_keys_classifier.missing_keys)

    moco_model.cuda()
    classifier.cuda()


import tkinter as tk


# ... (autres importations)

def list_available_cameras(max_cameras=5):
    available_indices = []
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            available_indices.append(index)
            cap.release()
    return available_indices


from screeninfo import get_monitors

fullscreen = False  # Démarrer en plein écran


def on_mouse(event, x, y, flags, param):
    global fullscreen
    if event == cv2.EVENT_LBUTTONDOWN:
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


def run_camera(classifier, moco_model, transform, class_names, save_video, save_dir, prob_threshold, measure_time,
               kalman_filter, camera_index):
    global fullscreen
    moco_model.eval()
    classifier.eval()

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra")
        return

    # Utiliser screeninfo pour obtenir les informations sur les écrans
    monitors = get_monitors()

    # Choisir le premier écran (ou l'écran principal par défaut)
    screen = monitors[0]  # Vous pouvez changer l'indice pour choisir un autre écran si nécessaire

    screen_width = screen.width
    screen_height = screen.height

    print(f"Résolution sélectionnée : {screen_width}x{screen_height}")

    # Création de la fenêtre et enregistrement du callback de la souris
    cv2.namedWindow("Camera", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Camera", on_mouse)

    if save_video:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        video_save_path = os.path.join(save_dir, "camera_output.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_save_path, fourcc, 20.0, (screen_width, screen_height))

    times = []

    if kalman_filter:
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

            pred_label_no_kalman = np.argmax(probabilities)
            prob_no_kalman = probabilities[pred_label_no_kalman]
            if prob_no_kalman < prob_threshold:
                label_no_kalman = "Unknown"
            else:
                label_no_kalman = class_names[pred_label_no_kalman]

            text = f"Label: {label_no_kalman}, Prob: {prob_no_kalman:.4f}"

            end_time = time.time()
            times.append(end_time - start_time)

            # Obtenir la taille du cadre de la caméra
            frame_height, frame_width = frame.shape[:2]

            # Calculer le ratio d'aspect du cadre et de l'écran
            frame_aspect_ratio = frame_width / frame_height
            screen_aspect_ratio = screen_width / screen_height

            if screen_aspect_ratio > frame_aspect_ratio:
                # L'écran est plus large, ajuster la hauteur et centrer l'image
                new_height = screen_height
                new_width = int(frame_aspect_ratio * new_height)
            else:
                # L'écran est plus étroit, ajuster la largeur et centrer l'image
                new_width = screen_width
                new_height = int(new_width / frame_aspect_ratio)

            # Redimensionner l'image pour correspondre à la nouvelle taille
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # Créer une image noire pour remplir l'écran
            frame_to_show = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

            # Calculer les offsets pour centrer l'image
            x_offset = (screen_width - new_width) // 2
            y_offset = (screen_height - new_height) // 2

            # Placer l'image redimensionnée au centre de l'écran
            frame_to_show[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_frame

            # Ajouter le texte après le redimensionnement
            font_scale = 3.0
            thickness = 13
            y_pos = 150
            cv2.putText(frame_to_show, text, (x_offset + 10, y_offset + y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 255, 0), thickness)

            cv2.imshow('Camera', frame_to_show)

            if save_video:
                out.write(frame_to_show)

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

    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[color_map[int(label)] for label in labels],
                         picker=True)
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


def plot_and_save_confusion_matrix(cm, class_names, save_dir, filename="confusion_matrix.png"):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normaliser la matrice de confusion
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Afficher les valeurs dans les cellules de la matrice
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]} ({cm_normalized[i, j]:.2f})",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Sauvegarder l'image de la matrice de confusion
    confusion_matrix_path = os.path.join(save_dir, filename)
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Matrice de confusion enregistrée sous: {confusion_matrix_path}")


from matplotlib.colors import LinearSegmentedColormap


def create_custom_colormap():
    # Exemple : créer une colormap dégradée du bleu au rouge
    colors = [(0, 0, 1), (1, 0, 0)]  # Bleu à Rouge
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    return custom_cmap


def generate_heatmap(grad_cam, img, target_category, colormap='hot'):
    input_tensor = img.unsqueeze(0).requires_grad_(True).cuda()
    targets = [ClassifierOutputTarget(target_category)]
    grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    # Dénormaliser l'image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * std) + mean
    img_np = np.clip(img_np, 0, 1)

    # Générer la visualisation colorée du Grad-CAM
    visualization = show_cam_on_image(img_np.astype(np.float32), grayscale_cam, use_rgb=True, colormap=colormap)

    # Générer la version en niveaux de gris du Grad-CAM
    grayscale_cam_img = np.uint8(255 * grayscale_cam)
    grayscale_cam_img = cv2.cvtColor(grayscale_cam_img, cv2.COLOR_GRAY2RGB)
    grayscale_cam_img = grayscale_cam_img / 255.0  # Normaliser entre 0 et 1

    return visualization, grayscale_cam_img


def test(classifier, moco_model, loader, criterion, writer, class_names, save_dir, transform, prob_threshold,
         visualize_gradcam, save_gradcam_images, measure_time, save_test_images, colormap, compute_auc=False):
    moco_model.eval()
    classifier.eval()
    correct = 0
    total = 0
    total_loss = 0
    all_preds = []
    all_targets = []
    all_probs = []  # Pour stocker les probabilités
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
            max_probs = max_probs.cpu()
            predicted = predicted.cpu()
            target_cpu = target.cpu()
            total += target.size(0)

            # Stocker les vraies étiquettes et les probabilités
            all_targets.extend(target_cpu.numpy())
            all_probs.extend(probabilities.cpu().numpy())

            # Application du seuil de probabilité
            unknown_mask = max_probs < prob_threshold
            predicted_with_threshold = predicted.clone()
            predicted_with_threshold[unknown_mask] = -1  # On peut utiliser -1 pour représenter "Inconnu"

            # Calcul des métriques en tenant compte du seuil
            correct += (predicted_with_threshold == target_cpu).sum().item()
            all_preds.extend(predicted_with_threshold.numpy())
            all_features.append(features.cpu().numpy())

            end_time = time.time()
            times.append(end_time - start_time)

            if save_test_images:
                for j in range(data.size(0)):
                    img_path = original_images[j]
                    img = Image.open(img_path)
                    img = np.array(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    label = class_names[target_cpu[j].item()]
                    if predicted_with_threshold[j].item() == -1:
                        pred = "Unknown"
                    else:
                        pred = class_names[predicted_with_threshold[j].item()]
                    prob = max_probs[j].item()

                    text = f"Label: {label}, Pred: {pred}, Prob: {prob:.4f}"
                    cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    class_dir = os.path.join(save_dir, label)
                    img_save_path = os.path.join(class_dir, f"{i}_{j}.png")
                    cv2.imwrite(img_save_path, img)

                    if writer:
                        img_tensor = torch.tensor(img).permute(2, 0, 1)
                        writer.add_image(f'Test/Images/{i}_{j}', img_tensor, global_step=i)
                        writer.add_text(f'Test/Prédictions/{i}_{j}', text, global_step=i)

                    combined_model = CombinedModel(moco_model, classifier)
                    combined_model.eval()
                    # ...
                    if visualize_gradcam:
                        # Accéder à layer4
                        layer4 = combined_model.moco_model.truncated_encoder[-2]
                        # Accéder au dernier bloc Bottleneck de layer4
                        last_bottleneck = layer4[-1]
                        # Accéder à la dernière couche convolutionnelle conv3
                        target_layer = last_bottleneck.conv3
                        grad_cam = GradCAM(model=combined_model, target_layers=[target_layer])

                        input_img = data[j]
                        label_idx = target[j].item()
                        with torch.set_grad_enabled(True):
                            visualization, grayscale_cam_img = generate_heatmap(grad_cam, input_img, label_idx,
                                                                                colormap=colormap)

                        # Chargement de l'image originale et redimensionnement
                        orig_img = Image.open(img_path)
                        orig_img = orig_img.resize((224, 224))
                        orig_img = np.array(orig_img)
                        # S'assurer que l'image est en RGB
                        if orig_img.shape[2] == 4:  # Si l'image a un canal alpha
                            orig_img = orig_img[:, :, :3]
                        orig_img = orig_img / 255.0  # Normaliser entre 0 et 1

                        # Créer l'image combinée avec l'image originale, le Grad-CAM coloré et le Grad-CAM en niveaux de gris
                        combined_image = np.hstack((orig_img, visualization, grayscale_cam_img))
                        # Convertir en BGR pour OpenCV
                        combined_image_bgr = cv2.cvtColor((combined_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                        if save_gradcam_images:
                            gradcam_class_dir = os.path.join(save_dir, f"GradCAM_{label}")
                            if not os.path.exists(gradcam_class_dir):
                                os.makedirs(gradcam_class_dir)
                            combined_image_save_path = os.path.join(gradcam_class_dir, f"gradcam_{i}_{j}.png")
                            cv2.imwrite(combined_image_save_path, combined_image_bgr)

                    if writer:
                        combined_image_tensor = torch.tensor(combined_image).permute(2, 0, 1)
                        writer.add_image(f'GradCAM/Images/{i}_{j}', combined_image_tensor, global_step=i)

    # Calcul des métriques supplémentaires après la boucle
    # Mise à jour des métriques pour tenir compte de la classe "Inconnue"
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Calcul de l'exactitude
    accuracy = 100 * correct / total
    average_loss = total_loss / len(loader)

    # Calcul des métriques classiques
    if np.any(all_preds != -1):
        precision = precision_score(all_targets[all_preds != -1], all_preds[all_preds != -1], average='weighted',
                                    zero_division=0)
        recall = recall_score(all_targets[all_preds != -1], all_preds[all_preds != -1], average='weighted',
                              zero_division=0)
        f1 = f1_score(all_targets[all_preds != -1], all_preds[all_preds != -1], average='weighted', zero_division=0)
        cm = confusion_matrix(all_targets[all_preds != -1], all_preds[all_preds != -1],
                              labels=list(range(len(class_names))))
    else:
        precision = recall = f1 = 0
        cm = np.zeros((len(class_names), len(class_names)), dtype=int)

    # Calcul de l'AUC si demandé
    if compute_auc:
        try:
            # Pour multi-classes, utiliser 'ovr' (one-vs-rest)
            if len(class_names) > 2:
                auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(all_targets, all_probs[:, 1])  # Probabilité de la classe positive
        except ValueError as e:
            print(f"Erreur lors du calcul de l'AUC: {e}")
            auc = None
    else:
        auc = None

    # Sauvegarder les métriques dans un fichier texte
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"Précision du test: {accuracy}\n")
        f.write(f"Perte du test: {average_loss}\n")
        f.write(f"Précision: {precision}\n")
        f.write(f"Rappel: {recall}\n")
        f.write(f"Score F1: {f1}\n")
        if auc is not None:
            f.write(f"Score AUC: {auc}\n")
        f.write(f"Matrice de confusion:\n{cm}\n")

    # Afficher les métriques
    print(
        f"Précision du test: {accuracy}, Perte du test: {average_loss}, Précision: {precision}, Rappel: {recall}, Score F1: {f1}")
    if auc is not None:
        print(f"Score AUC: {auc}")

    # Sauvegarder la matrice de confusion
    plot_and_save_confusion_matrix(cm, class_names, save_dir)

    # Sauvegarder l'AUC dans un fichier séparé si nécessaire
    if compute_auc and auc is not None:
        with open(os.path.join(save_dir, "auc_score.txt"), "w") as f:
            f.write(f"AUC Score: {auc}\n")
        print(f"AUC Score sauvegardé dans 'auc_score.txt'")

    # Sauvegarder les temps si demandé
    if measure_time:
        with open(os.path.join(save_dir, "times_test.json"), "w") as f:
            json.dump(times, f, indent=4)
        print(f"Temps moyen de traitement par image: {np.mean(times)} secondes")
        print(f"Temps total de traitement: {np.sum(times)} secondes")

    return accuracy, f"{average_loss:.4f}", precision, recall, f1, cm, auc




