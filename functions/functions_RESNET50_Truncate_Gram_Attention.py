import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Subset
from sklearn.metrics import precision_score, recall_score
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image, ImageTk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from tkinter import ttk
import cv2
import time
from datetime import datetime

torch.autograd.set_detect_anomaly(True)  # Pour activer la détection d'anomalies






def load_model(model, model_path, device):
    if os.path.isfile(model_path):
        print(f"Loading pre-trained ResNet50 model from {model_path}")
        # Charger le state_dict du modèle pré-entraîné
        resnet_state_dict = torch.load(model_path, map_location=device)

        # Créer un nouveau state_dict pour le truncated_encoder
        truncated_encoder_state_dict = {}

        # Obtenir le state_dict du truncated_encoder de votre modèle
        model_state_dict = model.state_dict()

        # Mapper les clés du resnet_state_dict vers les clés du truncated_encoder
        for key in resnet_state_dict.keys():
            if key.startswith('fc.'):
                # Ignorer la couche fully connected du ResNet50 pré-entraîné
                continue
            else:
                # Ajuster la clé pour correspondre à celle de truncated_encoder
                new_key = f"truncated_encoder.{key}"
                if new_key in model_state_dict:
                    truncated_encoder_state_dict[new_key] = resnet_state_dict[key]

        # Charger les poids dans le modèle
        model_state_dict.update(truncated_encoder_state_dict)
        model.load_state_dict(model_state_dict, strict=True)

        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"No model found at {model_path}")



def save_model_weights(model, save_path):
    state_dict = {
        'truncated_encoder': model.truncated_encoder.state_dict(),
        'classifier': model.classifier.state_dict(),
        'attention': model.attention.state_dict()  # Ajouter les poids de l'attention
    }
    torch.save(state_dict, save_path)
    print(f"Model weights saved to {save_path}")


def load_model_weights(model, load_path):
    if os.path.isfile(load_path):
        state_dict = torch.load(load_path, map_location=model.device)

        try:
            if 'truncated_encoder' in state_dict:
                model.truncated_encoder.load_state_dict(state_dict['truncated_encoder'], strict=True)
            else:
                print("Warning: 'truncated_encoder' not found in state_dict.")

            if 'classifier' in state_dict:
                model.classifier.load_state_dict(state_dict['classifier'], strict=True)
            else:
                print("Warning: 'classifier' not found in state_dict.")

            if 'attention' in state_dict:
                model.attention.load_state_dict(state_dict['attention'], strict=True)
            else:
                print("Warning: 'attention' not found in state_dict.")

            print(f"Model weights loaded from {load_path} using direct method.")

        except (KeyError, RuntimeError) as e:
            print(f"Direct loading failed with error: {e}")
            print("Attempting to load weights by processing keys...")

            truncated_encoder_state_dict = {}
            classifier_state_dict = {}
            attention_state_dict = {}

            for key, value in state_dict.items():
                if key.startswith('truncated_encoder'):
                    new_key = key.replace('truncated_encoder.', '')
                    truncated_encoder_state_dict[new_key] = value
                elif key.startswith('classifier'):
                    new_key = key.replace('classifier.', '')
                    classifier_state_dict[new_key] = value
                elif key.startswith('attention'):
                    new_key = key.replace('attention.', '')
                    attention_state_dict[new_key] = value

            model.truncated_encoder.load_state_dict(truncated_encoder_state_dict, strict=True)
            model.classifier.load_state_dict(classifier_state_dict, strict=True)
            model.attention.load_state_dict(attention_state_dict, strict=True)

            print(f"Model weights loaded from {load_path} by processing keys.")
    else:
        print(f"No weights file found at {load_path}. Proceeding without loading weights.")


def train_model(model, train_loader, criterion, optimizer, num_epochs=25, writer=None, fold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            print(f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        if writer:
            writer.add_scalar(f"Fold_{fold}/Train/Loss", epoch_loss, epoch)
    return model


def evaluate_model(model, val_loader, criterion, writer=None, fold=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    val_loss = 0.0
    corrects = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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
    print(f'Fold {fold}, Validation Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    if writer:
        writer.add_scalar(f"Fold_{fold}/Validation/Loss", total_loss)
        writer.add_scalar(f"Fold_{fold}/Validation/Accuracy", accuracy)
        writer.add_scalar(f"Fold_{fold}/Validation/Precision", precision)
        writer.add_scalar(f"Fold_{fold}/Validation/Recall", recall)
    return total_loss, accuracy.item(), precision, recall

def evaluate_model_test(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_embeddings = []
    img_paths = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            embeddings, outputs = model(inputs)
            all_embeddings.append(embeddings.cpu().numpy())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if isinstance(data_loader.dataset, Subset):
                img_paths.extend([data_loader.dataset.dataset.samples[i][0] for i in data_loader.dataset.indices])
            else:
                img_paths.extend([sample[0] for sample in data_loader.dataset.samples])
    return np.concatenate(all_embeddings), np.array(all_preds), np.array(all_labels), img_paths

def set_parameter_requires_grad(model, freeze_encoder):
    if freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name and "attention" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                print(f"Layer {name} is unfrozen.")
    else:
        for param in model.parameters():
            param.requires_grad = True




def denormalize(tensor, mean, std):
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # dénormaliser
    return tensor
def style_transfer(model, data_loader, device, save_dir, layers=None, threshold=1e-4, num_iterations=500,
                   learning_rate=0.01):
    model.eval()

    # Créer le sous-dossier pour enregistrer les résultats
    current_date = datetime.now().strftime("%Y-%m-%d")
    style_transfer_dir = os.path.join(save_dir, f'style_transfer_{current_date}')
    os.makedirs(style_transfer_dir, exist_ok=True)

    mse_loss = nn.MSELoss()

    # Les valeurs de normalisation utilisées lors de la transformation des images
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        for i, input_image in enumerate(inputs):
            input_image = input_image.unsqueeze(0)

            # Adapter le modèle tronqué en fonction de la couche spécifiée par layers
            truncated_encoder = nn.Sequential(
                *list(model.truncated_encoder.children())[:layers]
            ).to(device)

            # Obtenir la matrice de Gram des images de test
            with torch.no_grad():
                original_features = truncated_encoder(input_image)
                original_gram = model.gram_matrix(original_features)

            # Créer un répertoire pour la classe correspondante
            class_dir = os.path.join(style_transfer_dir, str(labels[i].item()))
            os.makedirs(class_dir, exist_ok=True)

            # Réinitialiser l'image de bruit pour chaque image cible et s'assurer que le gradient est calculé
            noise_image = torch.randn((1, 3, 224, 224), device=device, requires_grad=True)

            optimizer = torch.optim.Adam([noise_image], lr=learning_rate)

            for iteration in range(num_iterations):
                optimizer.zero_grad()

                # Assurez-vous que les features extraites sont calculées avec le gradient activé
                noise_features = truncated_encoder(noise_image)
                noise_gram = model.gram_matrix(noise_features)

                loss = mse_loss(noise_gram, original_gram)

                loss.backward(retain_graph=True)

                optimizer.step()

                if loss.item() < threshold:
                    print(f"Seuil atteint pour l'image {i}, itération {iteration}")
                    break

            # Dénormaliser les images avant de les afficher
            output_image = noise_image.detach().cpu().squeeze()
            output_image = denormalize(output_image, mean, std).clamp_(0, 1).numpy().transpose(1, 2, 0)

            original_image = input_image.detach().cpu().squeeze()
            original_image = denormalize(original_image, mean, std).clamp_(0, 1).numpy().transpose(1, 2, 0)

            combined_image = np.hstack((original_image, output_image))
            save_path = os.path.join(class_dir, f'style_transfer_{i}.png')
            plt.imsave(save_path, combined_image)

            print(f"Style transféré pour l'image {i}, sauvegardée à {save_path}")














def load_hyperparameters(hyperparams_path):
    if os.path.isfile(hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            hyperparams = json.load(f)
        print(f"Hyperparameters loaded from {hyperparams_path}")
        return hyperparams
    else:
        print(f"No hyperparameters file found at {hyperparams_path}. Proceeding with default hyperparameters.")
        return None





def perform_tsne(embeddings, labels, save_path, colors=None):
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    if colors and len(colors) >= num_classes:
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    else:
        color_palette = plt.cm.get_cmap("tab20", num_classes)
        color_map = {label: color_palette(label) for label in unique_labels}

    for label in unique_labels:
        indices = labels == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                    label=f'Class {label}', color=color_map[int(label)])
    plt.legend()
    plt.title('t-SNE of Embeddings')
    plt.savefig(save_path)
    plt.show()
    print(f"t-SNE visualization saved to {save_path}")


def create_onpick_function(dataset, img_paths, img_label, label_text, classes, labels):
    def onpick(event):
        ind = event.ind[0]
        img_path = img_paths[ind]
        print(f"Selected img_path: {img_path}")
        img = Image.open(img_path)
        img = img.resize((400, 400), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
        label_text.set(f"Label: {classes[int(labels[ind])]}")

    return onpick


def plot_tsne_interactive(embeddings, labels, classes, img_paths, dataset, colors=None):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    root = tk.Tk()
    root.title("Interactive t-SNE with Images")

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
    ax.legend(handles=scatter.legend_elements()[0], labels=[classes[int(label)] for label in unique_labels])

    img_label = tk.Label(root)
    img_label.grid(row=0, column=1, sticky='nsew')
    label_text = tk.StringVar()
    label_label = tk.Label(root, textvariable=label_text)
    label_label.grid(row=1, column=1, sticky='nsew')

    onpick = create_onpick_function(dataset, img_paths, img_label, label_text, classes, labels)
    fig.canvas.mpl_connect('pick_event', onpick)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, sticky='nsew')

    # Add Polygon Selector and Buttons
    polygon = []
    polygon_selector = None
    polygon_cleared = True  # Flag to indicate if the polygon is cleared

    def enable_polygon_selector(event):
        nonlocal polygon_selector, polygon_cleared
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
        polygon_path = Path(polygon)

        for i, (x, y) in enumerate(tsne_results):
            point = (x, y)
            if polygon_path.contains_point(point):
                inside_points.append({"path": img_paths[i], "class": classes[int(labels[i])], "position": point})
            else:
                outside_points.append({"path": img_paths[i], "class": classes[int(labels[i])], "position": point})

        print(f"Points inside polygon: {len(inside_points)}")

    def clear_polygon():
        nonlocal polygon_selector, polygon_cleared
        polygon.clear()
        if polygon_selector:
            polygon_selector.disconnect_events()
            polygon_selector.set_visible(False)
            del polygon_selector
            polygon_selector = None
        while ax.patches:
            ax.patches.pop().remove()  # Remove all patches
        fig.canvas.draw()
        polygon_cleared = True  # Mark the polygon as cleared

    fig.canvas.mpl_connect('button_press_event', enable_polygon_selector)

    close_button = tk.Button(root, text="Close Polygon", command=analyze_polygon)
    close_button.grid(row=4, column=0, sticky='ew')

    clear_button = tk.Button(root, text="Clear Polygon", command=clear_polygon)
    clear_button.grid(row=4, column=1, sticky='ew')

    root.mainloop()


def run_camera(model, transform, class_names, save_video, save_dir, prob_threshold, measure_time):
    model.eval()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open the camera")
        return

    if save_video:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        video_save_path = os.path.join(save_dir, "camera_output.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_save_path, fourcc, 20.0, (640, 480))

    times = []

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read the image from the camera")
                break

            start_time = time.time()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            img_tensor = transform(pil_img).unsqueeze(0).to(model.device)

            _, outputs = model(img_tensor)

            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_label = np.argmax(probabilities)
            pred_class = class_names[pred_label] if probabilities[pred_label] >= prob_threshold else "Unknown"
            prob = probabilities[pred_label]

            end_time = time.time()
            times.append(end_time - start_time)

            text = f"Pred: {pred_class}, Prob: {prob:.4f}"
            cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Camera', frame)

            if save_video:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if measure_time:
        with open(os.path.join(save_dir, "times_camera.json"), "w") as f:
            json.dump(times, f, indent=4)
        print(f"Average processing time per image: {np.mean(times)} seconds")
        print(f"Total processing time: {np.sum(times)} seconds")

    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()

