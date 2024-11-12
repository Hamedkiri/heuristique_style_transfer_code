import os
import json
import torch
import torch.nn as nn

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
import time
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg







def train_model(model, train_loader, criterion, optimizer, num_epochs=25, writer=None, fold=0):
    """
    Entraîne le modèle sur les données d'entraînement.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            classification_loss = criterion(outputs, labels)
            loss = classification_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            print(
                f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        if writer:
            writer.add_scalar(f"Fold_{fold}/Train/Loss", epoch_loss, epoch)
    return model


def evaluate_model(model, val_loader, criterion, writer=None, fold=0):
    """
    Évalue le modèle sur les données de validation.
    """
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
            classification_loss = criterion(outputs, labels)
            loss = classification_loss
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
    print(f'Fold {fold}, Validation Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')
    if writer:
        writer.add_scalar(f"Fold_{fold}/Validation/Loss", total_loss, fold)
        writer.add_scalar(f"Fold_{fold}/Validation/Accuracy", accuracy, fold)
        writer.add_scalar(f"Fold_{fold}/Validation/Precision", precision, fold)
        writer.add_scalar(f"Fold_{fold}/Validation/Recall", recall, fold)
        writer.add_scalar(f"Fold_{fold}/Validation/F1", f1, fold)
    return total_loss, accuracy.item(), precision, recall, f1


def plot_confusion_matrix(cm, class_names, output_dir):
    """
    Enregistre la matrice de confusion sous forme d'image.
    """
    import matplotlib.pyplot as plt
    import itertools

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matrice de Confusion')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Normalisation de la matrice de confusion
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=8)

    plt.tight_layout()
    plt.ylabel('Vraies étiquettes')
    plt.xlabel('Prédictions')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()



def gram_matrix(activations):
    if activations.dim() == 4:
        # Cas où les activations sont de la forme (batch_size, channels, height, width)
        (b, ch, h, w) = activations.size()
        features = activations.view(b, ch, h * w)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(h * w)

    elif activations.dim() == 3:
        # Cas où les activations sont de la forme (channels, height, width) (sans dimension de batch)
        ch, h, w = activations.size()
        features = activations.view(ch, h * w)
        G = torch.mm(features, features.t())
        return G.div(h * w)

    elif activations.dim() == 2:
        # Cas où les activations sont de la forme (batch_size, num_patches)
        G = torch.mm(activations, activations.t())
        return G.div(activations.size(1))

    elif activations.dim() == 1:
        # Cas où les activations sont des vecteurs (ex: torch.Size([512]))
        # Pour les vecteurs 1D, nous pouvons simplement calculer un produit extérieur
        G = torch.ger(activations, activations)  # Produit extérieur pour un vecteur
        return G.div(activations.size(0))

    else:
        raise ValueError(f"Unsupported activation shape: {activations.size()}")

def evaluate_model_test(model, test_loader, criterion, measure_time=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    test_loss = 0.0
    corrects = 0
    all_preds = []
    all_labels = []
    all_embeddings = []
    times = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if measure_time:
                start_time = time.time()

            embeddings, outputs = model(inputs)

            if measure_time:
                elapsed_time = time.time() - start_time
                times.append(elapsed_time / inputs.size(0))

            classification_loss = criterion(outputs, labels)
            loss = classification_loss
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_embeddings.extend(embeddings.cpu().numpy())

    total_loss = test_loss / len(test_loader.dataset)
    accuracy = corrects.double() / len(test_loader.dataset)

    avg_time_per_image = np.mean(times) if measure_time else None

    return total_loss, accuracy, avg_time_per_image, np.array(all_embeddings), np.array(all_labels)

def evaluate_classification(model, test_loader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return precision, recall, f1, all_preds, all_labels

def style_transfer_patches(model, data_loader, device, save_dir, layers=None, threshold=1e-4, num_iterations=500,
                           learning_rate=0.01, max_images=None):
    model.to(device)
    model.eval()

    # Créer le sous-répertoire pour enregistrer les résultats
    current_date = datetime.now().strftime("%Y-%m-%d")
    style_transfer_dir = os.path.join(save_dir, f'style_transfer_{current_date}')
    os.makedirs(style_transfer_dir, exist_ok=True)

    mse_loss = nn.MSELoss()

    # Les valeurs de normalisation utilisées lors de la transformation des images
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)

    image_count = 0  # Compteur pour suivre le nombre d'images traitées

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        for i, input_image in enumerate(inputs):
            if max_images is not None and image_count >= max_images:
                print(f"Nombre maximal d'images ({max_images}) atteint.")
                return  # Quitter la fonction une fois que le nombre maximal est atteint

            input_image = input_image.unsqueeze(0).to(device)  # S'assurer que l'image est sur le même appareil

            # Utilisation des couches spécifiées par 'layers' pour extraire les patches implicites
            with torch.no_grad():
                if layers is not None:
                    truncated_model = nn.Sequential(*list(model.children())[:layers]).to(device)
                    original_patches, _ = truncated_model(input_image)  # Extraire les activations tronquées
                else:
                    original_patches, _ = model(input_image)  # Utiliser tout le modèle si 'layers' n'est pas spécifié

                # Calcul des matrices de Gram pour chaque patch de l'image d'origine
                original_gram_patches = [gram_matrix(patch) for patch in original_patches]

            # Créer un répertoire pour la classe correspondante
            class_dir = os.path.join(style_transfer_dir, str(labels[i].item()))
            os.makedirs(class_dir, exist_ok=True)

            # Réinitialiser l'image de bruit pour chaque image cible et s'assurer que le gradient est calculé
            noise_image = torch.randn((1, 3, 224, 224), device=device, requires_grad=True)

            optimizer = torch.optim.Adam([noise_image], lr=learning_rate)

            for iteration in range(num_iterations):
                optimizer.zero_grad()

                # Extraire les patches implicites de l'image bruitée
                if layers is not None:
                    noise_patches, _ = truncated_model(noise_image)
                else:
                    noise_patches, _ = model(noise_image)

                # Calcul des matrices de Gram pour chaque patch de l'image bruitée
                noise_gram_patches = [gram_matrix(patch) for patch in noise_patches]

                # Calcul de la perte entre les matrices de Gram des patches originaux et ceux de l'image bruitée
                loss = sum(mse_loss(noise_gram, original_gram) for noise_gram, original_gram in
                           zip(noise_gram_patches, original_gram_patches))

                # Calcul des gradients et optimisation
                loss.backward()
                optimizer.step()

                if loss.item() < threshold:
                    print(f"Seuil atteint pour l'image {i}, itération {iteration}")
                    break

            # Dénormaliser les images avant de les afficher
            output_image = noise_image.detach().cpu().squeeze()
            output_image = denormalize(output_image, mean, std).clamp_(0, 1).numpy().transpose(1, 2, 0)

            original_image = input_image.detach().cpu().squeeze()
            original_image = denormalize(original_image, mean, std).clamp_(0, 1).numpy().transpose(1, 2, 0)

            # Combiner les deux images (originale + stylisée)
            combined_image = np.hstack((original_image, output_image))

            # Créer un nom de fichier unique pour chaque image
            timestamp = int(time.time() * 1000)  # Timestamp en millisecondes
            unique_filename = f'style_transfer_{labels[i].item()}_{image_count}_{timestamp}.png'
            save_path = os.path.join(class_dir, unique_filename)
            plt.imsave(save_path, combined_image)

            print(f"Style transféré pour l'image {i}, sauvegardée à {save_path}")

            image_count += 1  # Incrémenter le compteur d'images traitées

        # Vérifier après chaque batch si le nombre maximal d'images a été atteint
        if max_images is not None and image_count >= max_images:
            print(f"Nombre maximal d'images ({max_images}) atteint.")
            return  # Quitter la fonction une fois que le nombre maximal est atteint

def plot_tsne(embeddings, labels, classes, colors=None, save_dir='results'):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    if colors and len(colors) >= num_classes:
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
    else:
        color_palette = plt.cm.get_cmap("tab10", num_classes)
        color_map = {label: color_palette(i / num_classes) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(10, 10))
    for label in unique_labels:
        indices = labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], c=[color_map[label]], label=classes[int(label)])
    plt.legend()
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(os.path.join(save_dir, 'tsne_plot.png'))
    plt.show()

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
        color_palette = plt.cm.get_cmap("tab10", num_classes)
        color_map = {label: color_palette(i / num_classes) for i, label in enumerate(unique_labels)}

    scatter = ax.scatter(tsne_results[:, 0], tsne_results[:, 1], c=[color_map[int(label)] for label in labels],
                         picker=True)
    # Ajouter une légende avec les noms des classes
    handles = []
    for label in unique_labels:
        handles.append(plt.Line2D([], [], marker='o', color=color_map[int(label)], linestyle='', markersize=5))
    ax.legend(handles, [classes[int(label)] for label in unique_labels], title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    img_label = tk.Label(root)
    img_label.grid(row=0, column=1, sticky='nsew')
    label_text = tk.StringVar()
    label_label = tk.Label(root, textvariable=label_text)
    label_label.grid(row=1, column=1, sticky='nsew')

    def onpick(event):
        ind = event.ind[0]
        if isinstance(img_paths[ind], str):
            img_path = img_paths[ind]
        else:
            img_path = dataset.samples[img_paths[ind]][0]
        img = Image.open(img_path)
        img = img.resize((400, 400), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
        label_text.set(f"Label: {classes[int(labels[ind])]}")
        print(f"Image path: {img_path}")

    fig.canvas.mpl_connect('pick_event', onpick)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, rowspan=2, sticky='nsew')

    global polygon_selector
    polygon = []
    polygon_selector = None

    def enable_polygon_selector(event):
        global polygon_selector
        if event.button == 3:
            if polygon_selector is None:
                polygon_selector = PolygonSelector(ax, onselect=onselect)
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
        polygon_path = Path(polygon)

        for i, (x, y) in enumerate(tsne_results):
            if polygon_path.contains_point([x, y]):
                inside_points.append(i)

        inside_labels = [labels[i] for i in inside_points]
        print(f"Points inside polygon: {len(inside_points)}")
        print(f"Labels inside polygon: {inside_labels}")

    def clear_polygon():
        global polygon_selector
        polygon.clear()
        if polygon_selector is not None:
            polygon_selector.set_visible(False)
            polygon_selector.disconnect_events()
            polygon_selector = None
        print("Polygon cleared.")
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', enable_polygon_selector)

    close_button = tk.Button(root, text="Close Polygon", command=analyze_polygon)
    close_button.grid(row=2, column=0, sticky='ew')

    clear_button = tk.Button(root, text="Clear Polygon", command=clear_polygon)
    clear_button.grid(row=2, column=1, sticky='ew')

    root.mainloop()

def run_camera(model, transform, class_names, save_video, save_dir, prob_threshold, measure_time):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
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

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Erreur: Impossible de lire l'image de la caméra")
                break

            start_time = time.time()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            img_tensor = transform(pil_img).unsqueeze(0).to(device)

            embeddings, outputs = model(img_tensor)
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
        print(f"Temps moyen de traitement par image: {np.mean(times)} secondes")
        print(f"Temps total de traitement: {np.sum(times)} secondes")

    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()

def denormalize(tensor, mean, std):
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # dénormaliser
    return tensor

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj