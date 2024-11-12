
import os
import json
import torch

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import cv2
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image







def load_model(model, model_path, device):
    if os.path.isfile(model_path):
        print(f"Loading pre-trained MoCo v3 model from {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"No model found at {model_path}")


def load_best_model(classifier, moco_model, filepath):
    checkpoint = torch.load(filepath)
    moco_state_dict = {k: v for k, v in checkpoint.items() if not k.startswith('classifier')}
    classifier_state_dict = {k.replace('classifier.', 'fc.'): v for k, v in checkpoint.items() if k.startswith('classifier')}

    moco_model.load_state_dict(moco_state_dict)
    classifier.load_state_dict(classifier_state_dict)
    moco_model.cuda()
    classifier.cuda()

def set_parameter_requires_grad(model, freeze_encoder):
    if freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name and "fc" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                print(f"Layer {name} is unfrozen.")

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

            print(
                f'Fold {fold}, Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

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
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    print(f'Fold {fold}, Validation Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    if writer:
        writer.add_scalar(f"Fold_{fold}/Validation/Loss", total_loss)
        writer.add_scalar(f"Fold_{fold}/Validation/Accuracy", accuracy)
        writer.add_scalar(f"Fold_{fold}/Validation/Precision", precision)
        writer.add_scalar(f"Fold_{fold}/Validation/Recall", recall)
    return total_loss, accuracy.item(), precision, recall


#trainning best function
def evaluate_model_best(moco_model, classifier, val_loader, criterion, writer=None, fold=0):
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
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
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

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def generate_heatmap(grad_cam, img, target_category):
    grad_cam.model.eval()
    input_tensor = img.unsqueeze(0).cuda()
    targets = [ClassifierOutputTarget(target_category)]
    grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.permute(1, 2, 0).cpu().numpy(), grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam

def show_images_side_by_side(orig_img, cam_img, grayscale_cam_img):
    grayscale_cam_img_rgb = cv2.cvtColor(grayscale_cam_img, cv2.COLOR_GRAY2RGB)
    combined_image = np.hstack((orig_img, cam_img, grayscale_cam_img_rgb))
    return combined_image

def show_cam_on_image(img, mask, use_rgb=False):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

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

def train_model_best(moco_model, classifier, train_loader, criterion, optimizer, num_epochs=25, writer=None, fold=0):
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



