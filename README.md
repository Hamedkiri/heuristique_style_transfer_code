
# Heuristic Style Transfer Code

## Demonstration in real-time : Weather Measurement Around the World





https://github.com/user-attachments/assets/107a542f-496b-48d7-acc6-50cd370eba04








## Installing Required Packages

To install the required packages, create a Python virtual environment and then run the following command:

```bash
pip install -r requirements.txt
```

## Testing the Models

### Learn through video 





https://github.com/user-attachments/assets/9caad4cc-fbd4-44fa-9d66-7232f939e77c





###  Learn by reading


To test the models, you need to specify the hyperparameters file with the `--config_path` option to rebuild the model architecture, as well as the weights file with `--model_path`.

The command-line options are similar for all three models and include the following parameters:

- `--mode`: Specifies the type of test, with options such as `classification`, `tsne`, `tsne_interactive`, `camera`, `inference`, or for some models, `clustering` or `style_transfer`.
- `--data`: Path to the data folder, which should contain the classes in a "test" subfolder (e.g., `dataset/test/fog rain snow sun`).
- `--classes`: List of classes in alphabetical order (e.g., `--classes fog rain snow sun`).
- `--num_samples`: Number of samples to use for testing.
- `--save_dir`: Folder to save the results.

An optional parameter, `--measure_time`, can be added in `camera` mode to measure the processing time per image accurately.

### Example Testing Commands for Each Model

- **Truncated ResNet50:**
  ```bash
  python test_RESNET50_Truncate.py --config_path checkpoints/Best_ResNet50_Truncated/best_model_fold_hyperparameters_all.json --model_path checkpoints/Best_ResNet50_Truncated/best_model_fold_all.pth --num_samples 12000 --mode camera --classes fog rain snow sun
  ```

- **Truncated ResNet50 + Gram Matrix + Attention:**
  ```bash
  python test_RESNET50_Truncate_gram_attention.py --data ./datasets --model_path checkpoints/Best_Resner50_Truncated_with_Attention/best_model_all.pth --config_path checkpoints/Best_Resner50_Truncated_with_Attention/best_performance_all.json --mode classification --num_samples 12000 --save_dir results/test_results_gram_attention_resnet50_convolution
  ```

- **Multi PatchGAN:**
  ```bash
  python test_Multi_PatchGAN.py --model_path checkpoints/Best_Multi_PatchGAN/best_model_all.pth --config_path checkpoints/Best_Multi_PatchGAN/best_hyperparameters_all.json --data ./datasets --num_samples 12000 --mode tsne_interactive
  ```

## Training the Models

To retrain the models, use `--config_path` to specify the architecture and `--data` for the path to the training data, with classes in a "train" subfolder (e.g., `dataset/train/fog rain snow sun`). If you want to retrain from existing weights, add the `--model_path` option. The `--save_dir` parameter specifies where to save the model weights. Other options include `--epochs` to define the number of epochs and `--k_folds` for the number of cross-validation folds. for the data to retrain the model you can find it here: [Datasets on Google Drive](https://drive.google.com/drive/folders/1eqnTRWLPH1FbhZdvnazt01fxp0vUN47n?usp=sharing).

### Example Training Commands for Each Model

- **Truncated ResNet50:**
  ```bash
  python train_best_RESNET50_Truncate.py --data ./datasets --config_path checkpoints/Best_ResNet50_Truncated/best_model_fold_hyperparameters_all.json --model_path checkpoints/Best_ResNet50_Truncated/best_model_fold_all.pth --save_dir results/test_cancer_best --epochs 10 --k_folds 2
  ```

- **Truncated ResNet50 + Gram Matrix + Attention:**
  ```bash
  python train_best_RESNET50_Truncate_gram_attention.py --data ./datasets --model_path checkpoints/Best_Resner50_Truncated_with_Attention/best_model_all.pth --config_path checkpoints/Best_Resner50_Truncated_with_Attention/best_performance_all.json --epochs 10 --save_dir results/Models_resnet50_attention
  ```

- **Multi PatchGAN:**
  ```bash
  python train_best_Multi_PatchGAN.py --data ./datasets --model_path checkpoints/Best_Multi_PatchGAN/best_model_all.pth --config_path checkpoints/Best_Multi_PatchGAN/best_hyperparameters_all.json --save_dir results/Models_Multi_patchGAN --epochs 200 --k_folds 2
  ```

This `README.md` provides instructions on installing, testing, and training the models. Model weights are available at the following link: [Model weights on Google Drive](https://drive.google.com/drive/folders/11Pllunglo-_XcZSI80WheTKOeqceW9II?usp=sharing).
```
