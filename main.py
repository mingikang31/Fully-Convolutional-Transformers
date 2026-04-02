"""Main File for Fully Convolutional Transformers (FCT)"""
import argparse 
from pathlib import Path 
import os
import torch 

# Datasets and Eval 
from train_eval import Train_Eval, Train_Eval_GPT
from dataset import CIFAR10, CIFAR100, WikiText103

# Models 
from Models.gpt2 import GPT2
from Models.vit import ViT
from Models.fcvt import VFCT

# Utils 
from utils import write_to_file, set_seed 

def args_parser():
    parser = argparse.ArgumentParser(description="Fully Convolutional Transformers (FCT)")

    # Model Specifications 
    parser.add_argument('--model', type=str, default='FCVT', choices=['FCVT', 'FCT', 'ViT', 'GPT2'], help='Model to use')
    parser.add_argument('--model_size', type=str, default='tiny', choices=['tiny', 'small', 'medium', 'large'], help='Model size variant')

    # Dataset  
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100", "wikitext103"], help="Dataset to use for training and evaluation")
    parser.add_argument("--resize", type=int, default=None, help="Resize images to 224x224")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation")
    parser.set_defaults(augment=False)
    parser.add_argument("--noise", type=float, default=0.0, help="Standard deviation of Gaussian noise to add to the data")
    parser.add_argument("--data_path", type=str, default="./Data", help="Path to the dataset")

    # Training Hyperparameters 
    parser.add_argument("--compile", action="store_true", help="Use compiled model for training and evaluation")
    parser.set_defaults(compile=False)
    parser.add_argument("--compile_mode", type=str, default="default", choices=["default", "reduce-overhead", "reduce-memory", "reduce-overhead", "max-autotune"], help="Compilation mode for torch.compile")
    
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of epochs for training")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision training")
    parser.set_defaults(use_amp=False)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping value")

    # Loss Function 
    parser.add_argument("--criterion", type=str, default="CrossEntropy", choices=["CrossEntropy", "MSE"], help="Loss function to use for training")

    # Optimizer 
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='Default Optimizer: adamw')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for optimizer') # For Adam & Adamw

    # LR + Scheduler 
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate for the optimizer")
    parser.add_argument('--scheduler', type=str, default='linear', choices=['step', 'cosine', 'plateau', 'linear'], help='Learning rate scheduler type')
    parser.add_argument('--lr_step', type=int, default=20, help='Step size for learning rate scheduler') # Only for StepLR
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Gamma for StepLR scheduler') # Only for StepLR

    # Device + Seed 
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training and evaluation')
    parser.add_argument('--seed', default=0, type=int)

    # Output Directory 
    parser.add_argument("--output_dir", type=str, default="./Output/TEST", help="Directory to save the output files")

    # Test Only
    parser.add_argument("--test_only", action="store_true", help="Only test the model")
    parser.set_defaults(test_only=False)

    
    return parser.parse_args()


def main(args):
    pass 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Fully Convolutional Transformers (FCT)", parents=[args_parser()], add_help=False)
    args = parser.parse_args()
    main(args)
