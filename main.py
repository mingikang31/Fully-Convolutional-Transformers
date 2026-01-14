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

from utils import write_to_file, set_seed 

