'''CIFAR100, CIFAR10, MNIST Datasets'''
import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np 
import os 
from pathlib import Path 

# Huggingface Datasets + GPT Tokenizer
from datasets import load_dataset, load_from_disk
from transformers import GPT2Tokenizer 


'''Wikitext-103 Dataset Class'''
class WikiText103:
    def __init__(self, args):
        self.max_seq_length = args.max_seq_length
        self.block_size = self.max_seq_length
        self.cache_dir = os.path.join(args.data_path, f"wikitext103_cache_{args.max_seq_length}")
        
        # Tokenizer 
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if os.path.exists(self.cache_dir):
            print(f"Loading preprocessed dataset from {self.cache_dir}")
            self.lm_dataset = load_from_disk(self.cache_dir)
        else: 
            os.makedirs(self.cache_dir, exist_ok=True)

            # Dataset
            self.original_dataset = load_dataset("wikitext", "wikitext-103-v1")

        
            self.tokenized_dataset = self.original_dataset.map(
                self.tokenize_function, 
                batched=True, 
                remove_columns=["text"]
            )
            
            self.lm_dataset = self.tokenized_dataset.map(
                self.group_texts, 
                batched=True
            )
            self.lm_dataset.set_format(type="torch", columns=["input_ids"])
            self.lm_dataset.save_to_disk(self.cache_dir)
            print(f"Preprocessed dataset saved to {self.cache_dir}")

        # Data Loaders 
        self.train_loader = DataLoader(
            dataset=self.lm_dataset["train"], 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True)
        
        self.test_loader = DataLoader(
            dataset=self.lm_dataset["test"], 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True)
        
        self.val_loader = DataLoader(
            dataset=self.lm_dataset["validation"], 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=2, 
            pin_memory=True)

    def group_texts(self, examples): 
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

'''CIFAR-10 and CIFAR-100 Dataset Classes'''
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class CIFAR100(datasets.CIFAR100): 
    def __init__(self, args): 

        self.CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
        self.CIFAR100_STD = (0.2675, 0.2565, 0.2761)

        # Train Transformations
        self.train_transform_list = []
        if args.resize: 
            self.train_transform_list += [
                transforms.Resize((args.resize, args.resize))
                ]
        if args.augment:
            self.train_transform_list += [
                transforms.RandomCrop(args.resize if args.resize else 32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
            
        self.train_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR100_MEAN, std=self.CIFAR100_STD),
            ]
        if args.noise > 0.0:
            self.train_transform_list += [AddGaussianNoise(mean=0., std=args.noise)]

        # Test Transformations
        self.test_transform_list = []
        if args.resize: 
            self.test_transform_list += [transforms.Resize((args.resize, args.resize))]
        self.test_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR100_MEAN, std=self.CIFAR100_STD)
        ]
        
        # Define transforms
        train_transform = transforms.Compose(self.train_transform_list)
        test_transform = transforms.Compose(self.test_transform_list)

        # Load Datasets
        self.train_data = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=train_transform)
        self.test_data = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=test_transform)

        # Data Loaders
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # Set image size and number of classes
        self.img_size = (3, args.resize, args.resize) if args.resize else (3, 32, 32)
        self.num_classes = 100

    def shape(self):
        return self.train_data[0][0].shape
        
class CIFAR10(datasets.CIFAR10): 
    def __init__(self, args):

        self.CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        self.CIFAR10_STD = (0.2470, 0.2435, 0.2616)

        # Transformations
        # Train Transformations
        self.train_transform_list = []
        if args.resize: 
            self.train_transform_list += [transforms.Resize((args.resize, args.resize))]
        if args.augment:
            self.train_transform_list += [
                transforms.RandomCrop(args.resize if args.resize else 32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ]
        self.train_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR10_MEAN, std=self.CIFAR10_STD),
            ]
        if args.noise > 0.0:
            self.train_transform_list += [AddGaussianNoise(mean=0., std=args.noise)]

        # Test Transformations
        self.test_transform_list = []
        if args.resize: 
            self.test_transform_list += [transforms.Resize((args.resize, args.resize))]
        self.test_transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=self.CIFAR10_MEAN, std=self.CIFAR10_STD)
        ]
        
        # Define transforms
        train_transform = transforms.Compose(self.train_transform_list)
        test_transform = transforms.Compose(self.test_transform_list)

        # Load Datasets
        self.train_data = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=train_transform)
        self.test_data = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transform)

        # Data Loaders
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        # Set image size and number of classes
        self.img_size = (3, args.resize, args.resize) if args.resize else (3, 32, 32)
        self.num_classes = 10

    def shape(self): 
        return self.train_data[0][0].shape
    