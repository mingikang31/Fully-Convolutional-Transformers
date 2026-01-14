'''Training & Evaluation Module for Convolutional Neural Networks'''

import torch
import torch.nn as nn
import torch.optim as optim
import time 
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import torch.profiler
from utils import set_seed
from transformers import get_linear_schedule_with_warmup

def Train_Eval(args, 
               model: nn.Module, 
               train_loader, 
               test_loader
               ):
    
    if args.seed != 0:
        set_seed(args.seed)

    # Loss Criterion
    if args.criterion == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == 'MSE':
        criterion = nn.MSELoss()
            
    # Optimizer 
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        

    # Learning Rate Scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
        
    # Device
    device = args.device
    model.to(device)
    criterion.to(device)

    scaler = GradScaler() if args.use_amp else None 
        
    epoch_results = []

    ## [GFLOPS] Computation using PyTorch Profiler ##
    try:
        input_tensor, _ = next(iter(train_loader))
        input_tensor = input_tensor.to(device)

        # Profile a single forward pass
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            with torch.no_grad():
                model(input_tensor[0:1])

        total_flops = sum(event.flops for event in prof.key_averages())
        if total_flops > 0:
            gflops = total_flops / 1e9
            params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            print(f"Model Complexity (Profiler):")
            print(f"   - Total Parameters: {params:.8f} M")
            print(f"   - GFLOPs: {gflops:.8f}")
            epoch_results.append(f"Model Complexity (Profiler): GFLOPs: {gflops:.8f}, Trainable Parameters: {params:.8f} M")
    except Exception as e:
        print(f"Could not calculate GFLOPs with PyTorch Profiler: {e}")
    
    # Training Loop
    epoch_times = [] # Average Epoch Time 
    max_accuracy = 0.0 
    max_epoch = 0
    
    for epoch in range(args.num_epochs):
        # Model Training
        model.train() 
        train_running_loss = 0.0
        test_running_loss = 0.0
        epoch_result = ""
        
        start_time = time.time()

        train_top1_5 = [0, 0]
        for images, labels in train_loader: 
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # use mixed precision training
            if args.use_amp:
                with autocast(device_type=args.device):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                if args.clip_grad_norm:
                    scaler.unscale_(optimizer) # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:    
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                if args.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()            

            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            train_top1_5[0] += top1.item()
            train_top1_5[1] += top5.item()
            train_running_loss += loss.item()

        train_top1_5[0] /= len(train_loader)
        train_top1_5[1] /= len(train_loader)
        
        # Model Evaluation 
        model.eval()
        test_top1_5 = [0, 0]
        with torch.no_grad():
            for images, labels in test_loader: 
                images, labels = images.to(device), labels.to(device)
                if args.use_amp:
                    with autocast(device_type=args.device):
                        outputs = model(images)
                else: 
                    outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item()

                top1, top5 = accuracy(outputs, labels, topk=(1, 5))
                test_top1_5[0] += top1.item()
                test_top1_5[1] += top5.item()
        
        test_top1_5[0] /= len(test_loader)
        test_top1_5[1] /= len(test_loader)

        # Single Epoch Duration
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        # Save Epoch Results
        epoch_results.append(f"[Epoch {epoch+1:03d}] Time: {epoch_time:.4f}s | [Train] Loss: {train_running_loss/len(train_loader):.8f} Accuracy: Top1: {train_top1_5[0]:.4f}%, Top5: {train_top1_5[1]:.4f}% | [Test] Loss: {test_running_loss/len(test_loader):.8f} Accuracy: Top1: {test_top1_5[0]:.4f}%, Top5: {test_top1_5[1]:.4f}%")
        print(epoch_results[-1])
        
        # Max Accuracy Check
        if test_top1_5[0] > max_accuracy:
            max_accuracy = test_top1_5[0]
            max_epoch = epoch + 1    
            
        # Learning Rate Scheduler Step
        if scheduler: 
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_top1_5[0])
            else:
                scheduler.step()
                
                
    epoch_results.append(f"\nAverage Epoch Time: {sum(epoch_times) / len(epoch_times):.4f}s")
    epoch_results.append(f"Max Accuracy: {max_accuracy:.4f}% at Epoch {max_epoch}")
    
    return epoch_results

"""
Measuring Perplexity (PPL) for GPT Models. 
- Validation loop use a "sliding window" 
"""
def Train_Eval_GPT(args, 
                   model: nn.Module, 
                   train_loader, 
                   test_loader, 
                   val_loader
                   ):
    if args.seed != 0:
        set_seed(args.seed)

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler Setup
    total_steps = len(train_loader) * args.num_epochs 
    warmup_steps = int(0.05 * total_steps) # Warmup for 5% of training steps

    # Learning Rate Scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == 'cosine': 
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    elif args.scheduler == 'linear': ## MAIN ONE IN USE
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )

    # Device 
    device = args.device 
    model.to(device) 

    scaler = GradScaler() if args.use_amp else None
    
    epoch_results = [] 
    
    ## [GFLOPS] Computation using PyTorch Profiler ##
    try:
        batch = next(iter(train_loader))

        tokens = batch["input_ids"].to(device)
        
        inputs = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()

        # Profile a single forward pass
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            with torch.no_grad():
                logits, loss = model(inputs[0:1], target=targets[0:1])

        # A more robust way to get total FLOPs: sum them up from all events
        total_flops = sum(event.flops for event in prof.key_averages())

        if total_flops > 0:
            gflops = total_flops / 1e9
            params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

            print(f"Model Complexity (Profiler):")
            print(f"   - Total Parameters: {params:.8f} M")
            print(f"   - GFLOPs: {gflops:.8f}")
            epoch_results.append(f"Model Complexity (Profiler): GFLOPs: {gflops:.8f}, Trainable Parameters: {params:.8f} M")

    except Exception as e:
        print(f"Could not calculate GFLOPs with PyTorch Profiler: {e}")

    # Training Loop 
    epoch_times = [] # Average Epoch Time
    min_perplexity = float('inf')
    min_epoch = 0

    for epoch in range(args.num_epochs):
        start_time = time.time() 

        # Training Loop
        train_running_loss = 0.0
        model.train() 
        for batch in train_loader: 
            tokens = batch["input_ids"].to(device)

            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()

            optimizer.zero_grad() 
            if args.use_amp: 
                with autocast(device_type=args.device):
                    logits, loss = model(inputs, target=targets)

                # nan for loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss at Epoch {epoch+1}")
                    continue
                    
                scaler.scale(loss).backward()
                
                if args.clip_grad_norm: 
                    scaler.unscale_(optimizer) 
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
            else: 
                logits, loss = model(inputs, target=targets)
                loss.backward()
                if args.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
            if scheduler and args.scheduler == 'linear':
                scheduler.step()

            train_running_loss += loss.item()

        avg_train_loss = train_running_loss / len(train_loader)
        train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
        
        # Validation Loop
        val_running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:   
                tokens = batch["input_ids"].to(device)

                inputs = tokens[:, :-1].contiguous()
                targets = tokens[:, 1:].contiguous()

                if args.use_amp:
                    with autocast(device_type=args.device):
                        logits, loss = model(inputs, target=targets)
                else:
                    logits, loss = model(inputs, target=targets)

                val_running_loss += loss.item()
                
        avg_val_loss = val_running_loss / len(val_loader)
        val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()

        # Single Epoch Duration
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        

        # Save Epoch Results
        epoch_results.append(f"[Epoch {epoch+1:03d}] Time: {epoch_time:.4f}s | [Train] Loss: {avg_train_loss:.8f} Perplexity: {train_ppl:.2f} | [Val] Loss: {avg_val_loss:.8f} Perplexity: {val_ppl:.2f}")
        print(epoch_results[-1])
        
        # Min PPL 
        if val_ppl < min_perplexity:
            min_perplexity = val_ppl
            min_epoch = epoch + 1

        # Learning Rate Scheduler Step
        if scheduler and args.scheduler != 'linear': 
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_ppl)
            else:
                scheduler.step()


    # Test Loop (Final Evaluation)
    test_running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:   
            tokens = batch["input_ids"].to(device)

            inputs = tokens[:, :-1].contiguous()
            targets = tokens[:, 1:].contiguous()

            if args.use_amp:
                with autocast(device_type=args.device):
                    logits, loss = model(inputs, target=targets)
            else:
                logits, loss = model(inputs, target=targets)

            test_running_loss += loss.item()
                
    avg_test_loss = test_running_loss / len(test_loader)
    test_ppl = torch.exp(torch.tensor(avg_test_loss)).item()

    epoch_results.append(f"\n[Test] Loss: {avg_test_loss:.8f} Perplexity: {test_ppl:.2f}")
                
    epoch_results.append(f"\nAverage Epoch Time: {sum(epoch_times) / len(epoch_times):.4f}s")
    epoch_results.append(f"Min Perplexity: {min_perplexity:.4f} at Epoch {min_epoch}")

    return epoch_results
        

def accuracy(output, target, topk=(1,)):
    """Computes the top-1 and top-5 accuracy of the model."""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk] # [72.5, 91.3] - [top1, top5]


