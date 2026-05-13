import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from torchvision import transforms

# sigmoid_focal_loss removed - using weighted BCEWithLogitsLoss instead

from pathlib import Path
from tqdm import tqdm
import json
# from monai.losses import DiceCELoss

from finetune_utils.load_config import get_config
from finetune_utils.load_logger import Logger
from finetune_utils.load_checkpoint import get_sam_vit_t
from finetune_utils.datasets import SAMDataset, JointAugmentation
from finetune_utils.loss import DiceLoss, batch_iou
from finetune_utils.visualization import overlay_mask_on_image
from finetune_utils.save_checkpoint import save_checkpoint
from finetune_utils.schedular import LinearWarmup

torch.backends.cudnn.benchmark = True

def main(args, train_augmentation=None):
    # Assert that a CUDA device is available
    assert torch.cuda.is_available(), "CUDA is not available."

    # Create dataset and dataloader for training and validation
    # Note: augmentation is only applied to training set, not validation
    train_dataset = SAMDataset(root_dir=args.dataset.train_dataset, transform=[transform_img, transform_mask], max_bbox_shift=args.dataset.max_bbox_shift, augmentation=train_augmentation)
    val_dataset = SAMDataset(root_dir=args.dataset.val_dataset, transform=[transform_img, transform_mask], max_bbox_shift=args.dataset.max_bbox_shift, augmentation=None)
    train_loader = DataLoader(train_dataset, batch_size=args.train.batch_size, num_workers=args.dataset.num_workers, shuffle=True, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=args.train.batch_size, num_workers=args.dataset.num_workers, shuffle=False, pin_memory=True, persistent_workers=True)

    # Define checkpoint and saving paths
    checkpoint_path = Path(args.model.checkpoint_path)
    save_path = Path(args.model.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Initialize the logger
    logger = Logger(save_path / 'training.log').get_logger()

    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()

    # Load the MobileSAM checkpoint and move it to CUDA
    # TO DO: resume checkpoint from last.pth
    model = get_sam_vit_t(checkpoint=checkpoint_path, resume=args.train.resume).cuda()
    
    # Conditionally freeze layers based on args
    for param in model.image_encoder.parameters():
        param.requires_grad = not args.freeze.freeze_image_encoder
    for param in model.prompt_encoder.parameters():
        param.requires_grad = not args.freeze.freeze_prompt_encoder
    for param in model.mask_decoder.parameters():
        param.requires_grad = not args.freeze.freeze_mask_decoder

    # Initialize optimizer and loss function
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.train.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train.epochs * len(train_dataset))
    warmup_scheduler = LinearWarmup(optimizer, warmup_period=args.train.warmup_step)

    criterion_MSE = nn.MSELoss()
    criterion_Dice = DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')

    # Determine loss type from config (default: balanced)
    loss_type = getattr(getattr(args, 'loss', None), 'type', 'balanced') if hasattr(args, 'loss') else 'balanced'
    logger.info(f"Using loss type: {loss_type}")

    if loss_type == 'no_weight':
        # No class weighting - standard BCE
        criterion_BCE = nn.BCEWithLogitsLoss(reduction='mean')
    elif loss_type == 'dice_only':
        # Dice only - BCE not used but create dummy
        criterion_BCE = nn.BCEWithLogitsLoss(reduction='mean')
    else:
        # Default balanced: pos_weight=1.3
        pos_weight = torch.tensor([1.3]).cuda()
        criterion_BCE = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')

    # Initialize TensorBoard writer for logging
    writer = SummaryWriter()

    # Initialize the best validation loss variable
    best_val_loss = float('inf')

    # Training history for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'epochs': [],
        'val_epochs': []
    }

    # Main training loop
    for epoch in range(args.train.epochs):
        # Train for one epoch
        train_loss = train_epoch(train_loader, model, optimizer, criterion_MSE, criterion_Dice, criterion_BCE, epoch, writer, scaler, lr_scheduler, warmup_scheduler, loss_type)
        logger.info(f"Epoch {epoch+1}/{args.train.epochs}, Train Loss: {train_loss:.4f}")
        history['train_loss'].append(train_loss)
        history['epochs'].append(epoch + 1)

        # Validate and save the model at specified intervals
        if (epoch + 1) % args.train.val_freq == 0:
            val_loss = val_epoch(val_loader, model, criterion_MSE, criterion_Dice, criterion_BCE, epoch, writer, scaler, loss_type)
            logger.info(f"Epoch {epoch+1}/{args.train.epochs}, Val Loss: {val_loss:.4f}")
            history['val_loss'].append(val_loss)
            history['val_epochs'].append(epoch + 1)

            # Save the best model based on validation loss
            # the best model could be used like the original MobileSAM checkpoint without any modification
            is_best = val_loss < best_val_loss
            save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, is_best, save_path)
            if is_best:
                best_val_loss = val_loss

        # Save training history after each epoch
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)

    logger.info(f"Training completed. Best val loss: {best_val_loss:.4f}")

def train_epoch(dataloader, model, optimizer, criterion_MSE, criterion_Dice, criterion_BCE, epoch, writer, scaler, lr_scheduler, warmup_scheduler, loss_type='balanced'):
    """Main training function."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc="Training", total=num_batches)

    for batch_idx, (image, mask, bbox) in enumerate(progress_bar):
        # Move input and target data to the GPU
        image, mask, bbox = image.cuda(non_blocking=True), mask.cuda(non_blocking=True), bbox.cuda(non_blocking=True)

        # Forward pass with mixed precision
        with autocast(enabled=args.train.bf16, dtype=torch.bfloat16):
            pred_mask, pred_IOU = model(image, bbox)
            iou = batch_iou(mask, torch.sigmoid(pred_mask))

            # Calculate loss based on loss_type
            loss_dice = criterion_Dice(pred_mask, mask)
            loss_mse = criterion_MSE(pred_IOU, iou)

            if loss_type == 'dice_only':
                # Pure Dice Loss + MSE for IoU prediction
                loss = loss_dice + loss_mse
            elif loss_type == 'no_weight':
                # No class weighting: BCE + Dice + MSE
                loss_bce = criterion_BCE(pred_mask, mask)
                loss = loss_bce * 5 + loss_dice * 3 + loss_mse
            else:
                # Default balanced: BCE(weighted) + Dice + MSE
                loss_bce = criterion_BCE(pred_mask, mask)
                loss = loss_bce * 5 + loss_dice * 3 + loss_mse

        # Backward pass and update model parameters
        scaler.scale(loss).backward()
        if batch_idx % args.train.gradient_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            with warmup_scheduler.dampening():
                lr_scheduler.step()
        

        # Accumulate the loss for logging
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # Calculate average training loss for the epoch
    average_loss = total_loss / num_batches
            
    # Log the training loss to TensorBoard
    for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Learning_rate/group_{i}', param_group['lr'], epoch)
    writer.add_scalar('Training loss', average_loss, epoch)
    
    return average_loss

def val_epoch(dataloader, model, criterion_MSE, criterion_Dice, criterion_BCE, epoch, writer, scaler, loss_type='balanced'):
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, desc="Validating", total=num_batches)

    # Evaluation mode: no gradients needed
    with torch.no_grad():
        for batch_idx, (image, mask, bbox) in enumerate(progress_bar):
            # Move input and target data to the GPU
            image, mask, bbox = image.cuda(), mask.cuda(), bbox.cuda()

            # Forward pass
            pred_mask, pred_IOU = model(image, bbox)
            iou = batch_iou(mask, torch.sigmoid(pred_mask))

            # Calculate loss based on loss_type (consistent with training)
            loss_dice = criterion_Dice(pred_mask, mask)
            loss_mse = criterion_MSE(pred_IOU, iou)

            if loss_type == 'dice_only':
                loss = loss_dice + loss_mse
            elif loss_type == 'no_weight':
                loss_bce = criterion_BCE(pred_mask, mask)
                loss = loss_bce * 5 + loss_dice * 3 + loss_mse
            else:
                loss_bce = criterion_BCE(pred_mask, mask)
                loss = loss_bce * 5 + loss_dice * 3 + loss_mse

            # Accumulate the loss for logging
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            if args.visual.status:
                vis_image = image[0]
                vis_mask = pred_mask[0]
                vis_bbox = bbox[0]
                vis_mask = torch.sigmoid(vis_mask)
                mean = torch.tensor(MEAN, device=vis_image.device)
                std = torch.tensor(STD, device=vis_image.device)
                vis_image = vis_image * std[:, None, None] + mean[:, None, None]
                overlay_mask_on_image(vis_image, vis_mask, vis_bbox, threshold=args.visual.IOU_threshold, save_dir=args.visual.save_path, info=(epoch, batch_idx))

        # Calculate average validation loss for the epoch
        average_loss = total_loss / num_batches
            
     # Log the validation loss to TensorBoard
    writer.add_scalar('Val loss', average_loss, epoch)

    return average_loss


if __name__ == '__main__':
    # Load configuration settings
    args = get_config()

    # Define the desired image size
    IMAGE_SIZE = (args.model.image_size, args.model.image_size)

    # Define image and mask transformations
    # The normalization is the same in mobile_sam\modeling\sam.py
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    transform_img = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
    ])
    transform_mask = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    # Data augmentation for training (geometric + color transforms)
    # Set augmentation=None to disable, or customize parameters
    train_augmentation = JointAugmentation(
        horizontal_flip=True,
        vertical_flip=False,
        rotation_degrees=15,
        color_jitter=True,
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.05
    )

    main(args, train_augmentation)