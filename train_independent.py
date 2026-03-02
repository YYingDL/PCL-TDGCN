"""
Domain Adaptation Model Training Script
Author: YI Yang
Date: 2024
Purpose: Implementation of a domain adaptation framework for cross-subject EEG emotion recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import math
import os
import argparse
from typing import Tuple, List, Dict, Any
from sklearn.metrics import confusion_matrix

from model import DomainAdaptationModel, Discriminator
from Adversarial import DAANLoss
import utils
from utils import create_logger


def set_seed(seed: int = 20) -> None:
    """Set random seed for reproducibility across all random number generators."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(2)


class StepwiseLR_GRL:
    """Gradual learning rate scheduler with gradient reversal layer support."""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 init_lr: float = 0.01, gamma: float = 0.001,
                 decay_rate: float = 0.75, max_iter: int = 1000):
        """
        Initialize the learning rate scheduler.

        Args:
            optimizer: Optimizer instance
            init_lr: Initial learning rate
            gamma: Decay coefficient
            decay_rate: Decay rate exponent
            max_iter: Maximum number of iterations
        """
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iter = max_iter

    def get_lr(self) -> float:
        """Calculate current learning rate using polynomial decay."""
        lr = self.init_lr / (1.0 + self.gamma * (self.iter_num / self.max_iter)) ** (self.decay_rate)
        return lr

    def step(self) -> None:
        """Update learning rate for all parameter groups."""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group.setdefault('lr_mult', 1.)
            param_group['lr'] = lr * param_group['lr_mult']
        self.iter_num += 1


def test(test_loader: DataLoader, model: nn.Module,
         criterion: nn.Module, args: argparse.Namespace) -> Tuple[torch.Tensor, float, np.ndarray]:
    """
    Evaluate model performance on test dataset.

    Args:
        test_loader: Test data loader
        model: Model to evaluate
        criterion: Loss function
        args: Configuration parameters

    Returns:
        Average loss, accuracy, and confusion matrix
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for test_input, _, label in test_loader:
            test_input, label = test_input.to(args.device), label.to(args.device)
            output = model.target_predict(test_input)
            loss = criterion(output, label.view(-1))
            total_loss += loss.item()

            _, pred = torch.max(output, dim=1)
            correct += pred.eq(label.data.view_as(pred)).sum().item()

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)

    # Compute confusion matrix
    all_classes = np.arange(args.cls)
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=all_classes)

    return avg_loss, accuracy, conf_matrix


def initialize_source_banks(train_loader: DataLoader, model: nn.Module, args: argparse.Namespace) -> None:
    """Initialize source domain feature memory banks."""
    model.eval()
    with torch.no_grad():
        for tran_input, tran_idx, _ in train_loader:
            tran_input, tran_idx = tran_input.to(args.device), tran_idx.to(args.device)
            model.get_init_banks(tran_input, tran_idx)


def initialize_target_banks(train_loader: DataLoader, model: nn.Module, args: argparse.Namespace) -> None:
    """Initialize target domain feature memory banks."""
    model.eval()
    with torch.no_grad():
        for tran_input, tran_idx, _ in train_loader:
            tran_input, tran_idx = tran_input.to(args.device), tran_idx.to(args.device)
            model.get_init_banks_tgt(tran_input, tran_idx)


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross-entropy loss for regularization."""

    def __init__(self, classes: int = 3, epsilon: float = 0.0005):
        """
        Initialize label smoothing loss.

        Args:
            classes: Number of classes
            epsilon: Smoothing parameter
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for label smoothing loss.

        Args:
            input: Model predictions [batch_size, num_classes]
            target: Ground truth labels [batch_size]

        Returns:
            Smoothed cross-entropy loss
        """
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.epsilon / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.epsilon))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


def prepare_data(args: argparse.Namespace, test_id: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prepare source and target domain data for training.

    Args:
        args: Configuration parameters
        test_id: Target subject ID

    Returns:
        Target and source domain data dictionaries
    """
    # Load dataset
    data, label = utils.load_data(args.dataset)
    data_session, label_session = np.array(data[args.session]), np.array(label[args.session])

    # Target domain data (leave-one-subject-out)
    target_feature, target_label = data_session[test_id], label_session[test_id]

    # Source domain data (all other subjects)
    train_ids = list(range(15))
    train_ids.remove(test_id)
    source_features = [data_session[i] for i in train_ids]
    source_labels = [label_session[i] for i in train_ids]
    source_feature = np.vstack(source_features)
    source_label = np.vstack(source_labels)

    target_set = {'feature': target_feature, 'label': target_label}
    source_set = {'feature': source_feature, 'label': source_label}

    return target_set, source_set


def create_data_loaders(source_set: Dict[str, Any], target_set: Dict[str, Any],
                        args: argparse.Namespace) -> Tuple[Dict[str, DataLoader], int, int]:
    """
    Create PyTorch data loaders for source and target domains.

    Args:
        source_set: Source domain data
        target_set: Target domain data
        args: Configuration parameters

    Returns:
        Dictionary of data loaders, source and target sample counts
    """
    source_sample_num = source_set['feature'].shape[0]
    target_sample_num = target_set['feature'].shape[0]

    # Create TensorDatasets
    source_dataset = TensorDataset(
        torch.from_numpy(source_set['feature']).float(),
        torch.arange(source_sample_num).long(),
        torch.from_numpy(source_set['label']).long()
    )

    target_dataset = TensorDataset(
        torch.from_numpy(target_set['feature']).float(),
        torch.arange(target_sample_num).long(),
        torch.from_numpy(target_set['label']).long()
    )

    # Create DataLoaders
    source_loader = DataLoader(
        source_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return {
        "source_loader": source_loader,
        "target_loader": target_loader,
        "test_loader": test_loader
    }, source_sample_num, target_sample_num


def train_epoch(model: nn.Module, domain_discriminator: nn.Module,
                dann_loss: nn.Module, criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                data_loaders: Dict[str, DataLoader],
                epoch: int, args: argparse.Namespace) -> Tuple[float, float, Dict[str, float]]:
    """
    Train the model for one epoch.

    Args:
        model: Main domain adaptation model
        domain_discriminator: Domain discriminator for adversarial training
        dann_loss: Domain adversarial adaptation loss
        criterion: Classification loss
        optimizer: Model optimizer
        data_loaders: Data loader dictionary
        epoch: Current epoch number
        args: Configuration parameters

    Returns:
        Average training loss, accuracy, and detailed loss dictionary
    """
    model.train()
    dann_loss.train()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    loss_dict = {}

    # Create data iterators
    src_iter = iter(data_loaders["source_loader"])
    tar_iter = iter(data_loaders["target_loader"])
    num_batches = len(data_loaders["target_loader"].dataset) // args.batch_size
    for batch_idx in range(num_batches):
        # Get batch data
        src_data, src_idx, src_label = next(src_iter)
        tar_data, tar_idx, _ = next(tar_iter)

        # Move to device
        src_data, src_idx, src_label = (
            src_data.to(args.device),
            src_idx.to(args.device),
            src_label.to(args.device).view(-1)
        )
        tar_data, tar_idx = (
            tar_data.to(args.device),
            tar_idx.to(args.device)
        )

        # Forward pass
        (src_output_cls, src_feature, tar_output_cls, tar_feature,
         source_att, target_att, src_sim, tgt_sim, tgt_cluster_label,
         s2t_pro, t2s_pro, s2s_pro, t2t_pro) = model(
            src_data, tar_data, src_label, src_idx, tar_idx, epoch, args.epochs
        )

        # Classification loss
        cls_loss = criterion(src_output_cls, src_label)

        # Source domain loss with confidence filtering
        src_prob = F.softmax(src_output_cls, dim=1)
        max_prob, _ = src_prob.max(dim=1)
        mask = max_prob > 0.7

        if mask.any():
            filtered_prob = src_prob[mask]
            filtered_label = src_label[mask]
            source_loss = criterion(filtered_prob, filtered_label)
        else:
            source_loss = torch.tensor(0.0, device=src_prob.device)

        # Target domain classification loss
        target_loss = criterion(tgt_sim, tgt_cluster_label.long())

        # Domain adversarial loss
        global_transfer_loss = dann_loss(
            src_feature + 0.005 * torch.randn_like(src_feature).to(args.device),
            tar_feature + 0.005 * torch.randn_like(tar_feature).to(args.device),
            src_prob, F.softmax(tar_output_cls, dim=1)
        )

        # Cross-domain and within-domain consistency losses
        boost_factor = 2.0 * (2.0 / (1.0 + math.exp(-epoch / 1000)) - 1)

        s2t_entropy = -torch.sum(s2t_pro * torch.log(s2t_pro + 1e-10), dim=1).mean()
        t2s_entropy = -torch.sum(t2s_pro * torch.log(t2s_pro + 1e-10), dim=1).mean()
        cross_domain_loss = s2t_entropy + t2s_entropy

        s2s_entropy = -torch.sum(s2s_pro * torch.log(s2s_pro + 1e-10), dim=1).mean()
        t2t_entropy = -torch.sum(t2t_pro * torch.log(t2t_pro + 1e-10), dim=1).mean()
        in_domain_loss = s2s_entropy + t2t_entropy

        # Total loss
        loss = (cls_loss + global_transfer_loss + source_loss +
                boost_factor * target_loss + 0.2 * (cross_domain_loss + in_domain_loss))

        # Check for NaN values
        if torch.isnan(loss).any():
            print(f"Warning: NaN loss detected at epoch {epoch}, batch {batch_idx}")
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        _, pred = torch.max(src_prob, dim=1)
        total_correct += pred.eq(src_label).sum().item()
        total_samples += src_label.size(0)
        total_loss += loss.item()

        # Record detailed losses (first batch only)
        if batch_idx == 0:
            loss_dict = {
                'cls_loss': cls_loss.item(),
                'source_loss': source_loss.item(),
                'target_loss': target_loss.item(),
                'global_transfer_loss': global_transfer_loss.item(),
                'cross_domain_loss': cross_domain_loss.item(),
                'in_domain_loss': in_domain_loss.item(),
                'total_loss': loss.item()
            }

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples

    return avg_loss, accuracy, loss_dict


def main(test_id: int, writer: SummaryWriter, args: argparse.Namespace) -> Tuple[float, List, List, np.ndarray]:
    """
    Main training function for a single target subject.

    Args:
        test_id: Target subject ID
        writer: TensorBoard writer for logging
        args: Configuration parameters

    Returns:
        Best accuracy, source attention, target attention, and confusion matrix
    """
    set_seed(args.seed)

    # Prepare data
    target_set, source_set = prepare_data(args, test_id)
    data_loaders, source_sample_num, target_sample_num = create_data_loaders(source_set, target_set, args)

    # Initialize model
    model = DomainAdaptationModel(
        in_planes=args.in_planes,
        layers=args.layers,
        hidden_1=args.hidden_1,
        hidden_2=args.hidden_2,
        num_of_class=args.cls,
        device=args.device,
        source_num=source_sample_num,
        target_num=target_sample_num
    )
    domain_discriminator = Discriminator(args.hidden_2)

    # Loss functions
    criterion = LabelSmoothingCrossEntropy(classes=args.cls)
    dann_loss = DAANLoss(domain_discriminator, num_class=args.cls)

    # Move to device
    model = model.to(args.device)
    domain_discriminator = domain_discriminator.to(args.device)
    criterion = criterion.to(args.device)
    dann_loss = dann_loss.to(args.device)

    # Optimizer
    optimizer = torch.optim.RMSprop(
        list(model.parameters()) + list(domain_discriminator.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    lr_scheduler = StepwiseLR_GRL(
        optimizer,
        init_lr=args.lr,
        gamma=10,
        decay_rate=0.75,
        max_iter=args.epochs
    )

    # Initialize feature memory banks
    model.eval()
    initialize_source_banks(data_loaders["source_loader"], model, args)
    initialize_target_banks(data_loaders["target_loader"], model, args)

    # Training parameters
    best_acc = 0.0
    patience_counter = 0
    patience_limit = 40
    eval_interval = 10

    logger.info(f"Starting training for target subject {test_id}")

    for epoch in range(args.epochs):
        # Periodic evaluation
        if epoch % eval_interval == 0:
            test_loss, accuracy, conf_matrix = test(data_loaders["test_loader"], model, criterion, args)

            # Update best model
            if accuracy > best_acc:
                best_acc = accuracy
                patience_counter = 0

                # Save model
                model_dir = os.path.join(args.output_model_dir, args.dataset,
                                         'independent_re', str(args.session))
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f'CrossSub_no_nsal_{test_id}.pth')
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1

            # TensorBoard logging
            writer.add_scalar("test/loss", test_loss, epoch)
            writer.add_scalar("test/accuracy", accuracy, epoch)
            writer.add_scalar("test/best_accuracy", best_acc, epoch)

            logger.info(f"Epoch {epoch}: Test Accuracy={accuracy:.4f}, Best Accuracy={best_acc:.4f}")

            # Early stopping conditions
            if accuracy >= 1.0:
                logger.info("Perfect accuracy achieved, stopping training")
                break

            if patience_counter >= patience_limit:
                logger.info(f"Early stopping triggered after {patience_limit} epochs without improvement")
                break

        # Train for one epoch
        train_loss, train_acc, loss_dict = train_epoch(
            model, domain_discriminator, dann_loss, criterion,
            optimizer, data_loaders, epoch, args
        )

        # Training logging
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)

        for loss_name, loss_value in loss_dict.items():
            writer.add_scalar(f"train/{loss_name}", loss_value, epoch)

        # Learning rate adjustment
        lr_scheduler.step()

        # Print training progress
        if epoch % eval_interval == 0 and loss_dict:
            logger.info(f"Training Loss: Total={loss_dict['total_loss']:.4f}, "
                        f"Classification={loss_dict['cls_loss']:.4f}, "
                        f"Source={loss_dict['source_loss']:.4f}, "
                        f"Target={loss_dict['target_loss']:.4f}")

    # Final evaluation
    test_loss, final_acc, final_conf_matrix = test(data_loaders["test_loader"], model, criterion, args)
    logger.info(f"Target {test_id} Final Results: Accuracy={final_acc:.4f}, Best Accuracy={best_acc:.4f}")

    return best_acc, [], [], final_conf_matrix


if __name__ == "__main__":
    # Argument parser configuration
    parser = argparse.ArgumentParser(description='Transfer Learning')

    # Data parameters
    parser.add_argument('--dataset', type=str, nargs='?', default='seed3', help='select the dataset')
    parser.add_argument('--session', type=int, nargs='?', default='0', help='select the session')
    parser.add_argument('--cls', type=int, nargs='?', default=3, help="emotion classification")
    parser.add_argument('--in_planes', type=int, nargs='?', default=[5, 62], help="the size of input plane")
    parser.add_argument('--layers', type=int, nargs='?', default=2, help="DIAM squeeze ratio")
    parser.add_argument('--hidden_1', type=int, nargs='?', default=256, help="the size of hidden 1")
    parser.add_argument('--hidden_2', type=int, nargs='?', default=64, help="the size of hidden 2")
    parser.add_argument('--k', type=int, nargs='?', default=0.9, help="the size of k")

    parser.add_argument('--batch_size', type=int, nargs='?', default='48', help="batch_size")
    parser.add_argument('--epochs', type=int, nargs='?', default='1000', help="epochs")
    parser.add_argument('--lr', type=float, nargs='?', default='0.001', help="learning rate")
    parser.add_argument('--weight_decay', type=float, nargs='?', default='0.001', help="weight decay")
    parser.add_argument('--seed', type=int, nargs='?', default='200', help="random seed")
    parser.add_argument('--device', type=str, default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                        help='cuda or not')

    parser.add_argument('--output_log_dir', default='./train_log', type=str,
                        help='output path, subdir under output_root')
    parser.add_argument('--output_model_dir', default='./model', type=str,
                        help='output path, subdir under output_root')

    args = parser.parse_args()

    # Initialize logger
    logger = create_logger(args)
    logger.info(f"Training Configuration: {args}")

    # Cross-subject validation
    all_accuracies = []
    all_conf_matrices = []

    logger.info("Starting cross-subject domain adaptation training")

    for test_id in range(15):
        # Setup TensorBoard writer
        writer_dir = f"data/tensorboard/experiment_{args.dataset}/session_{args.session}_C3DA/target_{test_id}"
        writer = SummaryWriter(writer_dir)

        # Train for current target subject
        logger.info(f"Training for target subject {test_id}")
        source_ids = [i for i in range(15) if i != test_id]
        logger.info(f"Source subjects: {source_ids}, Target subject: {test_id}")

        best_acc, source_att, target_att, conf_matrix = main(test_id, writer, args)
        writer.close()

        # Collect results
        all_accuracies.append(best_acc)
        all_conf_matrices.append(conf_matrix)

        logger.info(f"Target subject {test_id} completed: Best Accuracy={best_acc:.4f}")

    # Aggregate results
    all_accuracies = np.array(all_accuracies)
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    total_conf_matrix = np.sum(all_conf_matrices, axis=0)

    logger.info("=" * 50)
    logger.info("Cross-Subject Validation Complete")
    logger.info(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"Individual Subject Accuracies: {all_accuracies}")
    logger.info(f"Aggregated Confusion Matrix:\n{total_conf_matrix}")
    logger.info("=" * 50)