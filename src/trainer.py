import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from .utils import apply_mask_flatten
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import subprocess

class Trainer:
    def __init__(self,
                 model,
                 train_set,
                 val_set=None,
                 test_set=None,
                 optimizer=None,
                 lr=1e-4,
                 max_len=10,
                 epoch_nums=10,
                 beta=0.5,
                 alpha=1e-7,
                 eval_steps=1000,
                 save_steps=500,
                 save_dir='./results',
                 max_saved_models=5,
                 project_name='my_project',
                 use_wandb=True,
                 use_tensorboard=False,
                 device='cpu',
                 lr_warmup_steps=1000,
                 lr_decay_factor=0.9,
                 lr_decay_steps=10000,
                 grad_clip_value=None):
        """
        Initialize the Trainer class with learning rate scheduling options.

        Arguments:
        lr_warmup_steps -- Number of steps for learning rate warmup
        lr_decay_factor -- The factor by which the learning rate decays
        lr_decay_steps -- Number of steps before applying decay
        """
        self.model = model
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.optimizer = optimizer if optimizer is not None else optim.Adam(model.parameters(), lr=lr)
        self.lr = lr
        self.max_len = max_len
        self.epoch_nums = epoch_nums
        self.beta = beta
        self.alpha = alpha
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.save_dir = save_dir
        self.max_saved_models = max_saved_models
        self.project_name = project_name
        self.use_wandb = use_wandb
        self.device = device
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            log_dir = os.path.join(self.save_dir, 'tensorboard_logs')
            self.writer = SummaryWriter(log_dir=log_dir)
            abs_log_dir = os.path.abspath(log_dir)
            print(f"[TensorBoard] Log path: {abs_log_dir}")
            print(f"[TensorBoard] Start command: tensorboard --logdir {abs_log_dir} --port 6006")
            print(f"[TensorBoard] Open in your local browser: http://localhost:6006")
            print(f"[TensorBoard] SSH forwarding command: ssh -L 6006:localhost:6006 your_user@your_server")

        # Learning rate warmup and decay
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_steps = lr_decay_steps

        # Loss functions
        self.mse_loss = nn.MSELoss()

        # Gradient clipping value
        self.grad_clip_value = grad_clip_value

        # Logs to store training statistics
        self.train_log = []

        # Initialize WandB
        if self.use_wandb:
            wandb.init(project=self.project_name, config={
                "lr": lr,
                "max_len": max_len,
                "epoch_nums": epoch_nums,
                "beta": beta,
                "eval_steps": eval_steps,
                "save_steps": save_steps,
                "save_dir": save_dir,
                "max_saved_models": max_saved_models,
                "lr_warmup_steps": lr_warmup_steps,
                "lr_decay_factor": lr_decay_factor,
                "lr_decay_steps": lr_decay_steps
            })
            print(f"[WandB] Started logging to project: {self.project_name}")

        # Create the save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Get keys
        for data, _ in self.train_set:
            self.keys = list(data.keys())
            break

    def fit(self, explain_interval=100):
        """
        Train the model on the training set and evaluate on the validation set with learning rate warmup and decay.
        Logs are saved to TensorBoard if use_tensorboard is True, and to Weights & Biases if use_wandb is True.

        Args:
            explain_interval (int): run explainability every this many epochs (default: 10)
        """

        # Train the model
        self.model.train()
        print("😀😀 Start Training 😀😀")
        step = 0

        # Track learning rate and adjust
        for epoch in tqdm(range(self.epoch_nums), desc='Epoch'):
            epoch_step = 0
            total_loss = 0
            total_mse_loss = 0
            total_kl_loss = 0

            # Track loss for each key
            total_key_mse_losses = {key: 0.0 for key in self.keys}
            total_key_kl_losses = {key: 0.0 for key in self.keys}

            for inputs_dict, labels_dicts in self.train_set:
                self.beta = 0.5 * (1 - np.exp(-self.alpha * step))
                self.optimizer.zero_grad()

                # Learning rate warmup
                if step < self.lr_warmup_steps:
                    lr_scale = (step + 1) / self.lr_warmup_steps
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr * lr_scale
                # Learning rate decay
                elif step % self.lr_decay_steps == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= self.lr_decay_factor

                # Get current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']

                # Forward pass
                out_dict, mu_dict, logvar_dict, _ = self.model(inputs_dict)

                loss_mse = torch.tensor(0.0, requires_grad=True, device=self.device)
                loss_kl = torch.tensor(0.0, requires_grad=True, device=self.device)

                # Calculate per-key loss
                for key in out_dict:
                    mask = labels_dicts[key]["mask"]
                    out_masked = apply_mask_flatten(value=out_dict[key], mask=mask)
                    labels_masked = apply_mask_flatten(value=labels_dicts[key]["data"], mask=mask)

                    key_loss_mse = F.mse_loss(out_masked, labels_masked, reduction='mean')
                    if self.beta > 0:
                        mu_masked = apply_mask_flatten(value=mu_dict[key], mask=mask)
                        logvar_masked = apply_mask_flatten(value=logvar_dict[key], mask=mask)
                        key_loss_kl = -0.5 * torch.sum(
                            1 + logvar_masked - mu_masked**2 - torch.exp(logvar_masked)
                        ) / (logvar_masked.size(0))
                    else:
                        key_loss_kl = torch.zeros_like(key_loss_mse, device=key_loss_mse.device)

                    loss_mse = loss_mse + key_loss_mse
                    loss_kl = loss_kl + key_loss_kl

                    total_key_mse_losses[key] += key_loss_mse.item()
                    total_key_kl_losses[key] += key_loss_kl.item()

                loss_mse = loss_mse / len(out_dict)
                loss_kl = loss_kl / len(out_dict)

                # Weighted sum of losses
                total_loss = self.beta * loss_kl + (1 - self.beta) * loss_mse

                # Backward pass
                total_loss.backward()
                if self.grad_clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
                self.optimizer.step()

                total_mse_loss += loss_mse.item()
                total_kl_loss += loss_kl.item()
                step += 1
                epoch_step += 1

                # Log training data for each step
                log_data = {
                    'step': step,
                    'epoch': epoch + 1,
                    'total_loss': total_loss.item(),
                    'mse_loss': loss_mse.item(),
                    'kl_loss': loss_kl.item(),
                    'lr': current_lr,
                    'beta': self.beta,
                    'epoch_avg_mse_loss': total_mse_loss / epoch_step,
                    'epoch_avg_kl_loss': total_kl_loss / epoch_step,
                }
                for key in out_dict:
                    log_data[f"epoch_avg_{key}_mse_loss"] = total_key_mse_losses[key] / epoch_step
                    log_data[f"epoch_avg_{key}_kl_loss"] = total_key_kl_losses[key] / epoch_step

                self.train_log.append(log_data)
                if self.use_wandb:
                    wandb.log(log_data)
                if self.use_tensorboard:
                    it = step + epoch * len(self.train_set)
                    self.writer.add_scalar('Loss/total_loss', total_loss.item(), it)
                    self.writer.add_scalar('Loss/mse_loss', loss_mse.item(), it)
                    self.writer.add_scalar('Loss/kl_loss', loss_kl.item(), it)
                    self.writer.add_scalar('LearningRate', current_lr, it)

                if step % self.eval_steps == 0:
                    print(f"Epoch [{epoch+1}/{self.epoch_nums}], Step [{step}], "
                          f"Total Loss: {total_loss.item():.4f}, "
                          f"MSE Loss: {total_mse_loss/epoch_step:.4f}, "
                          f"KL Loss: {total_kl_loss/epoch_step:.4f}, "
                          f"LR: {current_lr:.6f}")
                    for key in out_dict:
                        print(f"{key}: MSE {total_key_mse_losses[key]/step:.4f}, "
                              f"KL {total_key_kl_losses[key]/step:.4f}")
                    if self.val_set is not None:
                        _ = self.evaluate_on_val_set(epoch, step)

                if step % self.save_steps == 0:
                    self.save_model(epoch, step)

            # After each epoch, optionally run explainability
            if (epoch + 1) % explain_interval == 0:
                explain_path = os.path.join(
                    self.save_dir, "exp", f"epoch_{epoch+1}"
                )
                self.explainability(save_path=explain_path)
                if self.use_wandb:
                    # Upload all PNGs under explain_path
                    images = []
                    for root, _, files in os.walk(explain_path):
                        for fname in files:
                            if fname.endswith('.png'):
                                fullp = os.path.join(root, fname)
                                images.append(wandb.Image(fullp, caption=fname))
                    wandb.log({f'explain_images_epoch_{epoch+1}': images}, step=step)

        # Final validation and checkpoint
        val_final_loss = self.evaluate_on_val_set(epoch, step)
        self.save_model(epoch, step, final=True)
        if self.use_tensorboard:
            self.writer.close()

        return {'model': self.model, 'val_final_loss': val_final_loss}


    def save_model(self, epoch, step, final=False):
        """
        Save the model checkpoint and keep only the most recent 'max_saved_models' models.
        """
        model_filename = f"model_step_{step}.pt"
        model_path = os.path.join(self.save_dir, model_filename)

        # Save model weights
        torch.save(self.model.state_dict(), model_path)

        # Manage saved models: Keep only the most recent 'max_saved_models' models
        saved_models = sorted(os.listdir(self.save_dir), key=lambda x: os.path.getmtime(os.path.join(self.save_dir, x)))
        if len(saved_models) > self.max_saved_models:
            os.remove(os.path.join(self.save_dir, saved_models[0]))  # Remove the oldest model

        print(f"Model saved: {model_path}")

    def evaluate_on_val_set(self, epoch, step):
        """
        Evaluate the model on the validation set and log to TensorBoard.
        """
        self.model.eval()  # Set the model to evaluation mode

        # Evaluate the model on the validation set (similar to training)
        total_val_loss = 0
        total_mse_loss = 0
        total_kl_loss = 0
        with torch.no_grad():
            for inputs_dict, labels_dicts in self.val_set:
                out_dict, mu_dict, logvar_dict, _ = self.model(inputs_dict)

                loss_mse = torch.tensor(0.0)
                loss_kl = torch.tensor(0.0)

                for key, _ in out_dict.items():
                    mask = labels_dicts[key]["mask"]
                    out_masked = apply_mask_flatten(value=out_dict[key], mask=mask)
                    labels_masked = apply_mask_flatten(value=labels_dicts[key]["data"], mask=mask)
                    key_loss_mse = F.mse_loss(out_masked, labels_masked, reduction='mean')

                    if self.beta > 0:
                        mu_masked = apply_mask_flatten(value=mu_dict[key], mask=mask)
                        logvar_masked = apply_mask_flatten(value=logvar_dict[key], mask=mask)
                        key_loss_kl = - 0.5 * torch.sum(1 + logvar_masked - mu_masked**2 - torch.exp(logvar_masked)) / (logvar_masked.size(0))
                    else:
                        key_loss_kl = torch.zeros_like(key_loss_mse, device=key_loss_mse.device)

                    loss_mse += key_loss_mse.cpu()
                    loss_kl += key_loss_kl.cpu()

                loss_mse = loss_mse / len(out_dict)
                loss_kl = loss_kl / len(out_dict)

                total_val_loss += self.beta * loss_kl + (1 - self.beta) * loss_mse
                total_mse_loss += loss_mse.item()
                total_kl_loss += loss_kl.item()

        avg_val_loss = total_val_loss / len(self.val_set)
        avg_mse_loss = total_mse_loss / len(self.val_set)
        avg_kl_loss = total_kl_loss / len(self.val_set)

        # Print the validation loss
        print(f"Validation Loss: {avg_val_loss:.4f}, "
            f"MSE Loss: {avg_mse_loss:.4f}, "
            f"KL Loss: {avg_kl_loss:.4f}")

        # Log validation loss and metrics to wandb
        val_log_data = {
            'epoch': epoch + 1,
            'step': step,
            'val_loss': avg_val_loss,
            'val_mse_loss': avg_mse_loss,
            'val_kl_loss': avg_kl_loss,
        }
        if self.use_wandb:
            wandb.log(val_log_data)

        # Log the same data to TensorBoard
        if self.use_tensorboard:
            self.writer.add_scalar('Validation/Loss', avg_val_loss, step)
            self.writer.add_scalar('Validation/MSE_Loss', avg_mse_loss, step)
            self.writer.add_scalar('Validation/KL_Loss', avg_kl_loss, step)

        # Switch the model back to training mode
        self.model.train()

        return avg_val_loss

    def evaluate_on_test_set(self):
        """
        Evaluate the model on the test set.
        """
        self.model.eval()
        total_test_loss = 0
        total_mse_loss = 0
        total_kl_loss = 0
        with torch.no_grad():
            for inputs_dict, labels_dicts in self.test_set:
                out_dict, mu_dict, logvar_dict = self.model(inputs_dict)

                loss_mse = torch.tensor(0.0)
                loss_kl = torch.tensor(0.0)

                for key, _ in out_dict.items():
                    mask = labels_dicts[key]["mask"]
                    out_masked = apply_mask_flatten(value=out_dict[key], mask=mask)
                    labels_masked = apply_mask_flatten(value=labels_dicts[key]["data"], mask=mask)
                    mu_masked = apply_mask_flatten(value=mu_dict[key], mask=mask)
                    logvar_masked = apply_mask_flatten(value=logvar_dict[key], mask=mask)

                    key_loss_mse = F.mse_loss(out_masked, labels_masked, reduction='mean')
                    key_loss_kl = 0.5 * torch.sum(1 + logvar_masked - mu_masked**2 - torch.exp(logvar_masked)) / (logvar_masked.size(0))

                    loss_mse += key_loss_mse
                    loss_kl += key_loss_kl

                loss_mse = loss_mse / len(out_dict)
                loss_kl = loss_kl / len(out_dict)

                total_test_loss += self.beta * loss_kl + (1 - self.beta) * loss_mse
                total_mse_loss += loss_mse.item()
                total_kl_loss += loss_kl.item()

        avg_test_loss = total_test_loss / len(self.test_set)
        avg_mse_loss = total_mse_loss / len(self.test_set)
        avg_kl_loss = total_kl_loss / len(self.test_set)

        print(f"Test Loss: {avg_test_loss:.4f}, "
              f"MSE Loss: {avg_mse_loss:.4f}, "
              f"KL Loss: {avg_kl_loss:.4f}")
        return avg_test_loss




    def explainability(self,
                    save_path='./results/explanation/',
                    pca_sample=0.02,
                    tsne_sample=0.01):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        """
        对验证集做可解释性分析：
        1. 收集所有时间序列输出和末状态输出，分别存入 seq_dict 和 spot_dict；
        2. 对 spot_dict 中的输出按 pca_sample 比例采样，用 PCA 降到 2 维并绘图；
        3. 对 seq_dict 中 'pre_rnn' 和 'post_rnn' 两个序列同时按 tsne_sample 比例采样，确保同一 sample 下的 pre 与 post 成对处理；
            对每个 sample，分别对 pre 和 post 序列使用 TSNE（exact 方法）降到 6 维，并在一张 2x3 子图中绘制：
            上排为 pre_rnn 三视角（dims 1-2,3-4,5-6），下排为 post_rnn 三视角；
        4. 保存所有图到磁盘，目录结构清晰，命名系统化；
        5. 输出每部分实际采样数量，并展示进度条。
        """
        # 1. 收集所有序列和末状态
        seq_dict, spot_dict = {}, {}
        self.model.eval()
        with torch.no_grad():
            for inputs_dict, _ in tqdm(self.val_set, desc='Collecting features'):
                _, _, _, log_dict = self.model(inputs_dict)
                for key, tensor in log_dict.items():  # tensor.shape == [batch, seq_len, hidden_dim]
                    if key not in seq_dict:
                        seq_dict[key] = []
                        spot_dict[key] = []
                    for seq in tensor:
                        seq_dict[key].append(seq.cpu().detach().numpy())
                    last = tensor[:, -1, :].cpu().detach().numpy()
                    for vec in last:
                        spot_dict[key].append(vec)

        # 准备保存目录
        pca_dir = os.path.join(save_path, 'pca')
        tsne_dir = os.path.join(save_path, 'tsne')
        os.makedirs(pca_dir, exist_ok=True)
        os.makedirs(tsne_dir, exist_ok=True)

        # 2. PCA 可视化 spot_dict
        plt.figure(figsize=(6,6), dpi=300)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (key, vecs) in enumerate(tqdm(spot_dict.items(), desc='PCA Sampling')):
            vecs = np.array(vecs)
            n = len(vecs)
            m = max(1, int(n * pca_sample))
            print(f"[PCA] Key '{key}': sampled {m} of {n} vectors.")
            idx = np.random.choice(n, m, replace=False)
            sampled = vecs[idx]
            pca = PCA(n_components=2, random_state=0)
            emb = pca.fit_transform(sampled)
            plt.scatter(emb[:,0], emb[:,1], label=key, color=colors[i % len(colors)], s=10, alpha=0.7)
        plt.xlabel('PCA 1', fontsize=12)
        plt.ylabel('PCA 2', fontsize=12)
        plt.title('Spot Features PCA Projection', fontsize=14)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(pca_dir, 'spot_pca_2d.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 3. TSNE 可视化 seq_dict 中 'pre_rnn' 与 'post_rnn' 同时处理
        # 确保两个 key 存在且样本数量对齐
        for key in ['pre_rnn', 'post_rnn']:
            assert key in seq_dict, f"seq_dict 中缺少 {key}"
        total_pre = len(seq_dict['pre_rnn'])
        total_post = len(seq_dict['post_rnn'])
        total = min(total_pre, total_post)
        m = max(1, int(total * tsne_sample))
        print(f"[TSNE] Sampling {m} paired sequences from {total} total samples.")
        sampled_idx = np.random.choice(total, m, replace=False)

        for idx in tqdm(sampled_idx, desc='TSNE paired trajectories'):
            pre_seq = seq_dict['pre_rnn'][idx]
            post_seq = seq_dict['post_rnn'][idx]
            seq_len, hid_dim = pre_seq.shape
            assert post_seq.shape == (seq_len, hid_dim), "pre_rnn 与 post_rnn 序列长度或维度不匹配"

            # 对 pre 和 post 分别降维
            tsne_pre = TSNE(n_components=6, method='exact', random_state=0)
            emb_pre = tsne_pre.fit_transform(pre_seq)
            tsne_post = TSNE(n_components=6, method='exact', random_state=0)
            emb_post = tsne_post.fit_transform(post_seq)

            # 绘制 2x3 子图：上排 pre， 下排 post
            fig, axs = plt.subplots(2, 3, figsize=(15,10), dpi=300)
            views = [(0,1), (2,3), (4,5)]
            for col, (dx, dy) in enumerate(views):
                axs[0, col].plot(emb_pre[:,dx], emb_pre[:,dy], '-o', markersize=3, linewidth=1)
                axs[0, col].set_title(f'Pre-RNN TSNE dims {dx+1}-{dy+1}', fontsize=14)
                axs[0, col].set_xlabel(f'Dim {dx+1}', fontsize=12)
                axs[0, col].set_ylabel(f'Dim {dy+1}', fontsize=12)

                axs[1, col].plot(emb_post[:,dx], emb_post[:,dy], '-o', markersize=3, linewidth=1)
                axs[1, col].set_title(f'Post-RNN TSNE dims {dx+1}-{dy+1}', fontsize=14)
                axs[1, col].set_xlabel(f'Dim {dx+1}', fontsize=12)
                axs[1, col].set_ylabel(f'Dim {dy+1}', fontsize=12)

            fig.suptitle(f'TSNE Trajectories for Sample #{idx}', fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fname = f'tsne_paired_trajectory_sample_{idx}.png'
            fig.savefig(os.path.join(tsne_dir, fname), dpi=300, bbox_inches='tight')
            plt.close(fig)




    def get_train_log(self):
        """
        Return the training log collected during the training process.
        """
        return self.train_log