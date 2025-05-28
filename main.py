import torch
from src.model import VASE, SensorVAE
from src.trainer import Trainer
from data.load_kitti_oxts import get_data
from src.dataloader import create_dataloader
from src.baselines import BaseModel, Baseline
import argparse

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    # 命令行参数解析部分
    parser = argparse.ArgumentParser(description="Train the VASE model with configurable parameters")

    parser.add_argument('--mask_ratio', type=float, default=0.1, help="Ratio for masking the GPS/IMU data")
    parser.add_argument('--latent_dim', type=int, default=128, help="Latent dimension of the VAE")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension of the VAE")
    parser.add_argument('--model', type=str, default='VASE', choices=['VASE', 'mlp', 'rnn', 'lstm'], help="Activation function used in the network")
    parser.add_argument('--activation', type=str, default='leaky', choices=['leaky', 'relu', 'tanh'], help="Activation function used in the network")
    parser.add_argument('--proj_name', type=str, default='my_project', help="Set Project Name")
    parser.add_argument('--en_layers', type=int, default=2, help="Number of encoder layers")
    parser.add_argument('--de_layers', type=int, default=2, help="Number of decoder layers")
    parser.add_argument('--dropout_rate', type=float, default=0.5, help="Dropout rate used in the model")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in the VASE model")
    parser.add_argument('--rnn', type=str, default='lstm', choices=['lstm', 'rnn'], help="RNN type used in the VASE model")
    parser.add_argument('--fuse', type=str, default='prod', choices=['sum', 'mean', 'prod'], help="Fusion method in the VASE model")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size used for training and testing")
    parser.add_argument('--eval_steps', type=int, default=300, help="Steps interval for evaluation during training")
    parser.add_argument('--save_steps', type=int, default=2000, help="Steps interval for saving the model")
    parser.add_argument('--use_skip_connection', action='store_true', help="Use skip connections if specified")
    parser.add_argument('--use_wandb', action='store_true', help="Use Weights & Biases for logging if specified")
    parser.add_argument('--use_tensorboard', action='store_true', help="Use TensorBoard for logging if specified")
    parser.add_argument('--epoch_nums', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--beta', type=float, default=0.02, help="Beta value for VAE loss function")
    parser.add_argument('--lr_warmup_steps', type=int, default=200, help="Number of warmup steps for learning rate")
    parser.add_argument('--lr_decay_factor', type=float, default=0.98, help="Factor for learning rate decay")
    parser.add_argument('--lr_decay_steps', type=int, default=500, help="Steps interval for learning rate decay")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument('--grad_clip_value', type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument('--do_explanation', action='store_true', help="Do explanation process or NOT")


    args = parser.parse_args()

    mask_ratio = args.mask_ratio
    latent_dim = args.latent_dim
    hidden_dim = args.hidden_dim

    sensor_vaes = {}
    train_inputs = get_data(split='train', gps_mask_ratio=mask_ratio, imu_mask_ratio=mask_ratio)
    train_labels = get_data(split='train', gps_mask_ratio=0.0, imu_mask_ratio=0.0)
    test_inputs = get_data(split='test', gps_mask_ratio=mask_ratio, imu_mask_ratio=mask_ratio)
    test_labels = get_data(split='test', gps_mask_ratio=0.0, imu_mask_ratio=0.0)

    for key, value in train_inputs.items():
        input_dim = value[0]['data'].size(-1)
        if args.model == 'VASE':
            sensor_vaes[key] = SensorVAE(input_dim=input_dim, latent_dim=latent_dim, hidden_dim=hidden_dim, activation=args.activation,
                                        en_layers=args.en_layers, de_layers=args.de_layers, use_skip_connection=args.use_skip_connection,
                                        dropout_rate=args.dropout_rate)
        else:
            sensor_vaes[key] = BaseModel(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=args.en_layers, rnn=args.model)

    if args.model == 'VASE':
        model = VASE(sensor_vaes=sensor_vaes, latent_dim=latent_dim, hidden_dim=hidden_dim, num_layers=args.num_layers, rnn=args.rnn, fuse=args.fuse)
        beta = args.beta
        alpha = 1e-7
    else:
        beta = 0.0
        alpha = 0.0
        model = Baseline(base_models=sensor_vaes)

    print("👌👌 Model built successfully 👌👌")

    device = "cuda"
    model.to(device)
    train_loader = create_dataloader(x=train_inputs, label=train_labels, max_length=512, stride=1, drift=1, batch_size=args.batch_size, device=device)
    test_loader = create_dataloader(x=test_inputs, label=test_labels, max_length=512, stride=1, drift=1, batch_size=args.batch_size, device=device)
    print("✅✅ Dataloader built successfully ✅✅")

    trainer = Trainer(model=model,
                      train_set=train_loader,
                      val_set=test_loader,
                      eval_steps=args.eval_steps,
                      save_steps=args.save_steps,
                      use_wandb=args.use_wandb,
                      use_tensorboard=args.use_tensorboard,
                      device=device,
                      epoch_nums=args.epoch_nums,
                      beta=beta,
                      alpha=alpha,
                      project_name=args.proj_name,
                      lr_warmup_steps=args.lr_warmup_steps,
                      lr_decay_factor=args.lr_decay_factor,
                      lr_decay_steps=args.lr_decay_steps,
                      lr=args.lr,
                      grad_clip_value=args.grad_clip_value)

    trainer.fit()

    print("Model Training Finished!!")

    if args.do_explanation:
        print("Start Explanation")
        trainer.explainability()

        pass



if __name__ == "__main__":
    main()