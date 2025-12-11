import model
import os
import torch
from torch.utils.data import DataLoader
import data
from torch import Tensor, nn
import matplotlib.pyplot as plt
import numpy as np


def to_cuda(x: Tensor | nn.Module) -> Tensor | nn.Module:
    if torch.cuda.is_available():
        return x.cuda()
    return x


class LossSaver:
    def __init__(self, names: list[str]):
        self.root_dir = os.path.join(os.getcwd(), "losses")
        os.makedirs(self.root_dir, exist_ok=True)
        self.losses = {}
        for name in names:
            self.losses[name] = []

    def save_loss(self, file_name: str):
        for key in self.losses:
            plt.plot(
                list(range(len(self.losses[key]))), self.losses[key], label=key
            )
        plt.legend()
        plt.savefig(os.path.join(self.root_dir, file_name), bbox_inches="tight")
        plt.close()

    def __call__(self, values: list[Tensor]):
        for key, value in zip(self.losses, values, strict=True):
            self.losses[key].append(value)


def train_vae(train_dataset: DataLoader, test_dataset: DataLoader) -> model.VAE:
    print("train VAE")
    epochs = 30
    visualization_dir = os.path.join(os.getcwd(), "image_train")

    net = to_cuda(model.VAE())
    optimizer = torch.optim.Adam(net.parameters())

    loss_saver = LossSaver(names=["log reconstruction", "KL"])

    for epoch in range(epochs):
        _recon, _kl = 0.0, 0.0
        c = 0
        for batch in train_dataset:
            x = to_cuda(batch[1])

            optimizer.zero_grad()
            xhat, mean, log_var = net(x)

            loss, r_loss, kl_loss = model.vae_training_loss(
                xhat=xhat, x=x, log_var=log_var, mean=mean
            )
            loss.backward()
            optimizer.step()

            _recon += r_loss.item()
            _kl += kl_loss.item()
            loss_saver([np.log(np.log(r_loss.item())), kl_loss.item()])
            c += 1
        _recon = round(_recon / c, 5)
        _kl = round(_kl / c, 5)
        print(f"Epoch {epoch}; recon: {_recon}; kl: {_kl}")
        model.evaluate_image_model(
            net=net,
            test_set=test_dataset,
            save_dir_path=os.path.join(visualization_dir, f"epoch{epoch}"),
            visualize_img=data.visualize_img,
        )
    loss_saver.save_loss("vae_training_loss.png")
    data.cross_epoch_vae_gif(source_dir=visualization_dir)
    return net


def train_model_2(
    train_dataset: DataLoader, test_dataset: DataLoader, vae: model.VAE
) -> model.PhysicsToLatent:
    print("train physics 2 latent")

    to_cuda(vae.eval())
    net = to_cuda(model.PhysicsToLatent())
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    epochs = 30
    loss_saver = LossSaver(names=["MSE (mu)", "MSE (sigma)"])

    mse_loss_fn = torch.nn.MSELoss(reduction="sum")

    for epoch in range(epochs):
        l_mean, l_log_var = 0.0, 0.0
        c = 0
        for batch in train_dataset:
            x = to_cuda(batch[1])
            physics = to_cuda(batch[0])

            optimizer.zero_grad()
            with torch.no_grad():
                _, mean, log_var = vae(x)

            mean_pred, log_var_pred = net(physics)

            loss_mean = mse_loss_fn(
                torch.sigmoid(mean / 5 - 1), torch.sigmoid(mean_pred / 5 - 1)
            )
            loss_log_var = mse_loss_fn(log_var, log_var_pred)

            loss = loss_mean + loss_log_var

            loss.backward()
            optimizer.step()

            l_mean += loss_mean.item()
            l_log_var += loss_log_var.item()
            c += 1
            loss_saver([loss_mean.item(), loss_log_var.item()])
        l_mean = round(l_mean / c, 5)
        l_log_var = round(l_log_var / c, 5)
        print(f"Epoch {epoch}; mean: {l_mean}; log var: {l_log_var}")
    loss_saver.save_loss("model_2_training_loss.png")
    return net


def train_model_3(
    train_dataset: DataLoader, test_dataset: DataLoader, vae: model.VAE
) -> model.LatentToPhysics:
    print("train latent 2 physics")

    to_cuda(vae.eval())
    net = to_cuda(model.LatentToPhysics())
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    epochs = 30
    loss_saver = LossSaver(names=["MSE"])

    mse_loss_fn = torch.nn.MSELoss(reduction="sum")

    for epoch in range(epochs):
        l_mean = 0.0
        c = 0
        for batch in train_dataset:
            x = to_cuda(batch[1])
            physics = to_cuda(batch[0])

            optimizer.zero_grad()
            with torch.no_grad():
                _, mean, log_var = vae(x)

            pred_physics = net(torch.concat([mean, log_var], dim=1))

            loss = mse_loss_fn(pred_physics, physics)

            loss.backward()
            optimizer.step()

            l_mean += loss.item()
            loss_saver([loss.item()])
            c += 1
        l_mean = round(l_mean / c, 5)
        print(f"Epoch {epoch}; physics: {l_mean}")
    model.evaluate_latent_2_physics(vae=vae, net=net, test_set=test_dataset)
    loss_saver.save_loss("model_3_training_loss.png")
    return net
