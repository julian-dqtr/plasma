import data
import training_loop
import os
import argparse
import torch
import model
from torchinfo import summary
import pipeline

parser = argparse.ArgumentParser()
parser.add_argument(
    "--vae",
    default=os.path.join("checkpoints", "vae.pt"),
    type=str,
    help="path to vae params",
)
parser.add_argument(
    "--p2l",
    default=os.path.join("checkpoints", "p2l.pt"),
    type=str,
    help="path to model 2 params",
)
parser.add_argument(
    "--l2p",
    default=os.path.join("checkpoints", "l2p.pt"),
    type=str,
    help="path to model 2 params",
)
parser.add_argument("--pipeline", action="store_true", help="launch pipeline")
parser.add_argument(
    "--retrain-vae", action="store_true", help="force vae retraining"
)
parser.add_argument(
    "--retrain-p2l", action="store_true", help="force model 2 retraining"
)
parser.add_argument(
    "--retrain-l2p", action="store_true", help="force model 3 retraining"
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.pipeline:
        pipeline.launch_pipeline(
            vae_path=args.vae,
            physics_to_latent_path=args.p2l,
            latent_to_physics_path=args.l2p,
        )
    print("load data")
    pairs_labels_and_image_paths = data.prepare_raw_data(
        zip_file_path="data/plasma_images.zip", target_dir_path=os.getcwd()
    )
    train_dataset, test_dataset = data.get_datasets(
        pairs_labels_and_image_paths
    )
    vae_path = args.vae

    vae = model.VAE()
    summary(vae, input_size=(1, 1, 64, 64))
    os.makedirs(os.path.dirname(vae_path), exist_ok=True)
    if args.retrain_vae or not os.path.exists(vae_path):
        vae = training_loop.train_vae(
            train_dataset=train_dataset, test_dataset=test_dataset
        )
        torch.save(vae.state_dict(), vae_path)
    vae.load_state_dict(torch.load(vae_path, weights_only=True))
    vae.eval()

    physics_to_latent_path = args.p2l
    physics_to_latent = model.PhysicsToLatent()
    summary(physics_to_latent, input_size=(1, 2))
    if args.retrain_p2l or not os.path.exists(physics_to_latent_path):
        physics_to_latent = training_loop.train_model_2(
            train_dataset=train_dataset, test_dataset=test_dataset, vae=vae
        )
        torch.save(physics_to_latent.state_dict(), physics_to_latent_path)
    physics_to_latent.load_state_dict(
        torch.load(physics_to_latent_path, weights_only=True)
    )
    physics_to_latent.eval()

    latent_to_physics_path = args.l2p
    latent_to_physics = model.LatentToPhysics()
    summary(latent_to_physics, input_size=(1, 2 * 16 * 4 * 4))
    if args.retrain_l2p or not os.path.exists(latent_to_physics_path):
        latent_to_physics = training_loop.train_model_3(
            train_dataset=train_dataset, test_dataset=test_dataset, vae=vae
        )
        torch.save(latent_to_physics.state_dict(), latent_to_physics_path)
    latent_to_physics.load_state_dict(
        torch.load(latent_to_physics_path, weights_only=True)
    )
    latent_to_physics.eval()
