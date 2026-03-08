import sys
sys.path.append(".")

from src.models.gpt import GPT

from src.utils.gpt_utils import (
    load_dataset, 
    get_vocab, 
    map_tokens_to_ids, 
    encode_tokens, 
    decode_tokens, 
    split_data, 
    get_batch, 
    generate
)

import torch
import torch.nn as nn
import yaml
import math
from tqdm import tqdm
import random
import argparse
import matplotlib.pyplot as plt
import os

def train_gpt(
    dataset_path: str
):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # read dataset
    text = load_dataset(dataset_path)
    vocab = get_vocab(
        text
    )

    int_to_char, char_to_int = map_tokens_to_ids(
        vocab
    )

    tokens = encode_tokens(
        text=text,
        encoder_dict=char_to_int 
    )

    training_text, val_text = split_data(
        tokens=torch.tensor(tokens)
    )

    # read configs
    with open("configs/gpt.yaml") as file:
        config = yaml.safe_load(file)

    steps = config['training']['steps']
    batch_size = config['training']['batch_size']
    lr = float(config['training']['lr'])
    val_interval = config['training']['val_interval']
    val_iters = config['training']['val_iters']
    
    vocab_size = len(vocab)
    max_seq_len = config['model']['seq_len']

    # load model
    model = GPT(
        vocab_size=vocab_size, 
        seq_len=max_seq_len, 
        d_model=config["model"]["d_model"], 
        n_layers=config["model"]["n_layers"], 
        num_heads=config["model"]["num_heads"], 
        dropout=config["model"]["dropout"],
        hc=config["model"]["hc"], 
        expansion_rate=config["model"]["expansion_rate"]
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    training_text = training_text.to(device)
    val_text = val_text.to(device)

    train_losses = []
    val_losses = []
    train_ppls = []
    val_ppls = []
    grad_norms = []

    model.to(device)

    running_train_loss = 0

    for step in tqdm(range(steps)):

        model.train()

        x, y = get_batch(tokens=training_text,
                         block_size=max_seq_len,
                         batch_size=batch_size)

        x, y = x.to(device), y.to(device)

        logits = model(x)  # [B, T, V]
        loss = criterion(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        # compute gradient norm 
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)

        optimizer.step()

        running_train_loss += loss.item()

        # validation
        if (step + 1) % val_interval == 0:

            avg_train_loss = running_train_loss / val_interval
            train_losses.append(avg_train_loss)
            train_ppls.append(math.exp(avg_train_loss))

            running_train_loss = 0

            model.eval()
            val_loss_total = 0

            with torch.no_grad():
                for _ in range(val_iters):
                    x, y = get_batch(tokens=val_text,
                                     block_size=max_seq_len,
                                     batch_size=batch_size)
                    x, y = x.to(device), y.to(device)

                    logits = model(x)
                    val_loss = criterion(
                        logits.reshape(-1, vocab_size),
                        y.reshape(-1)
                    )
                    val_loss_total += val_loss.item()

            avg_val_loss = val_loss_total / val_iters
            val_losses.append(avg_val_loss)
            val_ppls.append(math.exp(avg_val_loss))

            print(f"\nStep {step+1}")
            print(f"Train Loss: {avg_train_loss:.4f} | PPL: {train_ppls[-1]:.2f}")
            print(f"Val   Loss: {avg_val_loss:.4f} | PPL: {val_ppls[-1]:.2f}")
            print(f"Grad Norm: {grad_norms[-1]:.4f}")

            # generate samples
            print("\nSamples:")
            for _ in range(3):
                context = torch.zeros((1,1), dtype=torch.long, device=device) + random.randint(0, 64)
                generated = generate(model, context, 100)
                sample = decode_tokens(generated[0].tolist(), int_to_char)
                print(sample)
                print("-" * 30)

    metrics = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_perplexity": train_ppls,
        "val_perplexity": val_ppls,
        "grad_norms": grad_norms,
    }

    return model, metrics

def save_plots(experiment_name: str, metrics: dict):
    os.makedirs(f"results/{experiment_name}", exist_ok=True)

    plt.figure()
    plt.plot(metrics["train_loss"])
    plt.plot(metrics["val_loss"])
    plt.title("Train vs Validation Loss")
    plt.xlabel("Validation Interval")
    plt.ylabel("Cross Entropy Loss")
    plt.legend(["Train Loss", "Validation Loss"])
    plt.grid(True)

    plt.savefig(f"results/{experiment_name}/train_vs_val_loss.png", dpi=300, bbox_inches="tight")

    plt.close()

    plt.figure()
    plt.plot(metrics["grad_norms"])
    plt.title("Gradient Norm Over Training")
    plt.xlabel("Training Step")
    plt.ylabel("L2 Gradient Norm")
    plt.ylim(0, 4) 
    plt.grid(True)

    plt.savefig(f"results/{experiment_name}/grad_norm.png", dpi=300, bbox_inches="tight")

    plt.close()

    plt.figure()
    plt.plot(metrics["train_perplexity"])
    plt.plot(metrics["val_perplexity"])
    plt.title("Train vs Validation Perplexity")
    plt.xlabel("Validation Interval")
    plt.ylabel("Perplexity")
    plt.ylim(0, 20) 
    plt.legend(["Train Perplexity", "Validation Perplexity"])
    plt.grid(True)

    plt.savefig(f"results/{experiment_name}/train_vs_val_ppl.png", dpi=300, bbox_inches="tight")

    plt.close()
    
def main():    
    parser = argparse.ArgumentParser(description="A script that runs an ml experiment")
    parser.add_argument("--experiment", help="The name of the experiment you are conducting")
    parser.add_argument("--dataset", help="The path of the dataset", default="tiny_shakespeare.txt")
    args = parser.parse_args()

    trained_model, metrics = train_gpt(
        args.dataset
    )

    if args.experiment is not None:
        # create and save plots
        save_plots(
            experiment_name=args.experiment, 
            metrics=metrics
        )

        # save model
        torch.save(
            trained_model, 
            f"results/{args.experiment}/model.pt"
        )
        print(f"Finished running experiement and saved plots")
    else:
        print("Finished running experiement")


if __name__ == "__main__":
    main()