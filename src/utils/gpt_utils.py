import random
import torch
import math
from tqdm import tqdm


def load_dataset(
    path: str
):
    """
    A function that loads the text and returns it

    args:- 
    - path(str): A .txt file path

    returns:-
    - text(str): The whole content of tiny shakespeare
    """
    assert path.split(".")[-1] == "txt"

    try:
        with open(path, 'r') as file:
            text = file.read()
    except FileNotFoundError:
        raise(FileNotFoundError)

    return text


def get_vocab(
    text: str, 
) -> list:
    """
    A function that returns the entire vocab of the dataset where tokenization is by character
    
    args:-
    - text(str): The entire text in string format
    """
    return sorted(list(set(text)))

def map_tokens_to_ids(
    vocab: list[str]
):
    """
    A function that maps tokens to IDs

    args:-
    - voacb(list[str]): A list of unique tokens in the dataset

    returns:-
    - tuple(dict): a dictionary that maps IDs to tokens and tokens to IDs
    """
    return {i:k for i, k in enumerate(vocab)}, {k:i for i, k in enumerate(vocab)}

def encode_tokens(
    text: str | list, 
    encoder_dict: dict
) -> list[int]:
    """
    A function that encodes tokens from string to IDs

    args:-
    - text(str): the actual text that you want to decode. You need to pass a list of words not a string if you want to decode by list
    - encoder_dict(dict): a dictionary that maps tokens to IDs
    """
    return [encoder_dict[char] for char in text]

def decode_tokens(
    ids: list[int], 
    decoder_dict: dict
) -> str:
    """
    A functions that decodes a list of token IDs to a string of tokens

    args:-
    - ids(list[int]): a list of ints representing IDs
    - encoding_dict(dict): A dictionary that maps IDs to tokens
    """
    return "".join([decoder_dict[i] for i in ids])

def split_data(
    tokens: torch.tensor
)-> tuple[torch.tensor]:
    train_text = tokens[:int(len(tokens)*0.95)]

    val_text = tokens[int(len(tokens)*0.95):]

    return train_text, val_text

def get_batch(tokens: torch.tensor, block_size: int, batch_size: int):
    """
    Gets random sequences from the list of tokens
    """
    ix = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([tokens[i:i+block_size] for i in ix])
    y = torch.stack([tokens[i+1:i+block_size+1] for i in ix])
    return x, y


@torch.no_grad()
def generate(model, idx, max_new_tokens):
    model.eval()

    for _ in range(max_new_tokens):

        # if sequence longer than model context, crop it
        idx_cond = idx[:, -model.T:]

        logits = model(idx_cond)  # [B, T, V]

        # Take last time step
        logits = logits[:, -1, :]  # [B, V]

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

        idx = torch.cat((idx, next_token), dim=1)

    return idx

