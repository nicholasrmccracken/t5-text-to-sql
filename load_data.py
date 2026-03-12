import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch
import numpy as np

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Initializes the dataset and tokenizer, and processes the data for the specified split.

        Inputs:
            * data_folder (str): Path to the dataset directory.
            * split (str): Dataset split to load ("train", "dev", or "test").
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.decoder_start_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        self.data = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        '''
        Loads and tokenizes the dataset, constructing encoder inputs and decoder inputs/targets
        depending on the split.

        Inputs:
            * data_folder (str): Path to the dataset directory.
            * split (str): Dataset split ("train", "dev", or "test").
            * tokenizer: T5 tokenizer used to convert text to token IDs.

        Returns:
            * processed (List[Any]): List of processed dataset examples.
        '''
        input_lines = load_lines(os.path.join(data_folder, f"{split}.nl"))
        processed = []
        
        if split != "test":
            output_lines = load_lines(os.path.join(data_folder, f"{split}.sql"))
            
            for input_text, output_text in zip(input_lines, output_lines):   
                encoder_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=True))
                
                # Tokenize the target SQL query
                target_ids = tokenizer.encode(output_text, add_special_tokens=True)
                
                # Shift the target sequence to the right to create decoder inputs
                # Ensures decoder cannot see the correct next token during prediction
                decoder_inputs = torch.tensor([self.decoder_start_id] + target_ids[:-1])  # i.e. [SELECT, flight_id, FROM, flights]
                decoder_targets = torch.tensor(target_ids)  # i.e. [START, SELECT, flight_id, FROM]
                
                # Save start-of-sequence token separately for generation
                initial_decoder_inputs = torch.tensor([self.decoder_start_id])
                
                processed.append((
                    encoder_ids,
                    decoder_inputs,
                    decoder_targets,
                    initial_decoder_inputs
                ))
        else:
            for input_text in input_lines:
                encoder_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=True))
                initial_decoder_inputs = torch.tensor([self.decoder_start_id])
                
                processed.append((
                    encoder_ids,
                    initial_decoder_inputs
                ))
                         
        return processed
    
    def __len__(self):
        '''
        Returns:
            * int: Total number of processed examples.
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Inputs:
            * idx (int): Index of the example.

        Returns:
            * Any: Processed dataset example.
        '''
        return self.data[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): 
            batch is a list of length batch_size, where each index contains what
            the dataset __getitem__ function returns.

    Returns:
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [item[0] for item in batch]
    decoder_inputs = [item[1] for item in batch]
    decoder_targets = [item[2] for item in batch]
    initial_decoder_inputs = torch.stack([item[3] for item in batch])  # no padding needed
    
    # Pad the variable-length sequences to the same length so they can be processed in a batch
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    
    # Mask is 1 for real tokens and 0 for padding tokens
    # Attention will ignore padding
    encoder_mask = (encoder_ids != PAD_IDX).long()

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): 
            batch is a list of length batch_size, where each index contains what the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = [item[0] for item in batch]
    initial_decoder_inputs = torch.stack([item[1] for item in batch])
    
    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def print_stats():
    """
    Prints dataset statistics for the train and dev splits using the T5 tokenizer.
    Computes number of examples, mean token lengths for NL and SQL, and vocabulary sizes.
    """
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    def compute_stats(split):
        input_lines = load_lines(os.path.join("data", f"{split}.nl"))
        output_lines = load_lines(os.path.join("data", f"{split}.sql"))

        sentence_lengths = []
        query_lengths = []
        nl_vocab = set()
        query_vocab = set()

        # Save input and output lengths and vocabularies for statistics
        for sentence in input_lines:
            token_ids = tokenizer.encode(sentence, add_special_tokens=True)
            sentence_lengths.append(len(token_ids))
            nl_vocab.update(token_ids)

        for query in output_lines:
            token_ids = tokenizer.encode(query, add_special_tokens=True)
            query_lengths.append(len(token_ids))
            query_vocab.update(token_ids)

        print(f"\n{split.upper()} STATS")
        print(f"Number of examples: {len(input_lines)}")
        print(f"Mean sentence length: {np.mean(sentence_lengths):.2f}")
        print(f"Mean SQL query length: {np.mean(query_lengths):.2f}")
        print(f"Vocabulary size (natural language): {len(nl_vocab)}")
        print(f"Vocabulary size (SQL): {len(query_vocab)}")
        print("")

    compute_stats("train")
    compute_stats("dev")
    
def load_t5_data(batch_size, test_batch_size):
    print_stats()  # for tables 1, 2 in report
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader

def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    '''
    Loads natural language instructions and SQL queries for prompting experiments.

    Inputs:
        * data_folder (str): Path to the dataset directory.

    Returns:
        * train_x (List[str]): Training natural language inputs.
        * train_y (List[str]): Training SQL queries.
        * dev_x (List[str]): Development natural language inputs.
        * dev_y (List[str]): Development SQL queries.
        * test_x (List[str]): Test natural language inputs.
    '''
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    
    return train_x, train_y, dev_x, dev_y, test_x
