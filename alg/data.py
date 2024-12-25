import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
class HuggingFaceDataset(IterableDataset):
    def __init__(self, dataset_args, tokenizer):
        """
        Args:
            dataset_name (str): Name of the Hugging Face dataset to load.
            tokenizer_name (str): Name of the tokenizer to use.
            split (str): Which split of the dataset to use (e.g., 'train', 'test', 'validation').
            max_length (int): Maximum length of the tokenized sequences.
            buffer_size (int): Buffer size for shuffling the streamed dataset.
        """
        # we only use redpajama
        rp = load_dataset(
                "semran1/fineweb_4096",
                split="train",
            )
        self.dataset = rp
        self.tokenizer = tokenizer
        self.max_length = dataset_args.dataset_max_seq_length
        self.args = dataset_args

    def __iter__(self):
        buffer = []
        for sample in self.dataset.shuffle():
            inputs = self.tokenizer(
                sample["text"],
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            input_ids = inputs['input_ids'].squeeze(0)# Remove batch dimension
            yield {
                'input_ids': input_ids
            }

def get_dataset(dataset_args, tokenizer):
    dataset = HuggingFaceDataset(dataset_args, tokenizer)
    return dataset
