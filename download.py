# Downloads model(s) during docker build
from pytorch_transformers.file_utils import cached_path

urls = [
    'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-vocab.json',
    'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-merges.txt',
    'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch_model.bin',
    'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-config.json'
]

for url in urls:
    cached_path(url)

