# Downloads model(s) during docker build
from pytorch_transformers.file_utils import cached_path

urls = [
    'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-vocab.json',
    'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-merges.txt',
    'https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch_model.bin',
]

for url in urls:
    cached_path(url)

