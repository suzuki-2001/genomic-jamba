## Genomic Jamba
<img src="https://img.shields.io/badge/-GitHub-181717.svg?logo=github&style=flat"> <img src="https://img.shields.io/badge/-Linux-6C6694.svg?logo=linux&style=flat"> <img src="https://img.shields.io/badge/-Python-F9DC3E.svg?logo=python&style=flat">

Genomic Jamba is a hybrid architecture combining Mamba2 and Flash Attention2 mechanisms for efficient sequence modeling. The model uses a 75:25 ratio of Mamba2 blocks to Flash-Attention2 blocks, leveraging the strengths of both architectures.

![benchmark results](md/results.png)

- [Liu et al (2025), Plant DNA Models](https://huggingface.co/collections/zhangtaolab/plant-foundation-models-668514bd5c384b679f9bdf17) - [[GitHub]](https://github.com/zhangtaolab/plant_DNA_LLMs) [[Paper]](https://www.sciencedirect.com/science/article/pii/S1674205224003903)
- [Evaluated Benchmarks of PDLLMs](https://huggingface.co/zhangtaolab)

## Requirements

- **Python:** 3.12  
- **PyTorch:** 2.5.1 (PyTorch 1.12+ is required by some dependencies)  
- **mamba-ssm:** 2.2.2  
  - **Note:** Installation of `mamba-ssm` is known to fail in several environments. If you encounter issues, try using the `--no-build-isolation` flag ([pypy mamba-ssm](https://pypi.org/project/mamba-ssm/)).  
- **flash-attn:** 2.7.0.post2  

## System Requirements
- **Operating System**: Linux
- **GPU**: NVIDIA GPU (for optimal performance)
- **CUDA**: 11.6+ (For AMD cards, please refer to additional prerequisites provided in the documentation.)

## Pre-trained checkpoints
- pretrained on agro-nucleotide-transformer-1b corpus is available at [here](https://huggingface.co/suzuki-2001/plant-genomic-jamba).

## Usage

- #### pre-training

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
from model import StripedMambaConfig

tokenizer = AutoTokenizer.from_pretrained("suzuki-2001/plant-genomic-jamba")
config = StripedMambaConfig(
    vocab_size=tokenizer.get_vocab_size(),
    hidden_size=512,
    num_hidden_layers=24,
    num_attention_heads=16,
    d_state=64,
    d_conv=4,
    expand=2,
)

model = AutoModelForMaskedLM.from_config(config)
```


- #### finetuning
```python
# load pretrained weights from huggingface hub
model_checkpoint = "suzuki-2001/plant-genomic-jamba"
tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint,
    trust_remote_code=True,
)

# load pre-trained genomic-jamba model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    ignore_mismatched_sizes=True,
    num_labels=1,
    problem_type="regression",
)
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
