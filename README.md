# Fine-Tuning MusicGen
The repository offers training code to fine-tune [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md), a controllable text-to-music model created by Meta. MusicGen is a single stage auto-regressive Transformer model trained over a 32kHz Encodec tokenizer with 4 codebooks sampled at 50 Hz.

The target of this repo is to let people genrate a unique niche of music wether it's a special genre or an artist, or more specifically, to make MusicGen learn a new type of music i.e. add the abillity to generate a new type of music without "forgetting" to make other genres. In order to achieve this goal, we apply a method called [Dreambooth](https://arxiv.org/pdf/2208.12242), a method that is initially intended for text-to-image diffusion models, but we implemented it on our model. You can read more about it in our paper: ________________

We have made a notebook that is ready to run on [google colab](https://colab.research.google.com/drive/1ZkRV4hJoPn0aVzlsIFL73QCFiCgav999?usp=sharing). Note that in order to run it there you'll need to use the A100 or the L4(with extra GPU RAM) because of the need for quite a large amount of RAM.


## Requirements
You first need to clone that repository and installing the requirements.
