# Fine-Tuning MusicGen

Link to Colab notebook : [Link](https://colab.research.google.com/drive/1ZkRV4hJoPn0aVzlsIFL73QCFiCgav999?usp=sharing)

Link to demo : [Link](https://colab.research.google.com/drive/1dMkrr9Mrso7gWgmAxSOqIataQyEkVJgv?usp=sharing)

The repository offers training code to fine-tune [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md), a controllable text-to-music model created by Meta. MusicGen is a single stage auto-regressive Transformer model trained over a 32kHz Encodec tokenizer with 4 codebooks sampled at 50 Hz.

The target of this repo is to let people genrate a unique niche of music wether it's a special genre or an artist, or more specifically, to make MusicGen learn a new type of music i.e. add the abillity to generate a new type of music without "forgetting" to make other genres. In order to achieve this goal, we take inspiration from [Dreambooth](https://arxiv.org/pdf/2208.12242). Originally, Dreambooth is a method to generate images of personal objects in text-to-image models. The method suggests training the model on examples of the personal object alongside similar examples of the pre-trained model. We adapt and augment this method for text-to-audio by adding additional examples that are not similar, enforcing flexibility of the fine-tuned result. For a review of our method please refer to [our summary of it](https://www.overleaf.com/read/txfjhjpnszcm#cf9b9c)

We have made a notebook that is ready to run on [Google Colab](https://colab.research.google.com/drive/1ZkRV4hJoPn0aVzlsIFL73QCFiCgav999?usp=sharing). Note that in order to run it there you'll need to use the A100 or the L4(with extra GPU RAM) because of the need for quite a large amount of RAM.


## Requirements
You first need to clone that repository and installing the requirements.
``` ruby
!git clone https://github.com/facebookresearch/audiocraft.git
%cd audiocraft
!pip install -e .
!pip install dora-search
!pip install numba
```
Then, prepare a few tracks of the music you would like the model to learn to generate. The tracks should be no longer than 30 seconds and to be 44100hz. We reccommend 5 tracks. If you change the amount of tracks make sure to modify the code to generate a reasonable amount of tracks.

The notebook includes a script to slice the tracks to 30 second long clips if you have not done so ahead of time. 


Now, you'll need to describe in a sentence your music to realize the dreamboothing as good as possible. It should look like something like this:
- Techno with strong bass and 140 bpm
- 80's rock with electric guitars and heavy drums
- chill ambient synths
- upbeat house with catchy bassline

If you dont know exactly how to describe it, the notebook includes a script to recommend a relevant description.

## Train
In order to use our code, save the path to your tracks in a variable 
``` ruby
path_to_sentnce
```
and your description in the variable
``` ruby
description
```
Also prepare a path to a folder where would you like the model generations to be held and save it in the variable 
``` ruby
generated_path
```
make sure to have "/" at the end of it.
__________
From here you are all set to run the training code!
If you are not running this code on Colab, you should probably change some of the code to fit your needs.

Note that it trains a model so it will take a while. As mentioned, make sure you have a GPU with at least 35gb of RAM.

_______________
## Extras
- The code we shared fine tunes the small model in order to fit the GPU that is available on Google Colab. If you have a GPU with more RAM you may modify the code to fine tune a larger model(e.g. medium or large model) through changing that part in the code
``` ruby
from audiocraft.models import MusicGen

musicgen = MusicGen.get_pretrained('facebook/musicgen-small')
musicgen.set_generation_params(duration=30)
```
- You can save the checkpoint you have trained through the following code
``` ruby
import shutil

source_path = f'/tmp/audiocraft_none/xps/ceae9dae/checkpoint.th'
destination_path = '/content/drive/MyDrive/musicgen_finetunes/checkpoints/new3'
os.makedirs(destination_path, exist_ok=True)
shutil.copy(source_path, destination_path)
```
After saving it, if you are unsatisfied from the result you can load that checkpoint and continue the training for more epochs through this code
``` ruby
sig = "YOUR SIG"

command = (
    "dora run solver=musicgen/musicgen_base_32khz"
    " model/lm/model_scale=small"

    # you can continue a run this way, if the filesystem still exists:
    f" continue_from=//SIG/{sig}"

    # or you can save the .th file, load it in a new runtime, and resume from just it:
    f" continue_from=/tmp/audiocraft_argov/xps/{sig}/checkpoint.th"

    ...
    Same parameters as the training
    ...
)

!{command}
```


