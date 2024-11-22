# Fine-Tuning MusicGen
The repository offers training code to fine-tune [MusicGen](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md), a controllable text-to-music model created by Meta. MusicGen is a single stage auto-regressive Transformer model trained over a 32kHz Encodec tokenizer with 4 codebooks sampled at 50 Hz.

The target of this repo is to let people genrate a unique niche of music wether it's a special genre or an artist, or more specifically, to make MusicGen learn a new type of music i.e. add the abillity to generate a new type of music without "forgetting" to make other genres. In order to achieve this goal, we apply a method called [Dreambooth](https://arxiv.org/pdf/2208.12242), a method that is initially intended for text-to-image diffusion models, but we implemented it on our model. You can read more about it in our paper: ________________

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
Then, prepare a few tracks of the music you would like the model to learn to generate. The tracks should be no longer than 30 seconds and to be 44100hz. We reccommend 5 tracks. If you change the amount of tracks make sure to modify the code to generate a reasonable amount of tracks. For more details about it read the paper_________________

In order to create 30 secoonds tracks out of longer tracks and resample you may use the following code _______________________


Now, you'll need to describe in a sentence your music to realize the dreamboothing as good as possible. It should look like something like this:
- Techno with strong bass and 140 bpm
- 80's rock with electric guitars and heavy drums
- chill ambient synths
- upbeat house with catchy bassline

If you dont know exactly how to describe it, you can use our code for help in _________________

## Train
In order to use our code, save the path to yuor tracks in a variable 
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
From here you are all set to run the code in ____________ if you are in a python notebook.
Otherwise, you should change:
- The code that sets the GPU
``` ruby
device = "cuda:0" if torch.cuda.is_available() else "cpu"
```
- The code that trains the model
``` ruby
%env USER=none
command = (
    "dora -P audiocraft run"
    " solver=musicgen/musicgen_base_32khz"
    " model/lm/model_scale=small"
    " continue_from=//pretrained/facebook/musicgen-small"
    " conditioner=text2music"
    " dset=audio/train"
    " dataset.num_workers=2"
    " dataset.valid.num_samples=1"
    " dataset.batch_size=10"
    " schedule.cosine.warmup=8"
    " optim.optimizer=adamw"
    " optim.lr=1e-4" # 1e-5 might be better
    " optim.epochs=1"
    " optim.updates_per_epoch=1000"
    " optim.adam.weight_decay=0.01"
    " generate.lm.prompted_samples=False"
    " generate.lm.gen_gt_samples=True"
)

# Run the command and capture the output
try:
    process = subprocess.run(
        command,
        shell=True,
        check=True,
        capture_output=True,
        text=True
    )
    output = process.stderr

    # Extract the sequence before "/checkpoint.th"
    match = re.search(r"xps/([a-f0-9]+)/checkpoint\.th", output)
    if match:
        sequence = match.group(1)
        print(f"Extracted sequence: {sequence}")

        # Use the extracted sequence for further processing
    else:
        print("Sequence not found in output.")
except subprocess.CalledProcessError as e:
    print(f"Error running command: {e.stderr}")
```
- bla bla
- bla bla

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
__________
```


