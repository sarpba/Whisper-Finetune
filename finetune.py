import argparse
import functools
import os
import platform
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch import nn
import platform

import torch
from peft import LoraConfig, get_peft_model, AdaLoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.utils import is_peft_available

from utils.callback import SavePeftModelCallback
from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding
from utils.model_utils import load_from_checkpoint
from utils.reader import CustomDataset
from utils.utils import print_arguments, make_inputs_require_grad, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("train_data",    type=str, default="dataset/train.json",       help="A tanító adathalmaz elérési útja")
add_arg("test_data",     type=str, default="dataset/test.json",        help="A teszt adathalmaz elérési útja")
add_arg("base_model",    type=str, default="openai/whisper-small",      help="A Whisper alapmodellje")
add_arg("output_dir",    type=str, default="output/",                  help="A modell mentési elérési útja")
add_arg("warmup_steps",  type=int, default=1500,      help="A betanítás előmelegítési lépéseinek száma")
add_arg("logging_steps", type=int, default=100,     help="Naplózási lépések száma")
add_arg("eval_steps",    type=int, default=1000,    help="Hány lépésenként történjen értékelés")
add_arg("save_steps",    type=int, default=1000,    help="Hány lépésenként mentsük a modellt")
add_arg("num_workers",   type=int, default=8,       help="Adatbetöltő szálak száma")
add_arg("learning_rate", type=float, default=1e-4,  help="Tanulási ráta")
add_arg("min_audio_len", type=float, default=0.5,   help="Minimális hangfájl hossz, másodpercben")
add_arg("max_audio_len", type=float, default=30,    help="Maximális hangfájl hossz, másodpercben")
add_arg("use_adalora",   type=bool,  default=False,  help="AdaLora használata Lora helyett")
add_arg("fp16",          type=bool,  default=True,  help="Használjunk-e fp16 pontosságú tanítást")
add_arg("use_8bit",      type=bool,  default=False, help="Model quantization 8 bitre")
add_arg("timestamps",    type=bool,  default=False, help="Időbélyeg használata a tanítás során")
add_arg("local_files_only", type=bool, default=False, help="Csak helyi fájlok használata, ne töltsön le semmit")
add_arg("num_train_epochs", type=int, default=5,      help="Tanítási epochok száma")
add_arg("language",      type=str, default="Hungarian", help="A feldolgozandó nyelv (teljes vagy rövidítés)")
add_arg("task",     type=str, default="transcribe", choices=['transcribe', 'translate'], help="A modell feladata")
add_arg("augment_config_path",         type=str, default=None, help="Adatbővítési konfigurációs fájl útvonala")
add_arg("resume_from_checkpoint",      type=str, default=None, help="Checkpoint elérési út, ha folytatni szeretnénk")
add_arg("per_device_train_batch_size", type=int, default=8,    help="Train batch mérete")
add_arg("per_device_eval_batch_size",  type=int, default=8,    help="Eval batch mérete")
add_arg("gradient_accumulation_steps", type=int, default=1,    help="Gradiensegyesítés lépésszáma")
args = parser.parse_args()
print_arguments(args)

# Whisper processzor lekérése: tartalmaz feature extractort és tokenizer-t
processor = WhisperProcessor.from_pretrained(args.base_model,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)

# Adatok beolvasása
train_dataset = CustomDataset(data_list_path=args.train_data,
                              processor=processor,
                              language=args.language,
                              timestamps=args.timestamps,
                              min_duration=args.min_audio_len,
                              max_duration=args.max_audio_len,
                              augment_config_path=args.augment_config_path)
test_dataset = CustomDataset(data_list_path=args.test_data,
                             processor=processor,
                             language=args.language,
                             timestamps=args.timestamps,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len)
print(f"Tanító adatok: {len(train_dataset)}, Teszt adatok: {len(test_dataset)}")

# Adatokhoz padding collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Whisper modell betöltése
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

model = WhisperForConditionalGeneration.from_pretrained(args.base_model,
                                                        load_in_8bit=args.use_8bit,
                                                        device_map=device_map,
                                                        local_files_only=args.local_files_only)
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# Modell kvantálása
model = prepare_model_for_kbit_training(model)

# Forward hook regisztrálása, hogy a multi-GPU tréning működjön
model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

print('LoRA modul betöltése...')
if args.resume_from_checkpoint:
    # Ha folytatjuk a tanítást, betöltjük a LoRA súlyokat
    print("Adapterek betöltése checkpointból.")
    model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
else:
    print(f'LoRA modulok hozzáadása...')
    target_modules = ["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"]
    print(target_modules)
    if args.use_adalora:
        config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                               lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, target_modules=target_modules, total_step=18186)
    else:
        config = LoraConfig(r=32, lora_alpha=64, target_modules=target_modules, lora_dropout=0.05, bias="none")
    model = get_peft_model(model, config)

if args.base_model.endswith("/"):
    args.base_model = args.base_model[:-1]
output_dir = os.path.join(args.output_dir, os.path.basename(args.base_model))

# Tanítási paraméterek meghatározása
training_args = \
    Seq2SeqTrainingArguments(output_dir=output_dir,  # Checkpointok és logok mentési helye
                             per_device_train_batch_size=args.per_device_train_batch_size,  # Train batch mérete
                             per_device_eval_batch_size=args.per_device_eval_batch_size,  # Eval batch mérete
                             gradient_accumulation_steps=args.gradient_accumulation_steps,  # Gradiensegyesítés
                             learning_rate=args.learning_rate,  # Tanulási ráta
                             warmup_steps=args.warmup_steps,  # Előmelegítő lépések
                             num_train_epochs=args.num_train_epochs,  # Epochok száma
                             save_strategy="steps",  # Checkpoint mentési stratégia
                             eval_strategy="steps",  # Értékelési stratégia
                             load_best_model_at_end=True,  # Legjobb modell betöltése a végén
                             fp16=args.fp16,  # Fél pontosságú tanítás
                             report_to=["tensorboard"],  # Log mentése tensorboard-ba
                             save_steps=args.save_steps,  # Checkpoint mentési lépések
                             eval_steps=args.eval_steps,  # Értékelési lépések
                             #save_total_limit=5,  # Maximum checkpointok száma
                             optim='adamw_torch',  # Optimalizálási algoritmus
                             ddp_find_unused_parameters=False if ddp else None,  # DDP beállítások
                             dataloader_num_workers=args.num_workers,  # Adatbetöltő szálak száma
                             logging_steps=args.logging_steps,  # Naplózási lépések
                             remove_unused_columns=False,  # Nem használt oszlopok eltávolítása
                             label_names=["labels"])  # Címkék nevei

if training_args.local_rank == 0 or training_args.local_rank == -1:
    print('=' * 90)
    model.print_trainable_parameters()
    print('=' * 90)

# Ha PyTorch 2.0+, használjuk a compiler-t (Windows kivétel)
# if torch.__version__ >= "2" and platform.system().lower() != 'windows':
#     model = torch.compile(model)

# Custom Trainer to handle the 'num_items_in_batch' argument issue
class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    # Add num_items_in_batch to the signature to accept the 4th positional argument
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        Override training_step to accept the unexpected 'num_items_in_batch' argument
        but not pass it to the parent method.
        """
        # Call the parent training_step without the num_items_in_batch argument
        return super().training_step(model, inputs)


# Tréner definiálása
trainer = CustomSeq2SeqTrainer(args=training_args,
                         model=model,
                         train_dataset=train_dataset,
                         eval_dataset=test_dataset,
                         data_collator=data_collator,
                         callbacks=None) # Removed SavePeftModelCallback
model.config.use_cache = False
trainer._load_from_checkpoint = load_from_checkpoint

# Tanítás indítása
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# Modell mentése
trainer.save_state()
if training_args.local_rank == 0 or training_args.local_rank == -1:
    model.save_pretrained(os.path.join(output_dir, "checkpoint-final"))
