import argparse
import functools
import gc
import os

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.data_utils import DataCollatorSpeechSeq2SeqWithPadding, remove_punctuation, to_simple
from utils.reader import CustomDataset
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("test_data",   type=str, default="dataset/test.json",            help="A teszt adathalmaz elérési útja")
add_arg("model_path",  type=str, default="models/whisper-tiny-finetune", help="Az egyesített modell elérési útja, vagy a huggingface modell neve")
add_arg("batch_size",  type=int, default=16,        help="Az értékelés batch mérete")
add_arg("num_workers", type=int, default=8,         help="Az adatok olvasásához használt szálak száma")
add_arg("language",    type=str, default="Chinese", help="Nyelv beállítása, lehet teljes név vagy rövidítés, ha None, akkor többnyelvű értékelés")
add_arg("remove_pun",  type=bool, default=True,     help="Írásjelek eltávolítása?")
add_arg("to_simple",   type=bool, default=True,     help="Egyszerűsített kínai karakterek használata?")
add_arg("timestamps",  type=bool, default=False,    help="Időbélyeg adatok használata az értékelés során?")
add_arg("min_audio_len",     type=float, default=0.5,  help="Minimális hanghossz, másodpercben")
add_arg("max_audio_len",     type=float, default=30,   help="Maximális hanghossz, másodpercben")
add_arg("local_files_only",  type=bool,  default=True, help="Csak helyileg töltse be a modellt, ne próbálja letölteni?")
add_arg("task",       type=str, default="transcribe", choices=['transcribe', 'translate'], help="A modell feladata")
# Removed metric argument as we will calculate both CER and WER
args = parser.parse_args()
print_arguments(args)

# Ellenőrizze, hogy a modell elérési útja érvényes-e
assert 'openai' == os.path.dirname(args.model_path) or os.path.exists(args.model_path), \
    f"模型文件{args.model_path}不存在，请检查是否已经成功合并模型，或者是否为huggingface存在模型"
# Whisper adatfeldolgozó lekérése, ez tartalmazza a jellemző-kiemelőt és a tokenizert
processor = WhisperProcessor.from_pretrained(args.model_path,
                                             language=args.language,
                                             task=args.task,
                                             no_timestamps=not args.timestamps,
                                             local_files_only=args.local_files_only)
# Modell lekérése
model = WhisperForConditionalGeneration.from_pretrained(args.model_path,
                                                        device_map="auto",
                                                        local_files_only=args.local_files_only)
# Explicitly set forced_decoder_ids to None in the generation config
model.generation_config.forced_decoder_ids = None
model.eval()

# Teszt adatok lekérése
test_dataset = CustomDataset(data_list_path=args.test_data,
                             processor=processor,
                             timestamps=args.timestamps,
                             min_duration=args.min_audio_len,
                             max_duration=args.max_audio_len)
print(f"Teszt adatok: {len(test_dataset)}")

# Adat kiegészítő (padding)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
eval_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, collate_fn=data_collator)

# Értékelési módszerek lekérése (CER és WER)
cer_metric = evaluate.load('metrics/cer.py')
wer_metric = evaluate.load('metrics/wer.py')

# Értékelés indítása
for step, batch in enumerate(tqdm(eval_dataloader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    input_features=batch["input_features"].cuda(),
                    decoder_input_ids=batch["labels"][:, :4].cuda(),
                    max_new_tokens=255).cpu().numpy())
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            # Az előrejelzett és a tényleges tokenek szöveggé alakítása
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            # Írásjelek törlése
            if args.remove_pun:
                decoded_preds = remove_punctuation(decoded_preds)
                decoded_labels = remove_punctuation(decoded_labels)
            # Karakterszett egyszerűsítése (ha releváns)
            if args.to_simple:
                decoded_preds = to_simple(decoded_preds)
                decoded_labels = to_simple(decoded_labels)
            # Add results to both metrics
            cer_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
            wer_metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    # Számítási rekordok törlése
    del generated_tokens, labels, batch
    gc.collect()
# Értékelési eredmények kiszámítása
cer = cer_metric.compute()
wer = wer_metric.compute()
# Print results, indicating normalization status
normalization_status = "(normalized)" if args.remove_pun or args.to_simple else ""
print(f"Értékelési eredmény {normalization_status}: CER = {round(cer, 5)}")
print(f"Értékelési eredmény {normalization_status}: WER = {round(wer, 5)}")
