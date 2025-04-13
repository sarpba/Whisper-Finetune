import argparse
import json
import os
import functools
from tqdm import tqdm
import soundfile

from datasets import load_dataset, concatenate_datasets
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("annotation_text", default="dataset/", type=str, help="Annotációs JSON fájlok mentési könyvtára")
add_arg("add_pun", default=False, type=bool, help="Központozást is hozzáad a szöveghez? (Eredeti írásjelek megtartása esetén ezt hagyd hamisra)")
args = parser.parse_args()


def prepare_dataset(annotation_path, add_pun):
    print("Betöltjük a HuggingFace-ről a Mozilla Common Voice 17.0 'hu' részhalmazát...")
    ds = load_dataset("mozilla-foundation/common_voice_17_0", "hu")
    splits = ds.keys()
    
    if "train" in splits and "dev" in splits:
        train_split = concatenate_datasets([ds["train"], ds["dev"]])
    elif "train" in splits and "validation" in splits:
        train_split = concatenate_datasets([ds["train"], ds["validation"]])
    elif "train" in splits:
        train_split = ds["train"]
    else:
        raise ValueError("A dataset nem tartalmaz 'train' split-et.")

    if "test" in splits:
        test_split = ds["test"]
    elif "validation" in splits and "train" in splits:
        test_split = ds["validation"]
    else:
        test_split = None

    def process_split(split, filename):
        output_path = os.path.join(annotation_path, filename)
        os.makedirs(annotation_path, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f_out:
            for record in tqdm(split, desc=f"Feldolgozás: {filename}"):
                # Az eredeti szöveget megtartjuk: belső szóközök, írásjelek változatlanok
                text = record["sentence"].strip()
                if add_pun:
                    import logging
                    from modelscope.pipelines import pipeline
                    from modelscope.utils.constant import Tasks
                    from modelscope.utils.logger import get_logger
                    logger = get_logger(log_level=logging.CRITICAL)
                    logger.setLevel(logging.CRITICAL)
                    inference_pipeline = pipeline(task=Tasks.punctuation,
                                                  model='damo/punc_ct-transformer_cn-en-common-vocab471067-large',
                                                  model_revision="v1.0.0")
                    text = inference_pipeline(text_in=text)['text']
                audio_data = record["audio"]
                if "array" in audio_data and audio_data["array"] is not None:
                    array = audio_data["array"]
                    sr = audio_data["sampling_rate"]
                    duration = round(len(array) / float(sr), 2)
                else:
                    try:
                        samples, sr = soundfile.read(audio_data["path"])
                        duration = round(len(samples) / float(sr), 2)
                    except Exception as e:
                        print(f"Hiba az audio beolvasásánál ({audio_data.get('path', '')}): {e}")
                        duration = None

                line = {
                    "audio": {"path": audio_data.get("path", "")},
                    "sentence": text,
                    "duration": duration,
                    "sentences": [{"start": 0, "end": duration, "text": text}] if duration is not None else []
                }
                f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"Elkészült: {filename} ({len(split)} bejegyzés)")
    
    process_split(train_split, "train.json")
    if test_split is not None:
        process_split(test_split, "test.json")
    else:
        print("Nincs 'test' split a datasetben.")


def main():
    print_arguments(args)
    prepare_dataset(annotation_path=args.annotation_text, add_pun=args.add_pun)


if __name__ == '__main__':
    main()
