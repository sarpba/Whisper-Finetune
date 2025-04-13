import re
from dataclasses import dataclass
from typing import Any, List, Dict, Union

import torch
from zhconv import convert


# 删除标点符号
def remove_punctuation(text: str or List[str]):
    punctuation = '!,.;:?、！，。；：？'
    if isinstance(text, str):
        text = re.sub(r'[{}]+'.format(punctuation), '', text).strip()
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = re.sub(r'[{}]+'.format(punctuation), '', t).strip()
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')


# 将繁体中文总成简体中文
def to_simple(text: str or List[str]):
    if isinstance(text, str):
        text = convert(text, 'zh-cn')
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = convert(t, 'zh-cn')
            result_text.append(t)
        return result_text
    else:
        raise Exception(f'不支持该类型{type(text)}')


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Csak az input_features és labels értékeket dolgozzuk fel, ignorálva minden extra kulcsot
        cleaned_features = []
        for feature in features:
            cleaned_feature = {
                "input_features": feature["input_features"][0],
                "labels": feature["labels"]
            }
            cleaned_features.append(cleaned_feature)

        # Audio input feldolgozása
        input_features = [{"input_features": cf["input_features"]} for cf in cleaned_features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Label-ek feldolgozása
        label_features = [{"input_ids": cf["labels"]} for cf in cleaned_features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
