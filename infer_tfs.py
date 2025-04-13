import argparse
import functools

import librosa
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from utils.utils import print_arguments, add_arguments


def main():
    parser = argparse.ArgumentParser(description="Whisper audió transzkripciós script")
    add_arg = functools.partial(add_arguments, argparser=parser)

    add_arg("audio_path", type=str, default="dataset/test.wav",
            help="A bemeneti audiófájl elérési útja")
    add_arg("model_path", type=str, default="models/whisper-tiny-finetune",
            help="A modell elérési útja vagy a HuggingFace-modell neve")
    add_arg("language", type=str, default="Hungarian",
            help="A nyelv beállítása (teljes név vagy rövidítés). Ha None, többnyelvű módot használ")
    add_arg("task", type=str, default="transcribe", choices=["transcribe", "translate"],
            help="A modell feladata: transcribe vagy translate")
    add_arg("local_files_only", type=bool, default=True,
            help="Csak helyi modellek használata, ne próbáljon letölteni semmit")

    args = parser.parse_args()
    print_arguments(args)

    # Whisper processzor betöltése
    processor = WhisperProcessor.from_pretrained(
        args.model_path,
        language=args.language,
        task=args.task,
        local_files_only=args.local_files_only
    )

    # Whisper modell betöltése és finomhangolása félprecíziós módra
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="auto",
        local_files_only=args.local_files_only
    ).half()
    model.eval()

    # Forced decoder azonosítók törlése a konfigurációból, hogy elkerüljük a hibát
    model.config.forced_decoder_ids = None
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.forced_decoder_ids = None

    # Audió fájl beolvasása
    sample, sr = librosa.load(args.audio_path, sr=16000)
    duration = sample.shape[-1] / sr
    assert duration < 30, (
        f"Ez a program csak 30 másodpercnél rövidebb audiófájlokra van tervezve, "
        f"de a beolvasott fájl {duration:.2f} mp hosszú."
    )

    # Preprocess: tokenek előállítása, valamint az opcionális attention mask előállítása
    inputs = processor(
        sample,
        sampling_rate=sr,
        return_tensors="pt",
        do_normalize=True
    )
    input_features = inputs.input_features.cuda().half()
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.cuda()

    # Generálás (transzkripció vagy fordítás)
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            max_new_tokens=256
        )

    # A generált tokenek dekódolása, a speciális tokeneket kihagyva
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"Felismert szöveg: {transcription}")


if __name__ == "__main__":
    main()
