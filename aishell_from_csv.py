import argparse
import json
import os
import functools
from tqdm import tqdm
import soundfile
import csv
import random
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("csv_path", default=None, type=str, help="A bemeneti CSV fájl elérési útja.")
add_arg("annotation_text", default="dataset/", type=str, help="Annotációs JSON fájlok mentési könyvtára")
add_arg("test_size", default=10000, type=int, help="A test.json fájlba kerülő véletlenszerű sorok száma.")
add_arg("add_pun", default=False, type=bool, help="Központozást is hozzáad a szöveghez? (Eredeti írásjelek megtartása esetén ezt hagyd hamisra)")
args = parser.parse_args()


def process_row(row, add_pun):
    """Egy CSV sor feldolgozása JSON formátumra."""
    if len(row) != 2:
        print(f"Figyelmeztetés: Hibás sor a CSV-ben (nem 2 oszlop): {row}")
        return None

    audio_path = row[0].strip()
    text = row[1].strip()

    if not os.path.exists(audio_path):
        print(f"Figyelmeztetés: Az audio fájl nem található: {audio_path}")
        duration = None
    else:
        try:
            samples, sr = soundfile.read(audio_path)
            duration = round(len(samples) / float(sr), 2)
        except Exception as e:
            print(f"Hiba az audio beolvasásánál ({audio_path}): {e}")
            duration = None

    if add_pun:
        # A központozás hozzáadása itt opcionális, ha szükséges
        print("Figyelmeztetés: A központozás hozzáadása ('add_pun') ebben a verzióban nincs implementálva.")
        pass # Helyőrző a központozás logikájához

    line_data = {
        "audio": {"path": audio_path},
        "sentence": text,
        "duration": duration,
        "sentences": [{"start": 0, "end": duration, "text": text}] if duration is not None else []
    }
    return line_data

def prepare_dataset(csv_path, annotation_path, test_size, add_pun):
    """
    Feldolgozza a megadott CSV fájlt, létrehozza a train.json és test.json annotációs fájlokat.
    A test.json 'test_size' számú véletlenszerű sort tartalmaz.
    """
    print(f"CSV fájl feldolgozása: {csv_path}")
    os.makedirs(annotation_path, exist_ok=True)
    
    all_lines_data = []
    print("CSV beolvasása...")
    with open(csv_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file, delimiter='|')
        for row in tqdm(reader, desc="CSV sorok olvasása"):
             processed_data = process_row(row, add_pun)
             if processed_data:
                 all_lines_data.append(processed_data)

    if not all_lines_data:
        print("Hiba: Nem sikerült érvényes adatot olvasni a CSV fájlból.")
        return

    print(f"Beolvasva {len(all_lines_data)} érvényes sor.")

    # Adatok összekeverése
    random.shuffle(all_lines_data)

    # Adatok szétválasztása train és test részre
    if test_size >= len(all_lines_data):
        print(f"Figyelmeztetés: A kért test méret ({test_size}) nagyobb vagy egyenlő, mint az összes sor ({len(all_lines_data)}). Az összes adat a test.json-be kerül.")
        test_data = all_lines_data
        train_data = []
    elif test_size <= 0:
         print(f"Figyelmeztetés: A kért test méret ({test_size}) érvénytelen. Az összes adat a train.json-be kerül.")
         test_data = []
         train_data = all_lines_data
    else:
        test_data = all_lines_data[:test_size]
        train_data = all_lines_data[test_size:]

    # Fájlok írása
    def write_json_file(data, filename):
        output_path = os.path.join(annotation_path, filename)
        line_count = 0
        with open(output_path, "w", encoding="utf-8") as f_out:
            for line_data in tqdm(data, desc=f"Írás: {filename}"):
                f_out.write(json.dumps(line_data, ensure_ascii=False) + "\n")
                line_count += 1
        print(f"Elkészült: {filename} ({line_count} bejegyzés)")

    if train_data:
        write_json_file(train_data, "train.json")
    else:
         print("Nincs adat a train.json fájlhoz.")
         
    if test_data:
        write_json_file(test_data, "test.json")
    else:
        print("Nincs adat a test.json fájlhoz.")


def main():
    print_arguments(args)
    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"A megadott CSV fájl nem található: {args.csv_path}")
    prepare_dataset(csv_path=args.csv_path, 
                    annotation_path=args.annotation_text, 
                    test_size=args.test_size, 
                    add_pun=args.add_pun)


if __name__ == '__main__':
    main()
