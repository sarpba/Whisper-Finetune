# Whisper beszéd felismerő modell finomhangolása és gyorsított következtetés

Egyszerűsített kínai | [English](./README_en.md)

![python version](https://img.shields.io/badge/python-3.8+-orange.svg)
![GitHub forks](https://img.shields.io/github/forks/shuaijiang/Whisper-Finetune)
![GitHub Repo stars](https://img.shields.io/github/stars/shuaijiang/Whisper-Finetune)
![GitHub](https://img.shields.io/github/license/shuaijiang/Whisper-Finetune)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging_Face-BELLE--Whisper-blue.svg)](https://huggingface.co/BELLE-2)
![Támogatott rendszerek](https://img.shields.io/badge/Támogatott%20rendszerek-Win/Linux/MAC-9cf)

## Előszó

Az OpenAI nyílt forráskódúvá tette a Whisper projektet, amely állításuk szerint emberi szintű angol beszédfelismerési képességekkel rendelkezik, és további 98 nyelv automatikus beszédfelismerését is támogatja. A Whisper automatikus beszédfelismerési és fordítási feladatokat kínál, amelyek képesek különböző nyelvek beszédét szöveggé alakítani, valamint ezeket a szövegeket angolra fordítani. Ennek a projektnek a fő célja a Whisper modell finomhangolása a Lora segítségével, **támogatva az időbélyeg nélküli adatokkal történő tanítást, az időbélyeggel rendelkező adatokkal történő tanítást és a beszédadatok nélküli tanítást**. Jelenleg több modell is nyílt forráskódú, amelyeket az [openai](https://huggingface.co/openai) oldalon lehet megtekinteni. Az alábbiakban felsorolunk néhány gyakran használt modellt. Ezenkívül a projekt támogatja a CTranslate2 gyorsított következtetést és a GGML gyorsított következtetést is. Megjegyzés: a gyorsított következtetés támogatja a Whisper eredeti modelljének közvetlen konvertálását, nem feltétlenül szükséges a finomhangolás. Támogatja a Windows asztali alkalmazásokat, az Android alkalmazásokat és a szerveroldali telepítést.

### Kérlek, először csillagozd meg :star:
## 🔄 Legújabb frissítések
* [2025/03/26] Továbbfejlesztettük a visszhang hozzáadása funkciót [Add reverb](https://github.com/shuaijiang/Whisper-Finetune/blob/master/docs/robust_asr.md#Add-reverb), növelve a beszédfelismerés robusztusságát.
* [2024/12/16] Továbbfejlesztettük a ggml modell konvertálási funkciót [convert-ggml](https://github.com/shuaijiang/Whisper-Finetune/tree/master/convert-ggml), támogatva a whisper.cpp-t.
* [2024/11/18] Új spektrális augmentációs [SpecAugment](https://github.com/shuaijiang/Whisper-Finetune/blob/master/docs/robust_asr.md#SpecAugment) funkció hozzáadása, amely hatékonyan növeli a beszédfelismerés robusztusságát.
* [2024/10/16] Kiadtuk a [Belle-whisper-large-v3-turbo-zh](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-turbo-zh) modellt, amely a whisper-large-v3-turbo alapú, javított kínai felismerési képességekkel (beleértve az írásjeleket), jelentősen javított kínai felismerési képességgel (24-64%-os relatív javulás) és 7-8-szoros sebességnövekedéssel.
* [2024/06/11] Kiadtuk a [Belle-whisper-large-v3-zh-punct](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh-punct) modellt, amely a Belle-whisper-large-v3 alapú, javított kínai írásjel-felismerési képességekkel, miközben a komplex jelenetek felismerési képessége tovább javult.
* [2024/03/11] Kiadtuk a [Belle-whisper-large-v3-zh](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh) modellt, amely a whisper-large-v3 alapú, javított kínai felismerési képességekkel, jelentősen javított komplex jelenetek felismerési képességgel.
* [2023/12/29] Kiadtuk a [Belle-whisper-large-v2-zh](https://huggingface.co/BELLE-2/Belle-whisper-large-v2-zh) modellt, amely a whisper-large-v2 alapú, javított kínai felismerési képességekkel, jelentősen javított kínai felismerési képességgel.
* [2023/12/29] Kiadtuk a [Belle-distilwhisper-large-v2-zh](https://huggingface.co/BELLE-2/Belle-distilwhisper-large-v2-zh) modellt, amely a distilwhisper-large-v2 alapú, javított kínai felismerési képességekkel, egyensúlyt teremtve a sebesség és a pontosság között.

## Támogatott modellek

- openai/whisper-large-v2
- openai/whisper-large-v3
- openai/whisper-large-v3-turbo
- distil-whisper

**Használati környezet:**

- Anaconda 3
- Python 3.10
- Pytorch 2.1.0
- GPU A100-PCIE-80GB


## Tartalomjegyzék
 - [A projekt fő programjainak bemutatása](#a-projekt-fő-programjainak-bemutatása)
 - [Modell leírása](#modell-leírása)
 - [Modell teljesítménye](#modell-teljesítménye)
 - [Környezet telepítése](#környezet-telepítése)
 - [Adatok előkészítése](#adatok-előkészítése)
 - [Modell finomhangolása](#modell-finomhangolása)
   - [Egykártyás tanítás](#egykártyás-tanítás)
   - [Többkártyás tanítás](#többkártyás-tanítás)
 - [Modellek egyesítése](#modellek-egyesítése)
 - [Modell értékelése](#modell-értékelése)
 - [Predikció](#predikció)
 - [Gyorsított predikció](#gyorsított-predikció)
 - [GUI felületes predikció](#gui-felületes-predikció)
 - [Webes telepítés](#webes-telepítés)
   - [API dokumentáció](#api-dokumentáció)
 - [Android telepítés](#android-telepítés)
 - [Windows asztali alkalmazás](#windows-asztali-alkalmazás)

<a name='a-projekt-fő-programjainak-bemutatása'></a>

## A projekt fő programjainak bemutatása

1. `aishell.py`: AIShell tanítási adatok készítése.
2. `finetune.py`: PEFT módszerrel történő modell finomhangolás.
3. `finetune_all.py`: Teljes paraméteres modell finomhangolás.
4. `merge_lora.py`: Whisper és Lora modellek egyesítése.
5. `evaluation.py`: Finomhangolt vagy eredeti Whisper modell értékelése.
6. `infer_tfs.py`: Transformers használata finomhangolt vagy eredeti Whisper modell predikciójához, csak rövid hanganyagokhoz alkalmas.
7. `infer_ct2.py`: CTranslate2-re konvertált modell használata predikcióhoz, főként ennek a programnak a használata ajánlott.
8. `infer_gui.py`: GUI felülettel rendelkező művelet, CTranslate2-re konvertált modell használata predikcióhoz.
9. `infer_server.py`: CTranslate2-re konvertált modell szerveroldali telepítése, kliensoldali hívásokhoz.
10. `convert-ggml.py`: Modell konvertálása GGML formátumba, Android vagy Windows alkalmazásokhoz.
11. `AndroidDemo`: Ez a könyvtár tartalmazza a modell Androidra történő telepítésének forráskódját.
12. `WhisperDesktop`: Ez a könyvtár tartalmazza a Windows asztali alkalmazás programját.


<a name='modell-leírása'></a>
## Modell leírása
|       Modell      | Paraméterek (M) | Alapmodell | Adat (Újra)mintavételezési ráta |                      Tanítási adathalmazok         | Finomhangolás (teljes vagy peft) |
|:----------------:|:-------:|:-------:|:-------:|:----------------------------------------------------------:|:-----------:|
| Belle-whisper-large-v2-zh | 1550 |whisper-large-v2| 16KHz | [AISHELL-1](https://openslr.magicdatatech.com/resources/33/) [AISHELL-2](https://www.aishelltech.com/aishell_2) [WenetSpeech](https://wenet.org.cn/WenetSpeech/) [HKUST](https://catalog.ldc.upenn.edu/LDC2005S15)  |   teljes finomhangolás   |
| Belle-distil-whisper-large-v2-zh | 756 | distil-whisper-large-v2 | 16KHz | [AISHELL-1](https://openslr.magicdatatech.com/resources/33/) [AISHELL-2](https://www.aishelltech.com/aishell_2) [WenetSpeech](https://wenet.org.cn/WenetSpeech/) [HKUST](https://catalog.ldc.upenn.edu/LDC2005S15)  |   teljes finomhangolás    |
| Belle-whisper-large-v3-zh | 1550 |whisper-large-v3 | 16KHz | [AISHELL-1](https://openslr.magicdatatech.com/resources/33/) [AISHELL-2](https://www.aishelltech.com/aishell_2) [WenetSpeech](https://wenet.org.cn/WenetSpeech/) [HKUST](https://catalog.ldc.upenn.edu/LDC2005S15)  |   teljes finomhangolás   |
| Belle-whisper-large-v3-zh-punct | 1550 | Belle-whisper-large-v3-zh | 16KHz | [AISHELL-1](https://openslr.magicdatatech.com/resources/33/) [AISHELL-2](https://www.aishelltech.com/aishell_2) [WenetSpeech](https://wenet.org.cn/WenetSpeech/) [HKUST](https://catalog.ldc.upenn.edu/LDC2005S15)  |   lora finomhangolás   |
| Belle-whisper-large-v3-turbo-zh | 809 | Belle-whisper-large-v3-turbo | 16KHz | [AISHELL-1](https://openslr.magicdatatech.com/resources/33/) [AISHELL-2](https://www.aishelltech.com/aishell_2) [WenetSpeech](https://wenet.org.cn/WenetSpeech/) [HKUST](https://catalog.ldc.upenn.edu/LDC2005S15)  |   teljes finomhangolás   |
<a name='modell-teljesítménye'></a>

## Modell teljesítménye CER(%) ↓
|      Modell       |  Nyelvi címke   | aishell_1 teszt |aishell_2 teszt| wenetspeech test_net | wenetspeech test_meeting | HKUST_dev| Modell link |
|:----------------:|:-------:|:-----------:|:-----------:|:--------:|:-----------:|:-------:|:-------:|
| whisper-large-v3-turbo | Kínai |   8.639    | 6.014 |   13.507   | 20.313 | 37.324 |[HF](https://huggingface.co/openai/whisper-large-v3-turbo) |
| Belle-whisper-large-v3-turbo-zh | Kínai |   3.070    | 4.114 |   10.230   | 13.357 | 18.944 |[HF](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-turbo-zh) |
| whisper-large-v2 | Kínai |   8.818   | 6.183  |   12.343  |  26.413  | 31.917 | [HF](https://huggingface.co/openai/whisper-large-v2)|
| Belle-whisper-large-v2-zh | Kínai |   **2.549**    | **3.746**  |   **8.503**   | 14.598 | **16.289** |[HF](https://huggingface.co/BELLE-2/Belle-whisper-large-v2-zh) |
| whisper-large-v3 | Kínai |   8.085   | 5.475  |   11.72  |  20.15  | 28.597 | [HF](https://huggingface.co/openai/whisper-large-v3)|
| Belle-whisper-large-v3-zh | Kínai |   2.781    | 3.786 |   8.865   | 11.246 | 16.440 |[HF](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh) |
| Belle-whisper-large-v3-zh-punct | Kínai |   2.945    | 3.808 |   8.998   | **10.973** | 17.196 |[HF](https://huggingface.co/BELLE-2/Belle-whisper-large-v3-zh-punct) |
| distil-whisper-large-v2 | Kínai |  -    | -  |   -  | - | -|[HF](https://huggingface.co/distil-whisper/distil-large-v2) |
| Belle-distilwhisper-large-v2-zh | Kínai |  5.958   | 6.477  |   12.786    | 17.039 | 20.771 | [HF](https://huggingface.co/BELLE-2/Belle-distilwhisper-large-v2-zh) |



**Fontos megjegyzések:**
1. Az értékelés során távolítsa el a modell kimenetéből az írásjeleket, és alakítsa át a hagyományos kínai karaktereket egyszerűsített kínaira.
2. Az `aishell_1_test` az AIShell-1 tesztkészlete, az `aishell_2_test` az AIShell-2 tesztkészlete, a `test_net` és a `test_meeting` a WenetSpeech tesztkészletei.
3. A distil-whisper-large-v2 angol adatokon alapuló desztilláció, csak angol kimenetet tud produkálni. Fontos megjegyezni, hogy az eredeti distil-whisper-large-v2 nem tud kínaiul átírni (csak angolul ad ki).
4. A Belle-whisper-large-v3-zh a Belle-whisper-large-v2-zh-hoz képest jelentős előnnyel rendelkezik komplex jelenetekben, jobb eredményeket ér el a wenetspeech meetingen, 22%-os relatív javulással.
5. A Belle-whisper-large-v3-zh-punct rendelkezik írásjel-felismerési képességgel, az írásjelek a [punc_ct-transformer_cn-en-common-vocab471067-large](https://www.modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/) modellből származnak. Ezenkívül a komplex jelenetek hatékonysága tovább javult.
6. A Belle-whisper-large-v3-turbo-zh a whisper-large-v3-turbo-hoz képest 24-64%-os relatív javulást mutat, a Belle-whisper-large-v3-zh-punct-hoz képest enyhe pontosságcsökkenés tapasztalható, de 7-8-szoros sebességnövekedéssel rendelkezik, ami jelentős alkalmazási értéket képvisel korlátozott számítási kapacitás mellett.
<a name='környezet-telepítése'></a>

## Környezet telepítése

- Először telepítse a Pytorch GPU verzióját. Az alábbiakban kétféle módon telepítheti a Pytorch-ot, csak válasszon egyet.

1. Az alábbiakban az Anaconda használatával történő Pytorch környezet telepítése látható, ha már telepítette, hagyja ki ezt a lépést.
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

2. Az alábbiakban Docker image használata látható, húzzon le egy Pytorch környezet image-et.
```shell
sudo docker pull pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel
```

Ezután lépjen be az image-be, és csatolja az aktuális elérési utat a konténer `/workspace` könyvtárához.
```shell
sudo nvidia-docker run --name pytorch -it -v $PWD:/workspace pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel /bin/bash
```

- Telepítse a szükséges függőségi könyvtárakat.

```shell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- Windows esetén külön kell telepíteni a bitsandbytes-ot.
```shell
python -m pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.40.1.post1-py3-none-win_amd64.whl
```

<a name='adatok-előkészítése'></a>

## Adatok előkészítése

A tanítási adathalmaz a következő, egy jsonlines adatlista, azaz minden sor egy JSON adat, az adatformátum a következő. Ez a projekt biztosít egy programot (`aishell.py`) az AIShell adathalmaz elkészítéséhez, ennek a programnak a futtatása automatikusan letölti és létrehozza a következő formátumú tanítási és tesztelési adathalmazokat. **Megjegyzés:** Ez a program megadhatja az AIShell tömörített fájlját a letöltési folyamat kihagyásához. Ha közvetlenül tölti le, az nagyon lassú lehet. Használhat olyan letöltőket, mint a Xunlei, a adathalmaz letöltéséhez, majd a `--filepath` paraméterrel adja meg a letöltött tömörített fájl elérési útját, például `/home/test/data_aishell.tgz`.

**Tippek:**
1. Ha nem használ időbélyegeket a tanításhoz, akkor nem kell tartalmaznia a `sentences` mezőt.
2. Ha csak egy nyelven vannak adatok, akkor nem kell tartalmaznia a `language` mezőt.
3. Ha üres beszédadatokat tanít, a `sentences` mező `[]`, a `sentence` mező `""`, és a `language` mező nem létezhet.
4. Az adatok nem feltétlenül tartalmaznak írásjeleket, de a finomhangolt modell elveszítheti az írásjel-hozzáadási képességét.

```json
{
   "audio": {
      "path": "dataset/0.wav"
   },
   "sentence": "Az elmúlt években nemcsak én adtam könyvet a lányomnak újévi ajándékként, hanem a rokonokat és barátokat is meggyőztem, hogy ne adjanak pénzt a lányomnak, hanem adjanak könyvet.",
   "language": "Chinese",
   "sentences": [
      {
         "start": 0,
         "end": 1.4,
         "text": "Az elmúlt években,"
      },
      {
         "start": 1.42,
         "end": 8.4,
         "text": "nemcsak én adtam könyvet a lányomnak újévi ajándékként, hanem a rokonokat és barátokat is meggyőztem, hogy ne adjanak pénzt a lányomnak, hanem adjanak könyvet."
      }
   ],
   "duration": 7.37
}
```

<a name='modell-finomhangolása'></a>

## Modell finomhangolása

Az adatok előkészítése után megkezdheti a modell finomhangolását. A tanítás két legfontosabb paramétere a következő: `--base_model` megadja a finomhangolandó Whisper modellt, ennek az értéknek léteznie kell a [HuggingFace](https://huggingface.co/openai) oldalon. Ezt nem kell előre letölteni, a tanítás indításakor automatikusan letöltődik. Természetesen előre is letöltheti, ebben az esetben a `--base_model` az elérési utat adja meg, és a `--local_files_only` értéke True legyen. A második `--output_path` a tanítás során mentett Lora ellenőrzőpontok elérési útja, mivel Lora-t használunk a modell finomhangolásához. Ha elegendő a memória, a legjobb, ha a `--use_8bit` értékét False-ra állítja, így a tanítás sokkal gyorsabb lesz. További paraméterekért tekintse meg ezt a programot.

<a name='egykártyás-tanítás'></a>

### Egykártyás tanítás

Az egykártyás tanítási parancs a következő, Windows rendszeren nem kell hozzáadni a `CUDA_VISIBLE_DEVICES` paramétert.
```shell
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model=openai/whisper-tiny --output_dir=output/
```

<a name='többkártyás-tanítás'></a>

### Többkártyás tanítás

A többkártyás tanításnak két módja van: a torchrun és az accelerate. A fejlesztők a saját szokásaiknak megfelelően használhatják a megfelelő módszert.

1. A torchrun használata többkártyás tanítás indításához, a parancs a következő, a `--nproc_per_node` paraméterrel adja meg a használt grafikus kártyák számát.
```shell
torchrun --nproc_per_node=2 finetune.py --base_model=openai/whisper-tiny --output_dir=output/
```

2. Az accelerate használata többkártyás tanítás indításához, ha először használja az accelerate-et, konfigurálnia kell a tanítási paramétereket, az alábbiak szerint.

Először konfigurálja a tanítási paramétereket, a folyamat során a fejlesztőnek néhány kérdésre kell válaszolnia, általában az alapértelmezett értékek megfelelőek, de néhány paramétert a tényleges helyzetnek megfelelően kell beállítani.
```shell
accelerate config
```

A folyamat nagyjából így néz ki:
```
--------------------------------------------------------------------Milyen számítási környezetben futtatja?
Ez a gép
--------------------------------------------------------------------Milyen típusú gépet használ?
több GPU-s
Hány különböző gépet fog használni (több mint 1-et használjon többcsomópontos tanításhoz)? [1]:
Szeretné optimalizálni a szkriptjét a torch dynamo segítségével? [igen/NEM]:
Szeretné használni a DeepSpeed-et? [igen/NEM]:
Szeretné használni a FullyShardedDataParallel-t? [igen/NEM]:
Szeretné használni a Megatron-LM-et? [igen/NEM]:
Hány GPU-t kell használni az elosztott tanításhoz? [1]:2
Melyik GPU(ka)t (azonosító szerint) kell használni a tanításhoz ezen a gépen vesszővel elválasztott listaként? [mind]:
--------------------------------------------------------------------Szeretne FP16-ot vagy BF16-ot használni (vegyes pontosság)?
fp16
az accelerate konfiguráció mentve a /home/test/.cache/huggingface/accelerate/default_config.yaml helyre
```

A konfigurálás befejezése után a következő paranccsal tekintheti meg a konfigurációt.
```shell
accelerate env
```

A tanítás indítási parancsa a következő.
```shell
accelerate launch finetune.py --base_model=openai/whisper-tiny --output_dir=output/
```


A kimeneti napló a következő:
```shell
{'loss': 0.9098, 'learning_rate': 0.000999046843662503, 'epoch': 0.01}
{'loss': 0.5898, 'learning_rate': 0.0009970611012927184, 'epoch': 0.01}
{'loss': 0.5583, 'learning_rate': 0.0009950753589229333, 'epoch': 0.02}
{'loss': 0.5469, 'learning_rate': 0.0009930896165531485, 'epoch': 0.02}
{'loss': 0.5959, 'learning_rate': 0.0009911038741833634, 'epoch': 0.03}
```

<a name='modellek-egyesítése'></a>

## Modellek egyesítése

A PEFT módszerrel történő modell finomhangolása után két modell lesz: az első a Whisper alapmodell, a második a Lora modell. Ezt a két modellt egyesíteni kell a további műveletekhez. Ez a program csak két paramétert igényel: a `--lora_model` megadja a tanítás után mentett Lora modell elérési útját, ami valójában az ellenőrzőpont mappa elérési útja, a második `--output_dir` az egyesített modell mentési könyvtára.
```shell
python merge_lora.py --lora_model=output/whisper-tiny/checkpoint-best/ --output_dir=models/
```

<a name='modell-értékelése'></a>

## Modell értékelése

Futtassa a következő programot a modell értékeléséhez. A két legfontosabb paraméter a következő: az első `--model_path` megadja az egyesített modell elérési útját, és támogatja az eredeti Whisper modell közvetlen használatát is, például közvetlenül megadva az `openai/whisper-large-v2`-t. A második `--metric` megadja az értékelési módszert, például karakterhiba arány (`cer`) és szóhiba arány (`wer`). **Tipp:** A nem finomhangolt modellek kimenete tartalmazhat írásjeleket, ami befolyásolhatja a pontosságot. További paraméterekért tekintse meg ezt a programot.
```shell
python evaluation.py --model_path=models/whisper-tiny-finetune --metric=cer
```

<a name='predikció'></a>

## Predikció

Futtassa a következő programot a beszédfelismeréshez. Ez a transformers használatával közvetlenül hívja a finomhangolt vagy eredeti Whisper modellt a predikcióhoz, csak rövid hanganyagokhoz alkalmas, hosszú hanganyagokhoz inkább az `infer_ct2.py` használata ajánlott. Az első `--audio_path` paraméter megadja a predikálandó hanganyag elérési útját. A második `--model_path` megadja az egyesített modell elérési útját, és támogatja az eredeti Whisper modell közvetlen használatát is, például közvetlenül megadva az `openai/whisper-large-v2`-t. További paraméterekért tekintse meg ezt a programot.
```shell
python infer_tfs.py --audio_path=dataset/test.wav --model_path=models/whisper-tiny-finetune
```

<a name='gyorsított-predikció'></a>

## Gyorsított predikció

Mint ismeretes, a Whisper modell közvetlen használata a következtetéshez viszonylag lassú, ezért itt egy gyorsítási módszert kínálunk, amely főként a CTranslate2-t használja a gyorsításhoz. Először konvertálni kell a modellt, az egyesített modellt CTranslate2 modellé kell alakítani. Az alábbi parancsban a `--model` paraméter megadja az egyesített modell elérési útját, és támogatja az eredeti Whisper modell közvetlen használatát is, például közvetlenül megadva az `openai/whisper-large-v2`-t. A `--output_dir` paraméter megadja a konvertált CTranslate2 modell elérési útját. A `--quantization` paraméter megadja a modell méretének kvantálását, ha nem szeretné kvantálni a modellt, egyszerűen hagyja ki ezt a paramétert.
```shell
ct2-transformers-converter --model models/whisper-tiny-finetune --output_dir models/whisper-tiny-finetune-ct2 --copy_files tokenizer.json --quantization float16
```

Futtassa a következő programot a gyorsított beszédfelismeréshez. Az `--audio_path` paraméter megadja a predikálandó hanganyag elérési útját. A `--model_path` megadja a konvertált CTranslate2 modellt. További paraméterekért tekintse meg ezt a programot.
```shell
python infer_ct2.py --audio_path=dataset/test.wav --model_path=models/whisper-tiny-finetune-ct2
```

A kimeneti eredmény a következő:
```shell
-----------  Konfigurációs argumentumok -----------
audio_path: dataset/test.wav
model_path: models/whisper-tiny-finetune-ct2
language: zh
use_gpu: True
use_int8: False
beam_size: 10
num_workers: 1
vad_filter: False
local_files_only: True
------------------------------------------------
[0.0 - 8.0]：Az elmúlt években, nemcsak én adtam könyvet a lányomnak ajándékba, hanem a rokonokat és barátokat is meggyőztem, hogy ne adjanak pénzt a lányomnak, hanem adjanak könyvet.
```

<a name='gui-felületes-predikció'></a>

## GUI felületes predikció

Itt is a CTranslate2-t használjuk a gyorsításhoz, a modell konvertálásának módját lásd a fenti dokumentációban. A `--model_path` megadja a konvertált CTranslate2 modellt. További paraméterekért tekintse meg ezt a programot.

```shell
python infer_gui.py --model_path=models/whisper-tiny-finetune-ct2
```

Indítás után a felület a következő:

<div align="center">
<img src="./docs/images/gui.jpg" alt="GUI felület" width="600"/>
</div>

<a name='webes-telepítés'></a>

## Webes telepítés

A webes telepítéshez is a CTranslate2-t használjuk a gyorsításhoz, a modell konvertálásának módját lásd a fenti dokumentációban. A `--host` megadja a szolgáltatás indítási címét, itt `0.0.0.0`-ra van állítva, ami azt jelenti, hogy bármely címről elérhető. A `--port` megadja a használt portszámot. A `--model_path` megadja a konvertált CTranslate2 modellt. A `--num_workers` megadja, hogy hány szálat használjon a párhuzamos következtetéshez, ez fontos a webes telepítésnél, amikor több párhuzamos hozzáférés van, egyszerre tud következtetni. További paraméterekért tekintse meg ezt a programot.

```shell
python infer_server.py --host=0.0.0.0 --port=5000 --model_path=models/whisper-tiny-finetune-ct2 --num_workers=2
```

### API dokumentáció

Jelenleg két interfész érhető el: a normál felismerési interfész `/recognition` és a folyamatos eredményt visszaadó `/recognition_stream`. Vegye figyelembe, hogy ez a folyamatos mód a felismerési eredmények folyamatos visszaadását jelenti, ugyanúgy feltölti a teljes hanganyagot, majd folyamatosan adja vissza a felismerési eredményeket. Ez a módszer nagyon jó élményt nyújt hosszú hanganyagok felismerésekor. A dokumentációs interfészeik teljesen megegyeznek, az interfész paraméterei a következők.

|     Mező     | Kötelező |   Típus   |    Alapértelmezett érték     |              Leírás               |
|:----------:|:----:|:------:|:----------:|:-----------------------------:|
|   audio    |  Igen   |  File  |            |           A felismerendő hangfájl            |
| to_simple  |  Nem   |  int   |     1      |            Hagyományosról egyszerűsítettre váltás            |
| remove_pun |  Nem   |  int   |     0      |           Írásjelek eltávolítása            |
|    task    |  Nem   | String | transcribe | Felismerési feladat típusa, támogatja a transcribe és translate |
|  language  |  Nem   | String |     zh     |    Nyelv beállítása, rövidítés, ha None, akkor automatikusan észleli a nyelvet     |


Visszatérési eredmény:

|   Mező    |  Típus  |      Leírás       |
|:-------:|:----:|:-------------:|
| results | list |    Szegmentált felismerési eredmények    |
| +result | str  |   Minden szegmens szöveges eredménye   |
| +start  | int  | Minden szegmens kezdési ideje, másodpercben |
|  +end   | int  | Minden szegmens befejezési ideje, másodpercben |
|  code   | int  |  Hibakód, 0 a sikeres felismerés  |

Példa:
```json
{
  "results": [
    {
      "result": "Az elmúlt években, nemcsak én adtam könyvet a lányomnak ajándékba, hanem a rokonokat és barátokat is meggyőztem, hogy ne adjanak pénzt a lányomnak, hanem adjanak könyvet.",
      "start": 0,
      "end": 8
    }
  ],
  "code": 0
}
```

A könnyebb érthetőség kedvéért itt egy Python kód a Web API hívásához, az alábbi a `/recognition` hívási módja.
```python
import requests

response = requests.post(url="http://127.0.0.1:5000/recognition",
                         files=[("audio", ("test.wav", open("dataset/test.wav", 'rb'), 'audio/wav'))],
                         json={"to_simple": 1, "remove_pun": 0, "language": "zh", "task": "transcribe"}, timeout=20)
print(response.text)
```

Az alábbi a `/recognition_stream` hívási módja.
```python
import json
import requests

response = requests.post(url="http://127.0.0.1:5000/recognition_stream",
                         files=[("audio", ("test.wav", open("dataset/test_long.wav", 'rb'), 'audio/wav'))],
                         json={"to_simple": 1, "remove_pun": 0, "language": "zh", "task": "transcribe"}, stream=True, timeout=20)
for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
    if chunk:
        result = json.loads(chunk.decode())
        text = result["result"]
        start = result["start"]
        end = result["end"]
        print(f"[{start} - {end}]：{text}")
```


A megadott tesztoldalak a következők:

A főoldal `http://127.0.0.1:5000/` oldala a következő:

<div align="center">
<img src="./docs/images/web.jpg" alt="Főoldal" width="600"/>
</div>

A dokumentációs oldal `http://127.0.0.1:5000/docs` oldala a következő:

<div align="center">
<img src="./docs/images/api.jpg" alt="Dokumentációs oldal" width="600"/>
</div>


<a name='android-telepítés'></a>
## Android telepítés

A telepítési forráskód az [AndroidDemo](./AndroidDemo) könyvtárban található, a részletes dokumentációt a könyvtárban található [README.md](AndroidDemo/README.md) fájlban találja.
<br/>
<div align="center">
<img src="./docs/images/android2.jpg" alt="Android hatáskép" width="200">
<img src="./docs/images/android1.jpg" alt="Android hatáskép" width="200">
<img src="./docs/images/android3.jpg" alt="Android hatáskép" width="200">
<img src="./docs/images/android4.jpg" alt="Android hatáskép" width="200">
</div>


<a name='windows-asztali-alkalmazás'></a>
## Windows asztali alkalmazás

A program a [WhisperDesktop](./WhisperDesktop) könyvtárban található, a részletes dokumentációt a könyvtárban található [README.md](WhisperDesktop/README.md) fájlban találja.

<br/>
<div align="center">
<img src="./docs/images/desktop1.jpg" alt="Windows asztali alkalmazás hatáskép">
</div>



## Referenciák

1. https://github.com/huggingface/peft
2. https://github.com/guillaumekln/faster-whisper
3. https://github.com/ggerganov/whisper.cpp
4. https://github.com/Const-me/Whisper
