# Persian SEA / research_sign — آماده‌سازی قبل از اولین Training واقعی

**مبنای این Patch:** شاخه `main`، Commit `ee63b37974740ae80c9f52c54f65047abfd33814` (۲۳ اکتبر ۲۰۲۶).  
**وظیفهٔ اصلی:** ویدیوی پیوستهٔ زبان اشارهٔ ایرانی → متن فارسی، بدون Gloss اجباری.  
**وضعیت:** پیاده‌سازی اولیهٔ قابل‌تست و قابل‌آموزش *GFSLT-VLP-inspired*؛ نه بازتولید دقیق معماری یا نتایج مقاله.

## تصمیم‌های معماری

```text
MP4 -> MediaPipe Pose -> E4s Segmentation -> DINOv2 Multi-view cache
                                  |
                         VideoTextModel.video / VideoTower
                                  |
                 مرحله ۱: BGE-M3 + Contrastive/MIL (موجود)
                                  |
                     best_video_text.pt
                                  |
                 مرحله ۲: VideoTower از Checkpoint
                                  |
                      VisualLanguageAdapter
                                  |
                    mT5-small Persian Decoder
                                  |
                          متن فارسی پیشنهادی
```

GFSLT-VLP اصلی علاوه بر Visual–Language Alignment از وظایف Pretraining دیگری مثل masked reconstruction نیز استفاده می‌کند. این Patch **آن مراحل را دقیقاً تکرار نمی‌کند**؛ به‌جایش از VLP Contrastive/MIL موجود مخزن و Decoder چندزبانهٔ قابل Fine-tune استفاده می‌کند. مدل mT5 پیش‌آموزش‌دیده به‌تنهایی ترجمهٔ اشاره نمی‌داند.

### چه چیزهایی اضافه شده؟

| فایل | کار | چرا لازم است؟ |
| --- | --- | --- |
| `configs/video_text_pretrain_first.yaml` | Baseline اولیهٔ VLP، batch=2 و ۱۰۰ Step | پیش‌آموزش Encoder و تولید Checkpoint قابل‌انتقال بدون تغییر Configهای قبلی |
| `scripts/pretrain_gate.py` | بررسی Train/Val/Test، Cache و Signer | جلوگیری از Training روی Cache معیوب یا دادهٔ نشت‌کرده |
| `src/persian_sea/translation/config.py` | Config مستقل با اعتبارسنجی | جدا ماندن آزمایش ترجمه از Retrieval |
| `src/persian_sea/translation/data.py` | تبدیل VTT و Visual Cache به نمونهٔ آموزش؛ Windowing بدون VTT برای Inference | آموزش با برچسب ضعیف؛ تست واقعی بدون دادن متن به مدل |
| `src/persian_sea/translation/model.py` | بارگذاری VideoTower پیش‌آموزش‌دیده + Adapter + mT5 Decoder | تبدیل بردارهای زمانی `[B,M,256]` به متن فارسی با Cross-Attention |
| `src/persian_sea/translation/readiness.py` | اعتبارسنجی دقیق داده قبل از Translation | جلوگیری از خطای دیرهنگام و حفظ Split علمی |
| `src/persian_sea/translation/train.py` | Teacher forcing، AdamW، accumulation، Validation، Checkpoint | اولین Training تولید متن با انتخاب Checkpoint براساس Val Loss |
| `src/persian_sea/translation/evaluate.py` | Loss، BLEU و chrF روی Val/Test | سنجهٔ اولیهٔ کیفیت متن با اعلام محدودیت پنجرهٔ وابسته به VTT |
| `src/persian_sea/translation/infer.py` | تولید JSON از فقط Visual Cache | در زمان استنتاج، **VTT و متن ورودی خوانده نمی‌شود** |
| `src/persian_sea/translation/cli.py` | دستورات `check/train/eval/infer` | اجرای یکسان روی VS Code یا سرور |
| `configs/translation_mt5_first.yaml` | پارامترهای ترجمه | قابل تکرار و قابل نسخه‌بندی شدن آزمایش |
| `requirements-translation.txt` | SentencePiece و SacreBLEU | Tokenizer mT5 و محاسبهٔ معیارهای ترجمه |
| `tests/test_translation_first.py` | تست‌های CPU بدون دانلود مدل | Shape، Mask، Gradient، Windowing، NaN و Gate |

**فایل‌های اصلی موجود جایگزین نشده‌اند:** `video_text/model.py`، `pose_encoder.py`، `rgb_encoder.py`، `features/scanner.py`، `alignment/*` و `pyproject.toml` دست‌نخورده‌اند. پکیج جدید با `setuptools.find` موجود شناسایی می‌شود.

## وضعیت واقعی Pilot تک‌ویدیویی

براساس لاگ‌های دریافت‌شده از مک M5 / 16GB:

- `news001.mp4`: حدود ۱۹۷ ثانیه، ۵۹۱۱ فریم، ۳۰ FPS.
- `news001.vtt`: ۲۸ Cue فارسی؛ زمان‌ها لزوماً Ground Truth دقیق نیستند.
- Pose با MediaPipe `model_complexity: 1` ساخته شد؛ ۵۳MB.
- E4s موفق؛ ۶۲۷ Segment در `outputs/segmentation/E4s-1_30_50/news001.eaf`.
- Cache متنی BGE-M3 ساخته شده و در Pilot Workflow بررسی شد.
- `python -m pytest -q tests/test_pilot_workflow.py` → **۲ تست موفق**.
- `python scripts/pilot_workflow.py check ...` → **PASS / 0 errors**.
- `python scripts/pilot_workflow.py sea ...` → خروجی `news001.proposed.vtt` و `review.json`.
- بستهٔ `news001_inputs.zip` → حدود ۷۴ MiB ساخته شده.
- **DINOv2 کامل و Cache `outputs/features/news001.npz` هنوز در لاگ‌ها تأیید نشده**؛ بدون آن Training نباید آغاز شود.
- خروجی ترجمهٔ آموزش‌دیده، Checkpoint VLP و ارزیابی مستقل هنوز وجود ندارند.

## محل نصب Patch

این ZIP برای استخراج از داخل پوشهٔ `Wiser_Lab-ai` طراحی شده؛ ساختار فایل‌ها از `Lab_project/sea_starter/...` آغاز می‌شود.

```bash
cd ~/wiser/Wiser_Lab-ai
unzip -l ~/Downloads/research_sign_before_first_train_patch.zip
unzip -o ~/Downloads/research_sign_before_first_train_patch.zip -d .
cd Lab_project/sea_starter
source .venv/bin/activate
```

**پیش از unzip:** اگر فایل هم‌نامی را خودت ویرایش کرده‌ای، از آن Backup بگیر؛ این Patch فقط فایل‌های جدیدی با مسیرهای بالا دارد و فایل‌های داده یا Cache را حذف نمی‌کند.

## گام A — تست سبک روی مک (بدون GPU)

```bash
cd ~/wiser/Wiser_Lab-ai/Lab_project/sea_starter
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q tests/test_translation_first.py
python scripts/pilot_workflow.py check --id news001 --config configs/pilot_mac.yaml
python scripts/pilot_workflow.py sea --id news001 --config configs/pilot_mac.yaml
```

تست‌های جدید **هیچ مدلی از Hugging Face دانلود نمی‌کنند**. برای این فاز نیازی به اجرای مجدد MediaPipe، E4s یا DINOv2 روی مک نیست. روی مک دستور `train` را اجرا نکن؛ در این نسخه برای Training واقعی CUDA الزامی است.

> **خروجی‌های تست این Patch در محیط ایزوله:** ۶ تست واحد با Stub وابستگی‌های پروژه پاس شدند و تمام فایل‌های Python با `compileall` بررسی شدند. این **به معنای تأیید اجرای mT5، GPU یا Training روی دیتاست اصلی نیست**؛ بعد از نصب در مخزن خودت `pytest` و تست GPU باید انجام شوند.

## گام B — آماده‌سازی دیتاست واقعی و Splitهای مستقل

بعد از استخراج کامل Featureهای DINOv2 روی سرور/Colab:

```text
data/videos/<id>.mp4
data/subtitles_raw/<id>.vtt
data/poses/<id>.pose
outputs/segmentation/E4s-1_30_50/<id>.eaf
outputs/features/<id>.npz
outputs/features/<id>.json
outputs/text_features/bge_m3/<id>.npz
outputs/text_features/bge_m3/<id>.json

data/train_ids.txt
data/val_ids.txt
data/test_ids.txt
```

داخل فایل‌های Split، **شناسه‌های ویدیویی متفاوت** بنویس؛ نه یک فایل را با چند نام تکرار کنی، نه بخش‌های برش‌خوردهٔ یک Recording اصلی را بین Splitها پخش کنی. Train و Val و Test باید از منبع تصویربرداری مستقل باشند. برای ارزیابی تعمیم به Signer ناشناخته، جداسازی Signer نیز لازم است.

Config VLP فعلی با `batch_size: 2` به **حداقل ۲ ویدیوی اصلی متفاوت در Train، ۲ ویدیو در Val و ۱ ویدیو در Test** نیاز دارد. این حداقل **فنی** است و برای ادعای دقت علمی کافی نیست.

اگر Metadata آماده باشد، یک CSV مثل زیر بساز:

```csv
video_id,signer_id,source_recording_id
news001,signer01,recording01
news002,signer02,recording02
```

در اولین آزمایش دانشگاه، این دستور را قبل از مصرف GPU اجرا کن:

```bash
python scripts/pretrain_gate.py \
  --config configs/video_text_pretrain_first.yaml \
  --app-config configs/baseline.yaml \
  --metadata data/video_metadata.csv \
  --strict-signer
```

اگر هنوز Metadata نداری، می‌توانی فقط برای **تست فنی و Debug** پارامترهای `--metadata` و `--strict-signer` را حذف کنی؛ خروجی هشدار می‌دهد. در آزمایش نهایی مقاله، بدون Audit از استقلال Signer/Recording ادعای تعمیم نکن.

خروجی: `outputs/reports/pretrain_gate.json`. اگر `ok: false` باشد **Training را شروع نکن**. این اسکریپت حجم زیادی از NPZها را می‌خواند، اما وزن DINOv2/BGE-M3 را دانلود یا اجرا نمی‌کند.

## گام C — اولین Training VLP موجود در پروژه (سرور CUDA)

```bash
python -m pip install -e ".[dev]"
persian-sea-train-video-text \
  --config configs/video_text_pretrain_first.yaml
```

خروجی مورد انتظار:

```text
outputs/video_text/pretrain_first/best_video_text.pt
outputs/video_text/pretrain_first/history.jsonl
```

این همان مسیر **Contrastive/MIL موجود** پروژه است، نه Decoder جدید. Config آزمایش ۱۰۰ Step حداکثر، ۸ Epoch، batch=2 دارد. اگر DataLoader به علت تعداد کم Signer/Video دادهٔ کافی نداشت باید Splitها را اصولی اصلاح کنی؛ تکرار مصنوعی ویدیوی یکسان راه‌حل نیست.

## گام D — Training تولید متن فارسی با mT5 (سرور CUDA)

```bash
python -m pip install -r requirements-translation.txt
python -m persian_sea.translation.cli check \
  --config configs/translation_mt5_first.yaml
python -m persian_sea.translation.cli train \
  --config configs/translation_mt5_first.yaml
```

این مرحله Checkpoint VLP را از `outputs/video_text/pretrain_first/best_video_text.pt` می‌خواند. VideoTower به‌صورت پیش‌فرض Freeze می‌شود. در ادامه فقط Adapter و Decoder آموزش می‌بینند. Training حتماً به GPU و Cache کامل نیاز دارد و در این نسخه بر CPU/macOS قفل شده تا تصادفی به مک فشار وارد نشود.

خروجی‌ها:

```text
outputs/translation/mt5_first/readiness.json
outputs/translation/mt5_first/history.jsonl
outputs/translation/mt5_first/last_translation.pt
outputs/translation/mt5_first/best_translation.pt
outputs/translation/mt5_first/tokenizer/
```

`best_translation.pt` برای Inference و Evaluation است؛ `last_translation.pt` حاوی Optimizer هم هست و Resume را پشتیبانی می‌کند:

```bash
python -m persian_sea.translation.cli train \
  --config configs/translation_mt5_first.yaml \
  --resume outputs/translation/mt5_first/last_translation.pt
```

اگر تعداد Step برنامهٔ قبلی تمام شده باشد Resume را فقط با Config برنامهٔ جدید و سیاست Versioning کنترل‌شده انجام بده؛ کد برای جلوگیری از اختلاف Config، تغییرات خاموش را رد می‌کند.

## گام E — ارزیابیِ وابسته به Cue در مجموعهٔ کنارگذاشته‌شده

```bash
python -m persian_sea.translation.cli eval \
  --config configs/translation_mt5_first.yaml \
  --checkpoint outputs/translation/mt5_first/best_translation.pt \
  --split val
```

گزارش شامل `teacher_forced_loss`, `corpus_bleu`, `corpus_chrf` و نمونه‌جمله‌های فارسی است.

**نکتهٔ علمی مهم:** این Evaluation از VTT برای انتخاب پنجرهٔ تصویری هر Cue استفاده می‌کند؛ بنابراین **امتیاز Continuous Video-to-Text بدون Subtitle نیست**. Test مستقل را فقط پس از تثبیت مدل و Hyperparameterها یک بار اجرا کن (`--split test`).

## گام F — استنتاج بدون VTT (سرور)

```bash
python -m persian_sea.translation.cli infer \
  --config configs/translation_mt5_first.yaml \
  --checkpoint outputs/translation/mt5_first/best_translation.pt \
  --video-id news001
```

فایل خروجی `outputs/translation/mt5_first/predictions/news001.json` شامل پنجره‌های زمانی و متن فارسی پیشنهادی است. در این مسیر **هیچ VTT، متن زیرنویس یا Gloss خوانده نمی‌شود** و فقط `outputs/features/news001.npz` استفاده می‌شود.

**محدودیت مهم:** پنجره‌ها از تعداد ثابت Segmentهای E4s (مثلاً ۳۲) ساخته می‌شوند. این‌ها الزاماً مرزهای واقعی جمله نیستند؛ Overlap و Repetition یا حذف معنا محتمل است. برای محصول نهایی باید Sentence Boundary Detection مستقل، Windowing بهتر و Evaluation بدون VTT اضافه شوند.

## چه کارهایی را عمداً وارد این نسخه نکرده‌ایم؟

- ST-GCN/ASGCN: فقط در صورت ضعف ثابت‌شدهٔ Pose Backend با Ablation مستقل.
- VQ-VAE/SOKE: نیازمند پیش‌آموزش حرکتی و شاید نمونه‌برداری متراکم‌تر.
- Neural Sign Actors Query Bottleneck: تا وقتی Memory در مدل ترجمه گلوگاه نشده، پیچیدگی اضافه لازم نیست.
- TwoStream Fusion: پس از Baseline و مقایسهٔ Pose-only / RGB-only / Motion.
- Positional Encoding داخلی Segment: Ablation کم‌هزینهٔ بعدی، نه تغییر در Baseline نخست.
- Oracle Coverage: ابزار موجود مخزن نیازمند VTT واقعاً **تأییدشدهٔ انسانی**؛ VTT خام را جای Ground Truth قرار نده.

## شرایط توقف قبل از اولین Training دانشگاه

- [ ] تمام Cacheهای Visual و Text، مدل یکسان، Shape یکسان و مقدار Finite دارند.
- [ ] Splitهای اصلی جدا هستند؛ Metadata از Signer/Recording نشت نمی‌کند.
- [ ] Pretrain Gate مقدار `ok: true` دارد.
- [ ] سرور GPU از طریق `nvidia-smi` در دسترس است.
- [ ] VLP Checkpoint بعد از Training مرحله اول وجود دارد.
- [ ] Translation Check مقدار `ok: true` دارد.
- [ ] حداقل یک Validation مستقل و چند نمونهٔ ترجمهٔ دستی بررسی شده.
- [ ] نسخهٔ Git Commit، Configها، کتابخانه‌ها و تقسیم داده در گزارش آزمایش ثبت شده‌اند.
