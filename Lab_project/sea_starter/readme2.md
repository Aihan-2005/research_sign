# Iranian Sign Language Research — Stage 3 & GPU Training Guide

> **وضعیت:** نسخهٔ پژوهشی / در حال آماده‌سازی داده و آزمایش روی GPU  
> **مخزن:** https://github.com/Aihan-2005/research_sign  
> **شاخهٔ مورد بررسی:** `fix/continuous-alignment`  
> **Commit بررسی‌شده:** `206c4c7590d36a5f20a1f959bb8de0cfc914a190` (۱۵ اکتبر ۲۰۲۶)  
> **ریشهٔ پروژه:** `Lab_project/sea_starter/`  
> **نکته:** این سند وضعیت فعلی *کد موجود در GitHub* را از *برنامه‌ها و Patchهای پیشنهادی* جدا می‌کند. آزمایش نهایی روی ۴۰ ساعت داده هنوز در مخزن اثبات یا گزارش نشده است.

## 1. هدف پروژه و نوع داده

هدف بلندمدت، **تولید متن فارسی از ویدیوی زبان اشاره ایرانی** بدون نیاز به زیرنویس در مرحلهٔ استنتاج است. پروژه به‌صورت تدریجی پیش می‌رود:

1. **Subtitle Alignment:** اصلاح یا تأیید زمان شروع/پایان زیرنویس‌های موجود در ویدیوی پیوسته.
2. **Video–Text Representation Learning:** یادگیری نمایش مشترک برای ویدیوی اشاره و متن فارسی (Contrastive/MIL).
3. **Sign Language Translation (آینده):** اتصال Visual Encoder به Decoder تولید متن فارسی.
4. **Continuous Translation (آینده):** پردازش ویدیوی چنددقیقه‌ای بدون زیرنویس در زمان استنتاج.

**دادهٔ فعلی:** حدود ۴۰ ساعت ویدیو، احتمالاً ۴ تا ۵ Signer، فایل‌های زیرنویس فارسی VTT با ترتیب درست و Timestampهایی که تا حدی به کمک ELAN بازبینی شده‌اند. زمان‌های خام یا تقریبی را نباید با Ground Truth انسانیِ تأییدشده اشتباه گرفت.

### Crop ویدیو

Crop **مکانی** برای نگه‌داشتن مجری زبان اشاره توسط همکار انجام می‌شود. در Pipeline حاضر، Crop تازه‌ای پیاده‌سازی نمی‌کنیم. باید صورت، دست‌ها، تنه و فضای حرکت حفظ شود و هیچ متن زیرنویسِ حک‌شده در تصویر به ورودی Visual Encoder راه پیدا نکند. **مبدأ زمانی، مدت و FPS/زمان‌بندی فریم‌ها** باید پس از Crop بررسی و با VTT هماهنگ باشند. فایل MP4 اصلی را به‌عنوان نسخهٔ مرجع نگه دارید.

## 2. مقالات و انتخاب معماری

| پژوهش | آنچه برای ما مفید است | مرز کاربرد |
|---|---|---|
| SEA (2026) | Sign Segmentation و Global Dynamic Programming برای Alignment زیرنویس‌ها | مدل تولید متن نیست |
| Jang et al. (2025) | Learnable Alignment، Negative Sampling و Self-Training | قواعد ویژهٔ BSL را مستقیماً برای فارسی کپی نمی‌کنیم |
| SpaMo (2025) | Spatial/Motion Representation و اتصال به Language Model | عمدتاً روی نمونه‌های ویدیو–جمله، نه کشف مرز Episode |
| VTaMo (2026) | Local/Global Video–Text Alignment و OT | OT جایگزین مستقیم Timestamp Alignment نیست |
| SignThought (2026) | Soft Segmentation و Context طولانی در Translation | پیاده‌سازی کامل برای Baseline فعلی سنگین است |
| SSVP-SLT (2024) | Self-supervised Video Pretraining، Video–Text LSP و T5 Decoder | پیش‌آموزش کامل SignHiera منابع بسیار بیشتری می‌خواهد |
| Shitara & Ochiai (2026) | هشدار دربارهٔ نویز Label و اهمیت Gradient/Loss Weight | GCE/SCE را نمی‌توان بی‌واسطه جایگزین InfoNCE کرد |

**انتخاب فعلی:** نگه‌داشتن `Pose + DINOv2 RGB + BGE-M3 + Temporal Encoder + Contrastive/MIL` به‌عنوان اولین Baseline قابل‌آموزش. بعد از Benchmark، گزینه‌های `pose_only` و `pose_rgb_motion` (از Patch مرحلهٔ سوم) به‌صورت Ablation آزمایش می‌شوند. Decoder فارسی در فاز بعدی توسعه داده خواهد شد.

## 3. معماری فعلی و اجزای آن

```text
MP4 پیوسته (فقط Signer)                Persian VTT
          │                                │
          ▼                                ▼
  MediaPipe Pose                    Text Normalization
          │                                │
          ▼                                ▼
  E4s Sign Segmentation                 BGE-M3
          │                                │
          ▼                                ▼
 Pose + DINOv2 RGB                   Text Cache (.npz)
          │                                │
          ▼                                │
 Visual Cache (.npz)                     │
          └─────────────┬──────────────────┘
                        ▼
            Video–Text Dataset
       Candidate Spans around VTT
                        │
                        ▼
        Visual Encoder + Text Projector
           + Temporal Context Encoder
                        │
                        ▼
          MIL / Contrastive Training
                        │
                        ▼
             Video–Text Embeddings
                        │
                        ▼
             Future Persian Decoder
```

- `src/persian_sea/pose/`: تبدیل ویدیو به `.pose` توسط MediaPipe.
- `src/persian_sea/segmentation/`: قطعه‌بندی Sign و ذخیرهٔ EAF.
- `src/persian_sea/features/`: استخراج Featureهای Pose و نماهای DINOv2، همراه با Cache.
- `src/persian_sea/text/`: ساخت Embeddingهای BGE-M3 از VTT.
- `src/persian_sea/alignment/`: SEA Temporal Baseline و خروجی زمان‌بندی.
- `src/persian_sea/semantic/`: Semantic Span Aligner جداگانه.
- `src/persian_sea/video_text/`: Training مشترک ویدیو–متن با Contrastive/MIL و Evaluation نوع Retrieval.
- `src/persian_sea/model/builders.py`: انتخاب Visual Encoder از Registry.

**ماژولار بودن** یعنی قرارداد Shape/Mask/Timestamp خروجی Encoder پایدار باشد؛ نه اینکه تغییر وزن‌ها و بعد Hidden Dimension بدون بررسی سازگاری Checkpoint انجام شود.

## 4. وضعیت واقعی GitHub و کارهای باقی‌مانده

وضعیت بررسی‌شده در شاخهٔ `fix/continuous-alignment`:

| مورد | وضعیت |
|---|---|
| SEA Temporal Alignment، Semantic Alignment، Video–Text Model | فایل‌های کد موجودند |
| `configs/baseline.yaml` و `configs/video_text_warmup.yaml` | موجودند |
| Moduleهای Stage 3 شامل Manifest / Preflight / Motion | در بستهٔ پیشنهادی قبلی آماده شده‌اند، ولی **در Commit بررسی‌شده دیده نشدند** |
| `alignment/quality_gate.py` | **در شاخه وجود ندارد، درحالی‌که Pipeline از آن Import می‌کند**؛ برای اجرای Alignment باید رفع شود |
| `data/video_ids.txt` | در Commit فعلی خالی است |
| فایل‌های Train/Val/Test و دادهٔ واقعی | باید محلی یا روی سرور آماده شوند |
| اجرای واقعی ۴۰ ساعت و نتایج Training | هنوز تأیید نشده است |

همچنین در `pyproject.toml` فعلی، `persian-sea-train-video-text` به‌عنوان Console Script ثبت نشده است؛ فعلاً از `python -m persian_sea.cli.train_video_text` استفاده کنید. دستورهای `persian-sea-preflight` و `persian-sea-preprocess` نیز فقط **پس از ادغام Patch مرحلهٔ سوم** در دسترس‌اند.

## 5. فرمت و سازمان‌دهی دیتاست

```text
Lab_project/sea_starter/
├── data/
│   ├── videos/             # news001.mp4 (نسخهٔ Cropشدهٔ زمانی-همگام)
│   ├── subtitles_raw/      # news001.vtt
│   ├── subtitles_approved/ # اختیاری: زمان‌های بازبینی‌شده
│   ├── subtitles_gt/       # فقط Annotation واقعاً تأییدشده
│   ├── poses/              # خروجی .pose
│   ├── video_ids.txt
│   ├── train_ids.txt
│   ├── val_ids.txt
│   └── test_ids.txt
├── outputs/
│   ├── segmentation/
│   ├── features/
│   ├── text_features/
│   └── video_text/
└── configs/
```

`news001.mp4` و `news001.vtt` باید نام پایهٔ یکسانی داشته باشند. `data/video_ids.txt` شامل `news001` در یک خط است، بدون پسوند. هر VTT باید `WEBVTT`، متن UTF-8 و زمان‌های `HH:MM:SS.mmm` داشته باشد.

داده‌ها را روی **فضای ذخیره‌سازی محلی یا سرور** بگذارید، نه داخل GitHub عمومی. فایل‌های شخصی/ویدیو و خروجی‌های حجیم را در `.gitignore` نگه دارید.

**Split پژوهشی:** ویدیوهای مشتق‌شده از یک ضبط اصلی نباید در Splitهای متفاوت باشند؛ Signer ID و Recording ID را ثبت کنید. با فقط چند Signer، یک ارزیابی Leave-One-Signer-Out یا Signer-Unseen بسیار ارزشمند است ولی اندازهٔ نمونهٔ کوچک باید گزارش شود.

## 6. اجرای مراحل آماده‌سازی داده — دستورات فعلی مخزن

ابتدا مخزن را روی سیستم دارای داده یا سرور GPU بگیرید:

```bash
git clone --branch fix/continuous-alignment https://github.com/Aihan-2005/research_sign.git
cd research_sign/Lab_project/sea_starter
python -m pip install -e '.[dev]'
python -m pytest -q tests/
```

> ابزارهای بیرونی `video_to_pose`، `pose_to_segments`، فایل وزن E4s و مدل‌های دانلودشده، ممکن است به نصب/پیکربندی مستقل نیاز داشته باشند. صرف `pip install -e` موفقیت تمام Pipeline را تضمین نمی‌کند.

پس از گذاشتن MP4، VTT و `video_ids.txt`:

```bash
persian-sea-audit-data --config configs/baseline.yaml
persian-sea-pose --config configs/baseline.yaml
persian-sea-segment --config configs/baseline.yaml
persian-sea-scan-rgbpose --config configs/baseline.yaml
persian-sea-scan-text --config configs/baseline.yaml
```

خروجی‌های مهم برای `news001`:

- `data/poses/news001.pose`
- `outputs/segmentation/<run-name>/news001.eaf`
- `outputs/features/news001.npz`
- `outputs/text_features/news001.npz`

برای هر ویدیو کیفیت Pose، پوشش دست/صورت، اتصال فریم به Timestamp، تعداد Sign Segmentها، درصد Candidate Windowهای معتبر و سازگاری VTT/Text Cache را بررسی کنید. اگر ویدیوی Cropشده یا وزن استخراج ویژگی عوض شد، Cache قدیمی را بدون کنترل دوباره استفاده نکنید.

### Training اولیه — دستور واقعی موجود

```bash
python -m persian_sea.cli.train_video_text \
  --config configs/video_text_warmup.yaml
```

ارزیابی روی Test پس از انتخاب مدل با Validation:

```bash
python -m persian_sea.cli.eval_video_text \
  --config configs/video_text_warmup.yaml \
  --checkpoint outputs/video_text/best_video_text.pt
```

**مهم:** فایل Config فعلی `max_train_steps: 200` دارد. این مقدار سقف آزمایش مقدماتی است و آموزش کامل چند Epoch روی تمام ۴۰ ساعت داده نیست. بعد از پایلوت می‌توان با **یک Config جدید مخصوص Full Training** و تغییر `max_train_steps: 0` سقف را برداشت؛ اما فقط پس از تأیید Splitها، GPU، Dataset و اعتبارسنجی.


### اتصال به سرور دانشگاه از VS Code

1. از استاد/ادمین، `hostname`، نام کاربری، روش ورود و اطلاعات GPU/سهمیه بگیرید.
2. افزونهٔ **Remote - SSH** را در VS Code نصب کنید.
3. با `Remote-SSH: Connect to Host` به `user@hostname` متصل شوید.
4. پوشهٔ پروژه را **روی خود سرور** باز کنید؛ Python Environment و فایل‌های Dataset نیز باید روی سرور قابل دسترسی باشند.
5. در ترمینال سرور اجرا کنید:

```bash
nvidia-smi
python -c "import torch; print('CUDA:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

اگر PyTorch CUDA را تشخیص نداد، نسخهٔ CUDA-compatible متناسب با Driver/GPU سرور را طبق دستور نصب رسمی PyTorch/ادمین نصب کنید. **دستورات نصب CUDA را بدون بررسی روی سرور مشترک اجرا نکنید.**

### انتقال فایل‌ها به سرور

کد با Git منتقل می‌شود؛ ویدیوها و VTT از طریق SFTP، SCP یا rsync. نمونهٔ کلی (مسیر و نام سرور را جایگزین کنید):

```bash
# روی لپ‌تاپ اجرا شود (از پوشه‌ای که داده در آن است):
rsync -avP ./videos/ user@server.example.edu:/work/research_sign/Lab_project/sea_starter/data/videos/
rsync -avP ./subtitles_raw/ user@server.example.edu:/work/research_sign/Lab_project/sea_starter/data/subtitles_raw/
```

### اجرای مستقل با tmux (فقط اگر سرور اجازهٔ اجرای مستقیم Job بدهد)

```bash
# در ترمینال SSH روی سرور و داخل sea_starter:
tmux new -s sign_training
mkdir -p outputs/video_text
python -u -m persian_sea.cli.train_video_text \
  --config configs/video_text_warmup.yaml \
  2>&1 | tee outputs/video_text/train.log
```

برای خروج بدون قطع Training، ابتدا `Ctrl+B` و سپس `D` را فشار دهید. بعد از این می‌توانید VS Code را ببندید و لپ‌تاپ را خاموش کنید. برای بازگشت:

```bash
tmux attach -t sign_training
```

**شرایط:** سرور باید روشن بماند؛ اگر Job تمام شود، خطا کند، سرور Restart شود یا سهمیهٔ پردازش تمام شود، Training ادامه نخواهد یافت. `tmux` محافظ در برابر خاموش‌شدن سرور نیست. اگر دانشگاه Slurm دارد، به‌جای اجرای مستقیم روی Login Node، از `sbatch`/`srun` مطابق دستورالعمل همان مرکز استفاده کنید.

### Checkpoint و ادامهٔ آموزش

کد Video–Text فعلی هنگام بهترشدن معیار Validation، فایل `outputs/video_text/best_video_text.pt` می‌نویسد و گزینهٔ `--resume` نیز دارد. این قابلیت را برای بازیابی پس از **قطع ناگهانی در وسط Epoch** کاملاً مقاوم فرض نکنید: نسخهٔ فعلی فقط Best Checkpoint را ذخیره می‌کند و تمام وضعیت RNG و Sampler/AMP را برای ادامهٔ دقیق ثبت نمی‌کند. پیش از Training طولانی، ذخیرهٔ `latest.pt` در فواصل مشخص و وضعیت کامل Optimizer/Scheduler/Scaler/RNG باید بهبود یابد.

## 7. برآورد زمان برای ۴۰ ساعت — بدون ادعای عدد قطعی

**زمان کل = استخراج Pose + Sign Segmentation + استخراج DINOv2 + ساخت BGE-M3 Cache + Training + Validation.**

مرحلهٔ Pose ممکن است هر فریم ویدیو را پردازش کند. برای **مثال فرضی**ِ ۴۰ ساعت ویدیوی ۲۵fps:

- تعداد کل فریم‌ها: `40 × 3600 × 25 = 3,600,000` فریم.
- اگر سرعت *مؤثرِ کل استخراج Pose* برابر ۱۰ فریم/ثانیه باشد: حدود **۱۰۰ ساعت**.
- اگر ۳۰ فریم/ثانیه باشد: حدود **۳۳ ساعت**.
- اگر ۱۰۰ فریم/ثانیه باشد: حدود **۱۰ ساعت**.

این‌ها **سناریوی محاسباتی** هستند، نه سرعتی که برای GPU تو اندازه‌گیری شده باشد. Segmentation، DINOv2، I/O، مدل‌های دانلودی و Text Encoding به آن اضافه می‌شوند. Scanner فعلی برای **هر Sign Segment تعداد ۸ زمان × ۴ نمای تصویری = تا ۳۲ Crop تصویری** را به DINOv2 می‌دهد؛ بنابراین Feature Extraction ممکن است قسمت پرهزینه‌ای از کار باشد.

### زمان Training فقط با Feature Cache آماده

این مثال فرضی را در نظر بگیرید:

- ۴۰ ساعت، میانگین یک جمله در هر ۴ ثانیه ⇒ تقریباً **۳۶ هزار Cue**.
- ۸۰٪ داده برای Train ⇒ حدود **۲۸٬۸۰۰ Cue**.
- `batch_size=4` ⇒ در مدل ساده حدود **۷٬۲۰۰ Step در هر Epoch**.
- ۸ Epoch ⇒ حدود **۵۷٬۶۰۰ Step**.
- اگر هر Step در سرور واقعی **۱ ثانیه** باشد: حدود **۱۶ ساعت** فقط برای Stepهای آموزش.
- اگر هر Step **۳ ثانیه** باشد: حدود **۴۸ ساعت** فقط برای Stepهای آموزش.

**همهٔ اعداد این پاراگراف سناریو هستند، نه پیش‌بینی تأییدشده.** تعداد Cue واقعی، Video-diverse Batch Sampler، Cache/RAM، Validation، ذخیره و توقف زودهنگام این زمان را تغییر می‌دهند. روی سخت‌افزار ضعیف‌تر، زمان حتی ممکن است بیشتر باشد.

### روش علمی و کم‌هزینهٔ تخمین واقعی

1. یک ویدیوی ۵ تا ۱۰ دقیقه‌ای را کامل پردازش کنید و زمان و RAM/VRAM هر مرحله را ثبت کنید.
2. همین کار را روی حدود یک ساعت نمونهٔ نماینده از Signerها و Layoutهای مختلف تکرار کنید.
3. چند ویدیوی مستقل با Split معتبر آماده و **۲۰۰ Step** Training اجرا کنید؛ میانگین پایدارِ `steps/sec`، Memory GPU و مدت Validation را اندازه بگیرید.
4. تعداد واقعی Cueها و Stepهای هر Epoch را از Log و Sampler بگیرید.
5. تخمین بزنید: `زمان آموزش ≈ Stepهای کل ÷ steps_per_second + زمان Validation/Checkpoint`.
6. ۲۰ تا ۳۰٪ حاشیهٔ زمانی برای I/O، نوسان بار سرور و خطاهای احتمالی در برنامه‌ریزی پروژه بگذارید؛ این فقط حاشیهٔ اجرایی پیشنهادی است.

با این روش متوجه می‌شویم کار چند ساعت، چند روز یا بیشتر طول می‌کشد. گزارش Training مقاله‌های دیگر را نباید به ۴۰ ساعت داده و معماری ما تعمیم مستقیم داد.





## 8. تعریف دقیق «اتمام این مرحله»

Stage 3 و ورود به Stage 2 Training تنها وقتی آماده محسوب می‌شوند که:

1. دست‌کم یک نمونهٔ واقعی کامل تا Feature Cache پردازش شده باشد.
2. ویدیو/VTT/POSE/EAF با هم هم‌زمان و کیفیت حرکت دست و صورت قابل قبول باشند.
3. Train/Val/Test مستقل، Recording ID و Signer ID ثبت شده باشد.
4. Training چند Step بدون NaN و با GPU/CUDA قابل اجرا باشد.
5. Log سرعت، مصرف VRAM، Loss، Retrieval Validation و Checkpoint ذخیره شود.
6. زمان‌بندی اولیه یا SEA بدون تأیید، به‌عنوان GT قطعی استفاده نشده باشد.

**خلاصه:** لپ‌تاپ ابزار توسعه است؛ سرور GPU محل مناسب اجرای محاسبات طولانی. ابتدا Pilot انجام می‌دهیم، زمان و حافظه را اندازه می‌گیریم، سپس تصمیم می‌گیریم آموزش کل ۴۰ ساعت با چه Config، GPU و تعداد Epoch اجرا شود.
