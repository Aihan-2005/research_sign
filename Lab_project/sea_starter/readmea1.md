# Iranian Sign Language Research — Continuous Subtitle Alignment and Video–Text Learning

> **وضعیت پروژه:** در حال توسعه و آزمایش پژوهشی (Research Prototype)  
> **مخزن:** [Aihan-2005/research_sign](https://github.com/Aihan-2005/research_sign)  
> **شاخهٔ مستندشده:** [`fix/continuous-alignment`](https://github.com/Aihan-2005/research_sign/tree/fix/continuous-alignment)  
> **آخرین Commit بررسی‌شده:** `3ba72ad50315174b7181bf7699301248192466b2` — ۱۲ اکتبر ۲۰۲۶  
> **محل پروژه:** `Lab_project/sea_starter/`

## ۱. معرفی و هدف پروژه

این پروژه به بررسی هم‌ترازی زمانی و معنایی ویدیوهای **زبان اشاره ایرانی** با زیرنویس‌های **فارسی** می‌پردازد و هدف بلندمدت آن توسعهٔ سامانهٔ **Sign Language Translation (SLT)** برای تولید متن فارسی از ویدیوی زبان اشاره است.

دادهٔ در اختیار ما حدود **۴۰ ساعت ویدیوی پیوسته** است. تعداد افراد اجراکنندهٔ زبان اشاره هنوز قطعی نیست، اما تخمین اولیه حدود **۴ تا ۵ نفر** است. ویدیوها زیرنویس فارسی دارند؛ ترتیب جمله‌ها صحیح در نظر گرفته شده و زمان شروع/پایان از زیرنویس اولیه استخراج و سپس با بازبینی انسانی و ELAN تا حدی اصلاح شده است. برخی زمان‌ها دقیق یا نزدیک به دقیق‌اند و بعضی ممکن است چند ثانیه اختلاف داشته باشند.

در این پروژه دو مسئله باید کاملاً از هم تفکیک شوند:

1. **Subtitle-to-Video Alignment:** متن زیرنویس معلوم است؛ باید بازهٔ زمانی اجرای اشارهٔ متناظر را در ویدیوی پیوسته پیدا یا دقیق‌تر کنیم.
2. **Video-to-Persian Translation:** هنگام استنتاج فقط ویدیو داده می‌شود و مدل باید متن فارسی تولید کند؛ این مرحله هنوز پیاده‌سازی و آموزش کامل نشده است.

هدف آن است که در عین حفظ ویدیوهای اصلی و توالی زمانی، Featureها را Cache کنیم و معماری را طوری توسعه دهیم که تعویض مدل‌های Segmentation، Visual Encoder، Text Encoder، Aligner و Decoder باعث بازنویسی کل Pipeline نشود.

## ۲. وضعیت فعلی به تفکیک مرحله

| مرحله | آنچه در مخزن وجود دارد | وضعیت واقعی |
|---|---|---|
| ورودی و کنترل کیفیت داده | خواندن MP4/VTT/SRT و کنترل زمان‌بندی؛ قرارداد Episode | پیاده‌سازی شده؛ اجرای کامل روی دادهٔ واقعی گزارش نشده |
| استخراج Pose | Runner برای MediaPipe و فایل `.pose` | پیاده‌سازی شده؛ به ابزار و دادهٔ واقعی نیاز دارد |
| تشخیص Sign Segment | E4s/SEA-compatible Segmentation و EAF | پیاده‌سازی شده؛ دقت روی زبان اشاره ایرانی باید سنجیده شود |
| SEA Baseline | `original` و `sea_temporal`، DP زمانی و گزارش جابه‌جایی | کد موجود؛ یک Import ناقص در Commit فعلی مانع اجرای برخی مسیرهاست |
| Semantic Alignment | Pose+RGB، متن BGE-M3، Candidate Span، Contrastive/MIL، Global Decoder | کد آموزش و استنتاج موجود؛ نتیجهٔ نهایی روی ۴۰ ساعت تأیید نشده |
| Video–Text Warm-up | پوشهٔ `video_text/` با Contrastive Learning، Retrieval و Checkpoint | به مخزن اضافه شده؛ تست‌های ساختگی وجود دارند؛ آموزش واقعی هنوز نیازمند اجرا و ارزیابی است |
| تولید متن فارسی از ویدیو | مدل Seq2Seq و Translation Decoder | **مرحلهٔ آینده؛ هنوز در کد حاضر نیست** |

**مهم:** وجود کد، Config و Unit Test به‌معنای انجام آموزش روی ۴۰ ساعت یا اثبات کیفیت مدل نیست. نتایج آزمایش و Benchmark باید جداگانه ثبت شوند.

## ۳. پژوهش‌ها و مبنای انتخاب معماری

شش کار در تصمیم‌گیری معماری بررسی شده‌اند:

| پژوهش | مسئلهٔ اصلی | نقش در پروژه |
|---|---|---|
| **SEA — Segment, Embed, and Align (ACL 2026)** | یافتن مرز زیرنویس‌ها در ویدیوی پیوسته با Segmentation، Embedding اختیاری و DP | Baseline اصلی Alignment در سطح Episode |
| **Deep Understanding of Sign Language for Sign to Subtitle Alignment (Jang et al., 2025)** | Alignment متن‌محور و یادگرفتنی، Selective Loss و Self-training | ایده برای توسعهٔ Aligner قابل‌آموزش و استفاده از برچسب‌های ضعیف |
| **SpaMo (NAACL 2025)** | Gloss-free SLT با ویژگی‌های Spatial و Motion، VT-Align و مدل زبانی | الهام برای Visual–Text Warm-up و مرحلهٔ آیندهٔ Translation |
| **VTaMo (2026)** | Alignment محلی/سراسری با Optimal Transport هنگام آموزش ترجمه | گزینهٔ تحقیقاتی برای بهبود تطبیق ریزدانه بعد از ساخت Baseline |
| **SignThought (2026)** | ترجمه با Latent Thought و Soft Temporal Segmentation | ایدهٔ پیشرفته برای آزمایش‌های بعدی؛ نه Baseline فعلی |
| **3D CNN + LSTM برای ASL** | تشخیص علامت‌های منفرد | مرجع جانبی؛ برای Alignment جمله‌ای یا ترجمهٔ پیوسته معماری اصلی انتخاب نشده |

**چرا SEA؟** چون در مرحلهٔ اول متن و ترتیب زیرنویس‌ها از پیش معلوم‌اند و مسئله، اصلاح زمان‌بندیِ آن‌ها روی Episode است. روش `Segment and Align` در SEA یک Baseline بدون آموزش گرادیانیِ جدید فراهم می‌کند؛ برخلاف SpaMo/VTaMo که عمدتاً مدل‌های ترجمه روی نمونه‌های ویدیو–جمله هستند. این انتخاب به معنی بهترین‌بودن SEA روی زبان اشاره ایرانی نیست؛ باید با Timestamp اولیه و مدل یادگرفتنی مقایسه شود.

در مقالهٔ SEA، مدل Segmentation و Embedding می‌توانند از پیش‌آموزش‌دیده باشند و **DP به Training گرادیانی نیاز ندارد**. زمان اجرای DP پس از آماده‌شدن ورودی‌ها را نباید با هزینهٔ استخراج Pose و Embedding از فایل ویدیویی اشتباه گرفت. انتقال مدل‌های آموزش‌دیده روی زبان‌های اشارهٔ دیگر به زبان اشاره ایرانی هنوز نیازمند ارزیابی تجربی است.

## ۴. داده‌ها و فرمت‌های ورودی

فرمت‌های استاندارد پیشنهادی و مورد پشتیبانی مسیرهای فعلی:

- ویدیوی اصلی: ترجیحاً `MP4`؛ بعضی Readerها `MOV`/`MKV`/`WebM` را هم پیدا می‌کنند.
- زیرنویس: **`VTT` پیشنهاد می‌شود**؛ `SRT` نیز خوانده می‌شود. متن فارسی باید UTF-8 باشد.
- Pose استخراج‌شده: `.pose`، مطابق `pose-format`.
- Annotation در ELAN و Sign Segmentها: `.eaf`.
- Feature Cache: `.npz` برای Pose، RGB و بردار متن؛ گزارش‌های همراه می‌توانند `.json` باشند.
- خروجی Alignment: `.vtt` و در صورت فعال‌بودن، `.eaf` و `.review.json`.

نمونهٔ VTT:

```vtt
WEBVTT

00:00:12.100 --> 00:00:15.400
امروز دربارهٔ آموزش صحبت می‌کنیم.

00:00:16.200 --> 00:00:19.800
این یک نمونه از زیرنویس فارسی است.
```

نام پایهٔ فایل‌ها باید یکسان باشد؛ به‌طور مثال `ep001.mp4`، `ep001.vtt` و `ep001.pose`. همهٔ زمان‌ها باید به یک مبنای زمانیِ روشن نسبت به ویدیوی اصلی ارجاع دهند. اگر برای آزمایش یک Episode سه‌دقیقه‌ای از ویدیوی بلند تعریف شود، فایل اصلی و ترتیب زیرنویس‌ها حفظ می‌شوند؛ Featureها و Timestampها به بازهٔ زمانی مربوطه اشاره می‌کنند. از بریدن مستقل هر جمله و از دست‌دادن Context به‌عنوان پیش‌فرض پروژه خودداری می‌شود.

ساختار دادهٔ مورد انتظار در مسیر `Lab_project/sea_starter`:

```text
data/
├── videos/                # ep001.mp4, ...
├── subtitles_raw/         # ep001.vtt, ... : زمان اولیه / نسبتاً دقیق
├── subtitles_approved/    # اختیاری: فقط زمان‌های بازبینی‌شده و تأییدشده
├── subtitles_gt/          # فقط مرجع واقعی برای ارزیابی/نظارت معتبر
├── poses/                 # ep001.pose, ...
├── segmentation_gt/       # اختیاری: Annotationهای معتبر مرزهای اشاره
├── video_ids.txt
├── train_ids.txt
├── val_ids.txt
└── test_ids.txt
```

`subtitles_raw` را نباید بدون بررسی به‌عنوان Ground Truth دقیق تلقی کرد. خروجی پیشنهادی خود مدل نیز نباید بدون کنترل کیفیت وارد `subtitles_gt` یا Test Set شود. در مرحلهٔ Video–Text می‌توان با `pair_source: raw` از زمان‌های تقریبی استفاده کرد یا با `pair_source: approved` از زیرنویس‌هایی استفاده کرد که قبلاً مستقل و دستی تأیید شده‌اند.

## ۵. معماری کل سیستم

```text
Continuous Video (MP4)             Persian Subtitle Sequence (VTT)
          |                                      |
          v                                      v
  MediaPipe Holistic Pose                 Subtitle Parser
          |                                      |
          v                                      +--> BGE-M3 frozen text encoder
   E4s Sign Segmentation                          |       |
          |                                        |       v
          v                                        |   Text feature cache (.npz)
        EAF                                       |
          |                                      |
          +--> [B0] Original timestamps           |
          |                                      |
          +--> [B1] SEA temporal DP <-------------+
          |              |
          |         Proposed / Applied VTT + review JSON / EAF
          |
          +--> DINOv2 four-view RGB + normalized Pose features
                         |
                         v
                  Visual feature cache (.npz)
                         |
             +-----------+----------------------+
             |                                  |
             v                                  v
    Semantic Alignment pipeline           Video–Text Stage 2
    (candidate spans + contrastive        (Pose+RGB + temporal context,
     / optional verified supervision      multi-scale MIL + contrastive)
     + monotonic decoder)                       |
             |                                  v
     Proposed / applied VTT                 Shared video–text embeddings
                                                |
                                                v
                                  Future: temporal adapter + Persian
                                         text-generation decoder
```

**نکته:** مدل جدید Video–Text در مرحلهٔ فعلی **متن تولید نمی‌کند**. خروجی آن Embedding و نمرهٔ تطبیق/بازیابی ویدیو–متن است؛ مرحلهٔ Video-to-Persian Translation در آینده اضافه می‌شود.

### ۵.۱. Baselineهای Alignment

- **B0 — Original:** استفاده از Timestampهای فعلی بدون اصلاح؛ برای سنجش ارزش افزودهٔ هر روش.
- **B1 — SEA Temporal:** به‌کارگیری Sign Segmentها، هزینهٔ فاصلهٔ زمانی/مدت/Gap و Dynamic Programming با قید ترتیب جملات.
- **B2 — Learned Semantic Alignment:** ویژگی‌های Pose+RGB و متن BGE-M3، تولید Candidate Span در اطراف زمان تقریبی، امتیاز معنایی و Temporal Prior، و انتخاب سراسری با Decoder یکنواخت از نظر ترتیب.

دلیل حفظ B0 آن است که بخشی از زمان‌های داده از قبل مناسب‌اند. تغییر بی‌مورد آن‌ها می‌تواند عملکرد را بدتر کند. Pipeline برای همین، **`proposed.vtt`، نسخهٔ محافظه‌کارانهٔ VTT و گزارش بازبینی** را از هم جدا می‌کند. آستانهٔ بازبینی و فعال‌بودن اصلاح خودکار باید روی دادهٔ تأییدشده تعیین شوند؛ Score مدل لزوماً احتمال اطمینان کالیبره‌شده نیست.

### ۵.۲. مسیر Visual Features

از Sign Segmentهای تشخیص‌داده‌شده، چند Timestamp نمونه‌برداری می‌شود؛ مختصات Pose و چهار Crop تصویری **signer، left_hand، right_hand و face** به Feature Extractor داده می‌شوند. مدل Frozen DINOv2-Base برای ویژگی‌های تصویری، و Graph-based Pose Encoder به‌همراه Temporal Encoder و Gated Fusion برای یادگیریِ ارتباط بین نمای بدنی و RGB استفاده می‌شوند.

Crop در این مرحله **مکانی** است (دست/صورت/فرد) و به معنای شکستن ویدیو به فایل‌های مستقل بر اساس جمله نیست. ویژگی‌ها به همراه بازه‌های شروع/پایان Sign Segmentها Cache می‌شوند.

### ۵.۳. مدل Semantic Alignment

ماژول `semantic/` یک راهکار قابل‌آموزش برای زمان‌بندی متن روی ویدیو دارد. زیرنویس‌ها Candidateهای نزدیک Timestamp اولیه را دریافت می‌کنند؛ نمایش Signها Contextualize می‌شود؛ Candidate Spanهای پیوسته با بردار متن مقایسه می‌شوند؛ Lossهای Contrastive/MIL و در صورت داشتن **GT واقعاً تأییدشده**، Supervised Span Loss به‌کار می‌رود؛ خروجی به Global Decoder منتقل می‌شود. این مسیر از نظر مسئله با Stage 2 تفاوت دارد: خروجی Semantic Aligner **بازهٔ زمانی زیرنویس** است.

### ۵.۴. مرحلهٔ جدید Video–Text Representation Learning

کد مرحلهٔ دوم در `src/persian_sea/video_text/` قرار دارد و مستقل از SEA نگهداری می‌شود:

- **Video Tower:** استفادهٔ مجدد از Visual Encoder ماژولار موجود؛ Tokenهای Sign Segment به یک Temporal Transformer سبک می‌روند؛ ابعاد خروجی پیش‌فرض ۲۵۶ است.
- **Text Tower:** تبدیل Featureهای ازپیش‌محاسبه‌شدهٔ BGE-M3 به همان فضای Embedding.
- **Candidate Window + Multi-scale Spans:** به‌جای فرض یک برش زمانی کاملاً دقیق، چند بازه با اندازهٔ ۱، ۲، ۴ و ۸ Sign Segment بررسی می‌شوند.
- **MIL / Log-Mean-Exp:** تطبیق متن با محتمل‌ترین Spanهای ویدیویی و کنترل اثر تعداد متفاوت Candidateها.
- **Contrastive Loss:** نزدیک‌کردن زوج‌های مرتبط و دورکردن زوج‌های نامرتبط؛ نمونه‌های هم‌ویدیو و متن کاملاً تکراری به‌عنوان Negative در همان Batch استفاده نمی‌شوند.
- **Video-diverse Batch Sampler:** ساخت Batch از ویدیوهای متمایز برای ایجاد Negativeهای مفید.
- **Evaluation:** Retrieval@1 و Retrieval@5 و Loss روی مجموعهٔ مستقل؛ نتایج Validation پیش‌فرض روی یک زیرمجموعه از نمونه‌ها محاسبه می‌شود.

این فاز مشابه ایدهٔ Visual–Text Warm-up در SpaMo است ولی **پیاده‌سازی کامل SpaMo، VTaMo، Sinkhorn OT یا Decoder ترجمه نیست**.

## ۶. ماژولار بودن و امکان تعویض مدل

هدف طراحی این است که تغییر مدل در یک بخش، بخش‌های دیگر را از کار نیندازد:

- `core/contracts.py`: قراردادهای `Episode`، `TimeSpan`، `AlignmentProposal` و Interface مربوط به Aligner.
- `core/registry.py`: سازوکار ثبت و انتخاب Backend.
- `alignment/backends.py`: انتخاب `original` یا `sea_temporal`.
- `model/builders.py`: انتخاب Visual Encoder، فعلاً `pose_rgb` و `pose_only`.
- `semantic/decoder_registry.py`: انتخاب Global Decoder، از جمله `monotonic_dp`.
- `video_text/config.py`: Config مجزا برای آموزش Video–Text.
- `io/`: جداکردن I/O و فرمت فایل از منطق مدل.
- `tests/`: تست قرارداد داده، مدل، Loss، Alignment و مسیرهای ترکیبی با ورودی ساختگی.

برای جایگزینی Visual Encoder در آینده (مثلاً CLIP/VideoMAE) باید یک Backend سازگار با قرارداد ورودی/خروجی و نسخهٔ Feature Cache تعریف شود. صرف عوض‌کردن نام مدل در Config بدون کنترل ابعاد Feature و سازگاری Checkpoint کافی نیست.

## ۷. ساختار اصلی کد

```text
Lab_project/sea_starter/
├── configs/
│   ├── baseline.yaml
│   └── video_text_warmup.yaml
├── data/
│   └── video_ids.txt
├── src/persian_sea/
│   ├── core/               # contracts, registry
│   ├── data/               # Episode handling/audit
│   ├── pose/               # video -> MediaPipe pose
│   ├── segmentation/       # E4s, EAF, evaluation
│   ├── features/           # DINOv2, Pose, crops, scanner, cache, audio utility
│   ├── text/               # frozen BGE-M3 and cached text vectors
│   ├── io/                 # VTT/SRT/EAF
│   ├── model/              # visual encoder, pose, RGB, fusion, backend factory
│   ├── alignment/          # SEA baseline, costs, DP, pipeline
│   ├── semantic/           # learned subtitle alignment, loss, train, infer, eval
│   ├── video_text/         # NEW: video-text retrieval representation learning
│   └── cli/                # command line entry points
└── tests/
```

> ساختار بالا بخش‌های اصلی Source را نشان می‌دهد؛ همهٔ فایل‌های میانی یا دادهٔ واقعی در مخزن Commit نشده‌اند.

## ۸. نصب و اجرای پیشنهادی

پیش‌نیازهای کلی: Python نسخهٔ ۳٫۱۰ به بالا، نصب وابستگی‌های `pyproject.toml`، `ffmpeg/ffprobe` برای بعضی بررسی‌ها، ابزارهای Pose و بستهٔ Segmentation سازگار با SEA، و برای پردازش عملی GPU انویدیا با PyTorch مناسب. وزن‌های مدل و اینترنت یا Cache وزن‌های ازپیش‌دانلودشده نیز مورد نیازند.

```bash
cd Lab_project/sea_starter
python -m pip install -e '.[dev]'
python -m pytest -q tests/
```

### ۸.۱. آماده‌سازی فهرست ویدیوها

فایل `data/video_ids.txt` باید نام پایهٔ ویدیوها را خط‌به‌خط داشته باشد، مانند:

```text
ep001
ep002
```

فایل‌های Split جداگانهٔ `train_ids.txt`، `val_ids.txt` و `test_ids.txt` برای مرحلهٔ آموزش ضروری‌اند. **در Snapshot بررسی‌شدهٔ GitHub فایل `video_ids.txt` خالی بود و Splitها در فهرست مخزن نبودند**؛ احتمالاً باید در محیط دادهٔ واقعی ساخته شوند.

### ۸.۲. کنترل کیفیت و پیش‌پردازش

```bash
# بررسی وجود و زمان‌بندی فایل‌های ویدیو و زیرنویس
persian-sea-audit-data --config configs/baseline.yaml

# استخراج Pose
persian-sea-pose --config configs/baseline.yaml

# تولید E4s Sign Segmentation در قالب EAF
persian-sea-segment --config configs/baseline.yaml

# استخراج ویژگی‌های RGB/Pose و Cache کردن
persian-sea-scan-rgbpose --config configs/baseline.yaml

# استخراج و Cache بردارهای متن فارسی با BGE-M3
persian-sea-scan-text --config configs/baseline.yaml
```

این فرمان‌ها به وجود فایل‌های MP4/VTT واقعی، وزن‌های مدل و ابزارهای خارجی مناسب وابسته‌اند. استخراج Feature ممکن است برای ۴۰ ساعت ویدیو زمان‌بر باشد؛ در شروع یک Pilot کوچک اجرا شود.

### ۸.۳. اجرای SEA Baseline

برای B0 در `configs/baseline.yaml` مقدار `alignment.backend` را `original` و برای B1 مقدار را `sea_temporal` قرار دهید. سپس:

```bash
persian-sea-align --config configs/baseline.yaml
```

خروجی در `outputs/aligned_subtitles/` قرار می‌گیرد و شامل نسخهٔ پیشنهادی، نسخهٔ نهاییِ محافظه‌کارانه و گزارش تصمیم‌هاست. پیش‌فرض `alignment.auto_apply_small_shifts: false` است؛ بنابراین اجرای مدل نباید به‌طور ضمنی Timestampهای اصلی را Ground Truth جدید فرض کند.

**مانع اجرایی شناخته‌شدهٔ Commit فعلی:** فایل `src/persian_sea/alignment/quality_gate.py` در شاخه وجود ندارد، با اینکه `alignment/pipeline.py` و `semantic/infer.py` آن را Import می‌کنند. این وابستگی باید بازیابی/اضافه و تست شود تا فرمان‌های Alignment مربوطه قابل اجرا باشند.

### ۸.۴. Video–Text Warm-up

Config پیش‌فرض `configs/video_text_warmup.yaml` برای **Pilot** است و `max_train_steps: 200` دارد؛ بنابراین اجرای همان Config به معنای آموزش کامل ۴۰ ساعت نیست.

```bash
python -m persian_sea.cli.audit_signers \
  --config configs/video_text_warmup.yaml \
  --metadata data/metadata.csv

python -m persian_sea.cli.train_video_text \
  --config configs/video_text_warmup.yaml

python -m persian_sea.cli.eval_video_text \
  --config configs/video_text_warmup.yaml \
  --checkpoint outputs/video_text/best_video_text.pt
```

حداقل دو ویدیوی **اصلی متمایز** برای Contrastive Loss لازم است؛ Config نمونه `batch_size: 4` دارد و اجرای Training فعلی به دست‌کم چهار ID ویدیویی متمایز در Train احتیاج دارد. صرف جداکردن یک ضبط طولانی به چند Episode با نام جدید، استقلال واقعی نمونه‌ها را تضمین نمی‌کند.

خروجی Stage 2: Checkpoint به نام `best_video_text.pt`، گزارش `history.jsonl`، و آمار Retrieval. این مدل هنوز متن فارسی آزاد تولید نمی‌کند. دستورهای Stage 2 فعلاً به‌صورت `python -m ...` در دسترس‌اند؛ CLIهای Stage 2 در `[project.scripts]` فعلی ثبت نشده‌اند.

## ۹. Split داده و مسئلهٔ ۴ تا ۵ Signer

باید حداقل این ستون‌ها برای هر ویدیو ثبت شود:

```csv
video_id,signer_id,source_recording_id
ep001,signer_01,recording_A
ep002,signer_02,recording_B
```

اصل ضروری: قسمت‌های مربوط به **یک Source Recording** در Train و Validation/Test پخش نشوند. اگر هدف سنجش تعمیم به افراد جدید است، یک آزمایش Signer-independent نیز طراحی شود. وقتی تعداد افراد فقط حدود ۴ تا ۵ نفر باشد، تقسیم‌های Signer-independent محدودیت آماری دارند؛ نتایج باید با ذکر تعداد Signer و روش Split گزارش شوند.

`video_text/split_audit.py` بررسی هم‌پوشانی ویدیوی اصلی، شناسهٔ ضبط و در حالت سخت‌گیرانه تداخل Signer را انجام می‌دهد. در صورتی که Signerها هنوز ناشناخته‌اند، ابتدا فرایند ثبت Metadata لازم است.

## ۱۰. ارزیابی و آزمایش‌های موردنیاز

### Alignment (دقیق‌شدن زمان زیرنویس)

روی یک مجموعهٔ کوچک با Timestamp **مستقل و دستی تأییدشده**، B0/B1/B2 را با معیارهای زیر مقایسه کنید:

- `mean IoU` و نرخ درست‌بودن در آستانه‌های IoU مانند ۰٫۵.
- خطای مطلق شروع و پایان برحسب ثانیه (`Start MAE` و `End MAE`).
- درصد زیرنویس‌هایی که SEA واقعاً بهتر/بدتر کرده است.
- پایداری در برابر جابه‌جایی عمدی زمان اولیه (مثلاً ۲، ۴ و ۶ ثانیه).
- Ablation حذف Temporal Prior برای سنجش اتکای واقعی به معنی تصویر/متن.

### Video–Text Learning

- Retrieval@1 و Retrieval@5 با Negativeهای واقعاً مستقل، نه فقط Batchهای آسان.
- مقایسهٔ `pose_rgb` با `pose_only` و در ادامه RGB-only و Motion.
- کنترل نمونه‌های متن یکسان یا بسیار شبیه برای جلوگیری از False Negative.
- بررسی Overfitting روی Signer و Session، و تحلیل کیفی نزدیک‌ترین بازیابی‌ها.

**BLEU/chrF و ارزیابی ترجمه فقط پس از ساخت مدل تولید متن معنی دارند**؛ نباید Retrieval Accuracy را به‌عنوان دقت ترجمه معرفی کرد.

### وضعیت تست‌ها

پوشهٔ `tests/` شامل Unit Test و Smoke Test برای قراردادها، Candidateها، Global Decoder، مدل‌های قابل‌تعویض، سناریوهای EAF/VTT و Stage 2 است. تست Stage 2 از دادهٔ ساختگی و یک `ToyVisual` برای بررسی Backpropagation و Checkpoint استفاده می‌کند. **در بررسی این Commit تست‌ها روی سخت‌افزار دیتاست واقعی دوباره اجرا نشده‌اند**؛ وجود تست مساوی با موفقیت تست End-to-End و Training GPU نیست.

## ۱۱. محدودیت‌ها و کارهای باز

1. **اصلاح Import شکستهٔ `alignment/quality_gate.py`** و افزودن تست Import/Smoke روی Commit فعلی.
2. `README.md` اصلی فعلاً خالی است؛ این فایل، مستند موقت دقیق وضعیت پروژه است.
3. داده، شناسه‌های Split، Feature Cacheها، Checkpointها و نتایج آموزش واقعی در Snapshot حاضر قابل مشاهده نیستند؛ باید از محیط محاسباتی فراهم شوند.
4. دقت E4s و Encoderهای پیش‌آموزش‌دیده روی **زبان اشاره ایرانی** هنوز تضمین نشده است.
5. تعداد و هویت Signerها و شناسهٔ Session باید تکمیل شوند و خطر Data Leakage کنترل شود.
6. برای متن‌های کم‌اطلاعات، تکراری یا زمان‌های نادقیق، Contrastive Learning ممکن است Negative/Positive اشتباه بسازد؛ داده باید ممیزی شود.
7. مسیر موجود `video_text/` برای **Embedding/Retrieval** است؛ هنوز Decoder فارسی، Translation Loss و Continuous Video-to-Text Inference وجود ندارد.
8. ماژول Audio Utility موجود است، اما Audio هنوز به‌عنوان سیگنال آموزش معتبر برای مرزهای اشاره ادغام نشده؛ هم‌زمانی گفتار و اشاره باید تجربی بررسی شود.
9. فایل‌های تولیدشده مانند `.DS_Store`، `.pyc` و `egg-info` نباید در ادامه به Git اضافه شوند؛ `.gitignore` و پاک‌سازی مخزن بررسی شود.

## ۱۲. نقشهٔ راه دقیق ادامهٔ پروژه

**گام ۰ — قابل‌اجراکردن و ممیزی Repository.** بازیابی `quality_gate.py`، اجرای `pytest`، آزمایش دستورات CLI و ثبت نسخه‌های ابزارها و مدل‌ها؛ آماده‌کردن فهرست ویدیوها و Metadata.

**گام ۱ — Pilot روی ۱ تا ۲ ساعت داده.** از MP4/VTT واقعی، Pose و EAF و ویژگی‌های DINOv2/BGE-M3 بسازید. Video↔Pose FPS/Timestamp و نمایش Crop دست/صورت را دستی بررسی کنید. یک گزارش زمان پردازش، VRAM و خطا ذخیره شود.

**گام ۲ — ساخت مجموعهٔ کوچک مرجع در ELAN.** نمونه‌های متنوع از Signer، طول جمله و کیفیت زیرنویس را با مرزهای مستقلِ تأییدشده مشخص کنید؛ GT را از خروجی خود SEA نسازید.

**گام ۳ — گزارش Baseline Alignment.** B0 (original) و B1 (SEA) و سپس B2 (semantic) را روی مرجع تأییدشده با MAE/IoU و نرخ بهتر/بدترشدن مقایسه کنید. اصلاحات را فقط در صورت ارزش افزودهٔ قابل‌اندازه‌گیری بپذیرید.

**گام ۴ — اجرای Video–Text Warm-up.** ابتدا Config آزمایشی ۲۰۰ Step را روی GPU واقعی اجرا و RAM/VRAM/سرعت Step را ثبت کنید؛ سپس بر اساس نتیجه، Epoch و Batch Size و تعداد Step را برای دادهٔ Train افزایش دهید. بهترین Checkpoint با Validation مستقل انتخاب شود.

**گام ۵ — آزمون تعمیم و Ablation.** Retrieval روی Test مستقل، بررسی Signer/Session Leakage، مقایسه Pose-only و Pose+RGB، و کنترل تکیه بر Timestamp اولیه.

**گام ۶ — توسعهٔ Translation.** استفادهٔ مجدد از Video Encoder به همراه Temporal Adapter و Decoder فارسیِ Seq2Seq (مثلاً mT5 به‌عنوان Baseline) با Teacher Forcing و Cross-Entropy، و ارزیابی BLEU/chrF و بازبینی انسانی. سپس برای ویدیوی پیوستهٔ بدون زیرنویس، Boundary Detection/Continuous Translation مستقل طراحی شود. بخش‌های SpaMo، VTaMo و SignThought فقط پس از وجود Baseline قابل‌اعتماد به‌صورت Ablation وارد شوند.

## ۱۳. منابع اصلی

- [SEA — Official Repository](https://github.com/J22Melody/SEA)
- [SEA — ACL Anthology](https://aclanthology.org/2026.acl-long.1401/)
- **Deep Understanding of Sign Language for Sign to Subtitle Alignment** — Jang et al. (2025).
- **SpaMo: An Efficient Gloss-Free Sign Language Translation Using Spatial Configurations and Motion Dynamics with LLMs** — NAACL 2025.
- **VTaMo: Video-Text Alignment Model for Sign Language Translation** — 2026.
- **Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation (SignThought)** — ACL 2026.
- **Real-Time ASL Recognition Using 3D CNN & LSTM** — مرجع جانبی تشخیص واژه.

---

**خلاصه:** ساختار ماژولار Baseline Alignment و Video–Text Warm-up در مخزن ایجاد شده است، اما کار پژوهشی تعیین‌کنندهٔ بعدی **اجرای صحیح روی دادهٔ واقعی، ارزیابی دقیق Timeline و اندازه‌گیری Retrieval** است. ادعای عملکرد نهایی یا آموزش موفق ۴۰ ساعت تا پیش از ثبت نتایج، مطرح نمی‌شود.
