
## محدودهٔ دقیق اجرا

| موضوع | این Patch چه کاری کرده است؟ | چه کاری هنوز نیاز دارد؟ |
|---|---|---|
| Baseline قدیمی | `pose_rgb`, `pose_only`, `pose_rgb_motion` را حفظ می‌کند | اجرای Training واقعی روی GPU |
| RGB-only | Backend مستقل `rgb_only` اضافه می‌کند | مقایسهٔ R@1 و BLEU/chrF بعد از Training |
| ترتیب درون‌Segment | Backend مستقل `pose_rgb_positional` با Sinusoidal PE روی هر دو Temporal Transformer | ارزیابی تأثیر روی Validation |
| Audit داده | کنترل Split IDs، VTT و Cache بصری، شواهد GT در فایل JSON | GT انسانی + Metadata واقعی Signer + در صورت نیاز Text Cache Gate موجود |
| Frame-order probe | حساسیت Encoder خام به معکوس‌کردن ۸ نمونه را گزارش می‌کند | تکرار تست روی Checkpoint آموزش‌دیده؛ Probe اثبات کیفیت نیست |
| Continuous Eval | فقط JSON حاصل از `translation/infer.py` را می‌خواند و **بعد** Transcript مرجع را برای BLEU/chrF/ROUGE-L استفاده می‌کند | اول Checkpoint واقعی؛ پنجرهٔ ثابت هنوز مرزبندی جمله نیست |
| SEA/Semantic | تغییر نکرده؛ `pair_source` موجود برای Raw/SEA/Semantic/Approved فعال می‌ماند | Oracle IoU نیاز به GT تأییدشده دارد |
| Signer audit | ابزار موجود تغییر نکرده | تأمین `signer_metadata`، اجرای حالت strict |
| E4s segmentation | ابزار معیار Segment IoU و Boundary F1 موجود را نگه می‌دارد | Annotation انسانی در `data/segmentation_gt` |

## معماری جدید بدون تغییر Baseline

```text
MP4 -> MediaPipe / E4s / DINOv2 -> cached Pose + RGB
                                    |
                         selectable visual backend
                   +----------------+------------------+
                   |                |                  |
               rgb_only       pose_rgb (original)  pose_only (original)
                   |                |                  |
                   +---------- pose_rgb_motion (existing)
                   +---------- pose_rgb_positional (NEW)
                                    |
                       VideoTower + MIL/Contrastive
                                    |
                         Visual Adapter + mT5/mBART
                                    |
                 Infer from visual cache ONLY (no VTT)
                                    |
                      JSON windows + text predictions
                                    |
              concat/dedupe predictions (no references)
                                    |
                   read reference ONLY for evaluation
```

نکته: Backend `pose_rgb_positional` روی همان Pose/RGB Cacheهای فعلی کار می‌کند؛ DINOv2 دوباره استخراج نمی‌شود. این Backend چون ساختار مدل جدیدی دارد، Checkpoint مستقل می‌خواهد و نباید وزن‌های قبلی را با این نام اشتباه بارگذاری کرد.


(ممکن است برای mT5/mBART به وابستگی‌های دیگری مثل `sentencepiece` نیز نیاز باشد؛ از محیط Training قبلی خود استفاده کنید.)

## فاز ۱ — آماده‌سازی و ارزیابی

### ۱.۱ Audit سبک Cache و Split، روی مک

```bash
python scripts/audit_phase12.py --report outputs/reports/phase12_audit.json
```

این ابزار صرفاً وجود و ساختار فایل‌ها را بررسی می‌کند. عدم وجود GT را به‌عنوان Warning نشان می‌دهد؛ به‌معنی درست بودن GT نیست. `news001` تنها یک ویدیوست و جای Train/Val/Test مستقل را نمی‌گیرد.

### ۱.۲ کنترل کامل‌تر Cacheها و Signer

```bash
python scripts/pretrain_gate.py \
  --config configs/video_text_pretrain_first.yaml \
  --app-config configs/baseline.yaml
```

وقتی metadata معتبر Signer/Session دارید، `--metadata` و `--strict-signer` را اضافه کنید. این Guard برای گزارش نهایی پروژه مهم است.

### ۱.۳ سنجش E4s، SEA و Oracle

- مرز اشاره را به‌صورت دستی در `data/segmentation_gt/<video_id>.eaf` تهیه کنید و از دستور CLI فعلی `persian-sea-eval-segments --config configs/baseline.yaml` بهره بگیرید.
- بازه‌های زمانی جمله را مستقل و دستی در `data/subtitles_gt/<video_id>.vtt` ثبت کنید. Ground Truth نباید همان خروجی `SEA` یا `SemanticAligner` باشد.
- سپس `persian-sea-oracle-coverage --config configs/baseline.yaml --video-text-config configs/video_text_pretrain_first.yaml --mode both --ids data/val_ids.txt` را اجرا کنید.
- `scripts/audit_alignment_sources.py` از قبل برای بررسی چهار منبع وجود دارد. در Training، `pair_source` باید به‌طور صریح یکی از `raw`, `sea`, `semantic`, `approved` باشد؛ **تغییر این گزینه مدل را خودکار بهتر نمی‌کند**.

### ۱.۴ ارزیابی واقعیِ بدون VTT هنگام استنتاج

ابتدا Checkpoint ترجمه و Cache تصویری برای ویدیوهای Test ایجاد شود. برای هر ویدیو:

```bash
python -m persian_sea.translation.cli infer \
  --config configs/translation_mt5_first.yaml \
  --checkpoint outputs/translation/mt5_first/best_translation.pt \
  --video-id news001
```

سپس (تنها بعد از تولید پیش‌بینی‌ها) گزارش Full-video:

```bash
python scripts/eval_continuous.py \
  --ids data/test_ids.txt \
  --prediction-dir outputs/translation/mt5_first/predictions \
  --reference-dir data/subtitles_raw \
  --output outputs/reports/continuous_test.json
```

این اسکریپت مرجع را برای انتخاب Window به مدل نمی‌دهد. فقط بعد از خواندن فایل پیش‌بینی، متن VTT را کنار هم می‌گذارد و امتیاز **یک Transcript در هر ویدیو** را محاسبه می‌کند. روش merge محافظه‌کارانه است: فقط تکرار کامل یا همپوشانی دقیق کلمه‌ای را حذف می‌کند. BLEU/chrF خروجی ثابت-window ضعیف یا ناقص را حل نمی‌کند. برای ادعای «ترجمهٔ پیوسته» باید این معیار را **جدا از Cue-guided evaluation** گزارش کنید و نمونه‌های انسانی را هم ببینید.

> نکتهٔ مهم: `--reference-dir` برای ارزیابی می‌تواند Transcript مرجع مجاز باشد؛ صرف وجود VTT در بخش Evaluation، نشت اطلاعات نیست. نشت وقتی رخ می‌دهد که Timestamp یا متن مرجع **پیش از تولید پیش‌بینی** به مدل داده شود.

## فاز ۲ — Ablationهای قابل اجرا

### ۲.۱ بررسی حساسیت Encoder به ترتیب ۸ فریم، روی CPU

```bash
python scripts/probe_frame_order.py \
  --output outputs/reports/frame_order_probe.json
```

این Probe مدل تصادفی را بررسی می‌کند، نه Checkpoint آموزش‌دیده. هدف یافتن احتمال عدم حساسیت معماری به ترتیب فریم‌هاست.

### ۲.۲ Baselineهای Video–Text

برای هر Config یک Checkpoint و Output جدا داریم:

```bash
persian-sea-train-video-text --config configs/video_text_pretrain_first.yaml
persian-sea-train-video-text --config configs/video_text_phase12_rgb_only.yaml
persian-sea-train-video-text --config configs/video_text_phase12_pose_only.yaml
persian-sea-train-video-text --config configs/video_text_phase12_pose_rgb_motion.yaml
persian-sea-train-video-text --config configs/video_text_phase12_pose_rgb_positional.yaml
```

`pose_rgb`، `pose_only` و `pose_rgb_motion` از قبل در مخزن وجود دارند. Config و Backendهای تازه برای آزمایش قابل مقایسه اضافه شده‌اند. **این دستورها روی GPU و پس از کامل‌شدن Cache و داشتن حداقل چند ویدیو در Splitهای مستقل اجرا شوند؛ روی MacBook شروع نکنید.**

### ۲.۳ برای آزمایش ترجمه به Configهای جدید نیاز است

یک VLP Checkpoint با Backbone متفاوت را نمی‌توان با `translation_mt5_first.yaml` بدون تغییر `visual_config` و `visual_checkpoint` بارگذاری کرد. برای هر Backend آزمایشی این دو مسیر را در یک نسخهٔ جداگانه از Config ترجمه تنظیم کنید. `output_dir` و فایل Checkpoint باید منحصربه‌فرد باشند. مقایسه روی Split و Seed یکسان باشد.

### ۲.۴ mT5 و mBART

این دو Backend از قبل در مخزن‌اند؛ برای مقایسه از `configs/translation_mt5_first.yaml` و `configs/translation_mbart50_optional.yaml` استفاده کنید. این Patch Backend تولید متن را بازنویسی نمی‌کند. مقایسه فقط با Visual Checkpoint یکسان معنی دارد.

### ۲.۵ انتخاب Checkpoint

الان `best_translation.pt` بر مبنای کمترین Validation Loss انتخاب می‌شود. برای نسخهٔ پژوهشی باید معیار Generated Text هم از پیش تعریف شود؛ در این Patch حلقهٔ Train تغییر نکرده است، زیرا نیازمند تست مصرف حافظهٔ واقعی mT5/mBART و جلوگیری از استفاده از Test برای انتخاب مدل است. معیارهای Continuous Test هرگز مبنای انتخاب Checkpoint نباشند.

## چه چیزی عمداً هنوز پیاده‌سازی نشده؟

- **Sentence Boundary Detector آموزش‌دیده:** به مرزهای انسانی/Weak Label معتبر و یک معیار Boundary F1 مستقل نیاز دارد؛ Overlap+Merge ابزار ارزیابی اولیه است، نه جایگزین detector.
- **Bidirectional RGB↔Pose Cross-Attention:** فاز سوم بعد از Baseline.
- **Face Landmark Stream:** نیاز به تغییر Feature Cache و کنترل سازگاری قدیمی دارد.
- **Pseudo-gloss/CTC و Adaptive GCN:** بدون Gloss معتبر یا Baseline قابل سنجش، وارد مدل اصلی نمی‌شوند.
- **انتخاب بهترین Checkpoint براساس BLEU/chrF در حلقهٔ Train:** جداگانه، پس از اندازه‌گیری GPU.
- **آموزش واقعی روی دیتاست کامل:** در این محیط GPU و دیتاست شخصی کاربر فراهم نیست.

## فایل‌های تحویلی

- `src/persian_sea/model/builders.py`: رجیستر Backendهای جدید، حفظ قبلی‌ها
- `src/persian_sea/model/ablation_backends.py`: RGB-only و Intra-segment PE
- `scripts/audit_phase12.py`: Readiness ابتدایی Split/VTT/Cache/GT
- `scripts/eval_continuous.py`: ارزیابی Transcript از پیش‌بینی بدون VTT
- `scripts/probe_frame_order.py`: تشخیص حساسیت ترتیب نمونه‌ها
- `tests/test_phase12_readiness.py`: تست‌های CPU
- `configs/video_text_phase12_*.yaml`: چهار Config برای Ablation

**هدف این Patch آماده‌کردن آزمایش دو فاز اول است؛ بهبود کیفیت ترجمه فقط از طریق Training/Evaluation مستقل قابل اثبات است.**
