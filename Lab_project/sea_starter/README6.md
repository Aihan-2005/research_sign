
## ۱) کنترل تأخیر زیرنویس و Sign

در `VideoTextConfig` مقدار پیش‌فرض زیر اضافه شده:

```yaml
subtitle_to_sign_offset_seconds: 2.0
subtitle_offsets_by_video: {}
```

مقدار مثبت یعنی Sign بعد از timestamp زیرنویس اتفاق می‌افتد و پنجره‌ی جست‌وجوی Candidate به جلو منتقل می‌شود.

برای تغییر عمومی:

```yaml
subtitle_to_sign_offset_seconds: 3.0
```

برای تغییر یک ویدیوی خاص:

```yaml
subtitle_to_sign_offset_seconds: 2.0
subtitle_offsets_by_video:
  news001: 3.0
  news007: 1.5
```

این Offset فقط وقتی `pair_source: raw` باشد اعمال می‌شود. اگر `pair_source` روی `approved`، `sea` یا `semantic` باشد، timestampها از قبل اصلاح‌شده فرض می‌شوند و Offset جدید اعمال نمی‌شود تا تأخیر دوبار اضافه نشود.

timestamp اصلی VTT تغییر نمی‌کند؛ فقط بازه‌ای که برای پیدا کردن Candidateهای تصویری استفاده می‌شود جابه‌جا می‌شود.

برای برآورد مقدار واقعی، فایل نمونه‌ی `examples/offset_landmarks.example.csv` را کپی کن، نقاط را با مشاهده‌ی دستی پر کن و اجرا کن:

```bash
python scripts/calibrate_subtitle_offset.py \
  --input data/offset_landmarks.csv \
  --output outputs/reports/offset_calibration.json
```

این اسکریپت میانه و MAD را برای کل داده و هر ویدیو گزارش می‌کند و Config را خودکار تغییر نمی‌دهد.

**نکته‌ی Checkpoint:** Checkpointهای قبلی که با کد بدون Offset آموزش دیده‌اند، در provenance معادل Offset صفر در نظر گرفته می‌شوند. برای استفاده از مقدار جدید ۲ ثانیه باید مدل Video-Text دوباره آموزش داده شود؛ در غیر این صورت کد به‌جای مقایسه‌ی نادرست، خطای ناسازگاری تنظیمات می‌دهد.

## ۲) Null Alignment اختیاری در SEA

در `baseline.yaml` این دو گزینه اضافه شده‌اند:

```yaml
allow_null_alignment: false
null_alignment_penalty: 30.0
```

برای آزمایش:

```yaml
alignment:
  allow_null_alignment: true
  null_alignment_penalty: 30.0
```

در این حالت DP می‌تواند برای بعضی Cueها هیچ Sign Segment انتخاب نکند. Quality Gate برای آن Cue تصمیم `no_proposal` ثبت می‌کند و timestamp اصلی را نگه می‌دارد.

این قابلیت فقط برای SEA DP است، نه Semantic/SAT decoder. مقدار penalty باید با GT دستی روی Validation تنظیم شود؛ فعلاً پیش‌فرض خاموش است تا Baseline تغییر نکند.

همچنین اگر `alignment.similarity_weight` غیرصفر شود، Pipeline خطا می‌دهد؛ چون در وضعیت فعلی ماتریس Semantic Similarity به DP داده نمی‌شود و تغییر صرف وزن، هیچ اثری نداشت.

## ۳) آزمایش Cross-Attention بین Pose و RGB

Backend جدید `pose_rgb_cross_attention` اضافه شده و Backend اصلی `pose_rgb` دست‌نخورده مانده است.

Config جدید:

`configs/video_text_phase12_pose_rgb_cross_attention.yaml`

اجرا:

```bash
persian-sea-train-video-text \
  --config configs/video_text_phase12_pose_rgb_cross_attention.yaml
```

این Backend به‌صورت دوطرفه Pose tokens را به RGB tokens و RGB tokens را به Pose tokens توجه می‌دهد. Checkpoint جداگانه لازم دارد و نباید با Checkpoint مدل `pose_rgb` بارگذاری شود.

## ۴) اسکریپت‌های ارزیابی پیش از تغییر معماری

ترتیب Subtitle و زمان GT را بررسی کن:

```bash
python scripts/audit_alignment_order.py \
  --config configs/baseline.yaml \
  --ids data/val_ids.txt
```

امکان‌پذیری طول CTC را با tokenizer و تعداد Candidateهای واقعی بسنج:

```bash
python scripts/audit_ctc_feasibility.py \
  --config configs/translation_mt5_first.yaml
```

این اسکریپت CTC head آموزش نمی‌دهد؛ فقط شرط لازم طول را بررسی می‌کند:

`input_steps >= target_tokens + adjacent_repeat_count`

## ۵) پیشنهادهایی که فعلاً نباید مستقیم وارد مدل اصلی شوند

- **SAT token-level Query Alignment:** کش فعلی فقط sentence embedding دارد، نه token embeddings و masks. همچنین به GT زمانی مستقل برای BCE نیاز داریم.
- **Global reorder-aware DTW:** ابتدا باید نرخ inversion را با GT اندازه بگیریم. Semantic decoder فعلی و Quality Gate بر اساس ترتیب cueهای اصلی طراحی شده‌اند؛ تغییر decoder به‌تنهایی کافی نیست.
- **MOT:** علاوه بر token cache، candidate bag فعلی شامل Segmentهای اطراف و احتمالاً نامرتبط است؛ OT متوازن ممکن است مجبور شود Segmentهای نامرتبط را به کلمات وصل کند. ابتدا باید partial/unbalanced OT یا span supervision معتبر طراحی شود.
- **Translation CTC:** حداکثر ۳۲ visual segment در برابر حداکثر ۹۶ target token داریم. قبل از head باید گزارش feasibility را اجرا کنیم. واژگان CTC نیز باید فقط از Train ساخته شود.
- **HMTF:** طول زمانی Sign Segmentها یکسان نیست؛ convolution روی index به‌تنهایی فاصله‌های زمانی متفاوت را یکسان فرض می‌کند.
- **Knowledge Distillation:** فعلاً دو شاخه‌ی Teacher/Student مستقل نداریم.

## ۶) تست‌ها

تست‌های جدید:

- `tests/test_temporal_offset.py`
- `tests/test_null_alignment.py`
- `tests/test_cross_attention_backend.py`

فایل‌های Python از نظر Syntax بررسی شده‌اند. Forward ماژول Cross-Attention، مسیر Null DP و accessor تنظیم Offset به‌صورت smoke test اجرا شده‌اند. اجرای کامل pytest، Training واقعی و ارزیابی روی داده‌های شخصی/سرور GPU هنوز باید در محیط پروژه انجام شود.
