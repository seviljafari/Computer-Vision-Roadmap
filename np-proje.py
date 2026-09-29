import numpy as np

# ============================
# ۱. ساخت یک تصویر مصنوعی ۱۰×۱۰
# ============================
img = np.random.randint(0, 256, (10, 10))
print("📸 تصویر اصلی:")
print(img)
print(f"شکل: {img.shape}\n")

# ============================
# ۲. آمار تصویر
# ============================
print("📊 آمار تصویر:")
print(f"میانگین: {np.mean(img):.2f}")
print(f"میانگین (با keepdims): {np.mean(img, keepdims=True)}")
print(f"بیشینه: {np.max(img)}")
print(f"کمینه: {np.min(img)}")
print(f"انحراف معیار: {np.std(img):.2f}\n")

# ============================
# ۳. Broadcasting: افزایش روشنایی
# ============================
bright = img + 50
bright = np.clip(bright, 0, 255)
print("☀️ تصویر روشن‌تر:")
print(bright)
print(f"میانگین جدید: {np.mean(bright):.2f}\n")

# ============================
# ۴. Boolean Indexing: پیدا کردن پیکسل‌های روشن
# ============================
bright_pixels = img[img > 150]
print(f"💡 تعداد پیکسل‌های روشن (>۱۵۰): {len(bright_pixels)}")
print(f"مقادیر: {bright_pixels}\n")

# ============================
# ۵. Fancy Indexing: انتخاب سطر و ستون خاص
# ============================
rows = [2, 5, 7]
cols = [1, 3, 8]
selected = img[rows, :][:, cols]
print("🎯 انتخاب سطرها و ستون‌های خاص:")
print(selected)
print(f"شکل انتخاب شده: {selected.shape}\n")

# ============================
# ۶. برش (Cropping) با Slicing
# ============================
crop = img[2:7, 3:8]
print("✂️ برش تصویر:")
print(crop)
print(f"شکل برش: {crop.shape}\n")

# ============================
# ۷. تغییر اندازه (View vs Copy)
# ============================
# تغییر شکل (View)
reshaped = crop.reshape(5, 5)  # چون ۵×۵ هست
print("🔄 تغییر شکل (View):")
print(reshaped)
print(f"آیا View هست؟ {reshaped.base is crop}\n")  # True

# کپی گرفتن
copy_crop = crop.copy()
copy_crop[0, 0] = 999  # تغییر در کپی
print("📋 کپی گرفتن و تغییر در کپی:")
print("کپی:", copy_crop[0, 0])
print("اصل:", crop[0, 0])
print("تغییر در کپی روی اصل اثر نذاشت!\n")

# ============================
# ۸. هیستوگرام ساده (تعداد پیکسل‌ها در هر سطح روشنایی)
# ============================
hist, bins = np.histogram(img.flatten(), bins=10, range=(0, 256))
print("📊 هیستوگرام (۱۰ بازه):")
for i in range(10):
    print(f"  بازه {i+1}: {hist[i]} پیکسل")
print()

# ============================
# ۹. نرمال‌سازی با Broadcasting
# ============================
mean = np.mean(img, axis=1, keepdims=True)  # میانگین هر سطر
normalized = img - mean  # Broadcasting
print("📐 نرمال‌سازی هر سطر (کم کردن میانگین):")
print(normalized)
print(f"شکل: {normalized.shape}\n")

# ============================
# ۱۰. جمع‌بندی نهایی
# ============================
print("✅ پروژه NumPy با موفقیت انجام شد!")
print("📌 مباحث استفاده شده:")
print("  - ایجاد آرایه با randint")
print("  - آمار (mean, max, min, std)")
print("  - Broadcasting (افزایش روشنایی)")
print("  - Boolean Indexing (پیکسل‌های روشن)")
print("  - Fancy Indexing (سطر و ستون خاص)")
print("  - Slicing (برش)")
print("  - View vs Copy (تغییر شکل و کپی)")
print("  - هیستوگرام با histogram")
print("  - keepdims=True (نرمال‌سازی)")