#  Classical Computer Vision
ترجمه: بینایی کامپیوتری کلاسیک
# 📌 چه فرقی با Deep Learning داره؟
# Classical CV	                 
روش‌های ریاضی	
دستی طراحی میشن	
سریع‌تر	
برای کارهای ساده	
قبل از ۲۰۱۲

# Deep Learning
روش‌های یادگیری
خودشون یاد می‌گیرن
کندتر (نیاز به GPU)
برای کارهای پیچیده
بعد از ۲۰۱۲

# 📌 مثال:
تشخیص چهره با Classical CV:
از فیلترهای Haar استفاده می‌کنه
خودت باید ویژگی‌ها رو طراحی کنی
سریع‌تره
تشخیص چهره با Deep Learning:
از شبکه‌های عصبی استفاده می‌کنه
شبکه خودش ویژگی‌ها رو یاد می‌گیره
دقیق‌تره

# 📌 چرا هنوز Classical CV مهمه؟
سریع‌تره (برای Real-time)
سبک‌تره (روی موبایل و دستگاه‌های ضعیف)
پایه‌ی Deep Learning هست
توی صنعت هنوز استفاده میشه

# 📌  Classical Computer Vision — زیرشاخه‌ها

# ۰۴.۱ — Feature Detection (تشخیص ویژگی‌ها)
Harris Corner	تشخیص گوشه
SIFT	     مقاوم به چرخش و مقیاس
ORB	            سریع‌تر از SIFT
FAST	       تشخیص سریع گوشه
Blob Detection	تشخیص لکه
# ۰۴.۲ — Feature Matching (تطابق ویژگی‌ها)
BFMatcher	  تطابق ساده
FLANN	       تطابق سریع
Ratio Test	فیلتر تطابق‌های اشتباه
# ۰۴.۳ — Geometric Transformations (تبدیلات هندسی)
Homography	           تبدیل صفحه به صفحه
Perspective Transform	تغییر پرسپکتیو
Affine Transform	  چرخش، مقیاس، انتقال
Image Stitching    چسباندن تصاویر (پانوراما)
# ۰۴.۴ — Camera Calibration (کالیبراسیون دوربین)
Intrinsic Parameters	پارامترهای داخلی
Extrinsic Parameters	پارامترهای خارجی
Distortion Correction	تصحیح اعوجاج لنز
Chessboard Calibration	کالیبراسیون با صفحه شطرنجی
# ۰۴.۵ — Object Tracking (ردیابی اشیا)
Optical Flow	  ردیابی حرکت پیکسل‌ها
MeanShift	    ردیابی بر اساس رنگ
CamShift	    نسخه بهبودیافته MeanShift
KLT Tracker    	ردیابی نقاط کلیدی


--------------------------------------
#  📌 شروع از اول: ۰۴.۱ — Feature Detection
# 📌 عنوان: Harris Corner
--------------------------------------
# 📌 ۱. توضیح کلی:
Harris Corner یه روش کلاسیک برای تشخیص گوشه‌ها توی تصویره.
اسمش از کجا اومده؟
Harris = اسم کاشفش (Chris Harris)
Corner = گوشه
# 📌 ۲. گوشه چیه؟
مربع:
┌─────────┐
│         │
│         │
│         │
└─────────┘
گوشه‌ها: چهار نقطه‌ی بالا-چپ، بالا-راست، پایین-چپ، پایین-راست
# 📌 اول: تصویر چیه؟
مثلاً یه تصویر ۵×۵:
10   20   30   40   50
60   70   80   90   100
110  120  130  140  150
160  170  180  190  200
210  220  230  240  250
هر عدد = روشنایی یه نقطه از تصویر.
۰ = سیاه
۲۵۵ = سفید
# 📌 حالا گوشه چیه؟
10   20   30
40   50   60
70   80   90
# حالا گوشه‌ی این جدول کجاست؟
گوشه‌ی بالا-چپ = عدد ۱۰
گوشه‌ی بالا-راست = عدد ۳۰
گوشه‌ی پایین-چپ = عدد ۷۰
گوشه‌ی پایین-راست = عدد ۹۰
اینا میشن گوشه. ✅
# 📌 چرا گوشه مهمه؟
فرض کن می‌خوای این جدول رو توی یه جدول بزرگ‌تر پیدا کنی.
اگه وسط جدول رو نگاه کنی → همه‌ی اعداد شبیه همن
اگه گوشه‌ها رو نگاه کنی → راحت پیداش می‌کنی
پس گوشه = نشونه‌ی خوب برای پیدا کردن.
# 📌 Harris Corner چیکار می‌کنه؟
میاد توی تصویر می‌گرده و همه‌ی گوشه‌ها رو پیدا می‌کنه.
# ✅ حالا سؤال:
به نظرت اگه یه عکس از یه ساختمون داشته باشیم، Harris Corner چه چیزایی رو پیدا می‌کنه؟
گوشه های ساختمان رو فقط پیدا میکنه برای پیدا گردن همه گوشه های رو ساختماان مثل پنجره از هیرکورنیر پیشرفته ترینمتد رو داریم
# 📌 سؤالت:
Harris    	پایه‌ای، سریع، ولی مقاوم به مقیاس نیست
SIFT	   مقاوم به چرخش و مقیاس (دقیق‌تر)
ORB	         سریع‌تر از SIFT
FAST	   خیلی سریع (فقط گوشه)
# 📌 پس مسیر ما اینه:
Harris  →  SIFT  →  ORB  →  FAST
(پایه)    (دقیق)   (سریع)   (سریع‌تر)
# 📌 ۱. اول کد رو ببین:
import cv2
import numpy as np

# ۱. خواندن تصویر
img = cv2.imread("E:/cp-vision/face_test.jpg")

# ۲. تبدیل به خاکستری
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ۳. تبدیل به float32
gray = np.float32(gray)

# ۴. اعمال Harris
corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# ۵. بزرگ کردن نقاط
corners = cv2.dilate(corners, None)

# ۶. مشخص کردن گوشه‌ها روی تصویر
img[corners > 0.01 * corners.max()] = [0, 0, 255]

# ۷. نمایش و ذخیره
cv2.imshow("Harris Corner", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("E:/cp-vision/harris_corners.jpg", img)
print("✅ تصویر Harris ذخیره شد.")

# 📌 ۲. خط به خط توضیح می‌دم (خیلی ساده):
خط ۱: import cv2
وارد کردن OpenCV

خط ۲: import numpy as np
وارد کردن NumPy

خط ۳: img = cv2.imread(...)
خوندن تصویر

خط ۴: gray = cv2.cvtColor(...)
تبدیل تصویر رنگی به خاکستری (چون Harris فقط با خاکستری کار می‌کنه)

خط ۵: gray = np.float32(gray)
تبدیل اعداد به اعشاری (Harris فقط با اعشاری کار می‌کنه)

خط ۶: corners = cv2.cornerHarris(gray, 2, 3, 0.04)
قلب ماجرا! این خط گوشه‌ها رو پیدا می‌کنه.
gray	تصویر خاکستری
2	اندازه پنجره (blockSize)
3	اندازه کرنل (ksize)
0.04	عدد k
خط ۷: corners = cv2.dilate(corners, None)
گوشه‌ها رو بزرگ‌تر می‌کنه تا بهتر دیده بشن.
خط ۸: img[corners > 0.01 * corners.max()] = [0, 0, 255]
گوشه‌های قوی رو قرمز می‌کنه.
0.01 = آستانه (هر چی بزرگ‌تر، گوشه کمتر)
[0, 0, 255] = رنگ قرمز
خط ۹: نمایش و ذخیره
تصویر رو نشون میده و ذخیره می‌کنه.

-------------------------
# 📌 قسمت ۲: SIFT
-------------------------
# 📌 عنوان: SIFT چیست؟
SIFT یه روش دقیق‌تر از Harris Corner برای پیدا کردن ویژگی‌هاست.
اسمش مخففه:
S = Scale (مقیاس)
I = Invariant (مستقل)
F = Feature (ویژگی)
T = Transform (تبدیل)
یعنی: روشی که ویژگی‌ها رو مستقل از اندازه پیدا می‌کنه.
# 📌 ۲. مشکل Harris Corner چی بود؟
مقاوم به مقیاس نبود
اگر تصویر بزرگ یا کوچیک بود نمیتونه درست تشخیص بده گوشه هارو
# 📌 ۳. مثال ساده:
فرض کن یه گوشه‌ی ساختمون داری:
عکس نزدیک: گوشه خیلی بزرگ دیده میشه
عکس دور: گوشه خیلی کوچیک دیده میشه
Harris: «این دو تا گوشه فرق دارن!» ❌
SIFT: «این دو تا یه گوشه‌ان!» ✅
# 📌 ۴. SIFT چیکار می‌کنه؟
SIFT میاد:
توی مقیاس‌های مختلف تصویر رو بررسی می‌کنه
جهت هر نقطه رو پیدا می‌کنه
یه توصیفگر برای هر نقطه می‌سازه
# 📌 ۵. مقایسه Harris و SIFT:
ویژگی	       Harris	SIFT
مقاوم به چرخش	✅	✅
مقاوم به مقیاس	❌	✅
مقاوم به نور	❌	✅
کند سریع            سرعت
بالا  متوسط            دقت
# 📌 چرا SIFT کندتره؟ (ساده)
# 🎯 داستان چیه؟
Harris فقط یه کار می‌کنه:
یه بار تصویر رو نگاه می‌کنه
گوشه‌ها رو پیدا می‌کنه
تموم!
SIFT چند تا کار می‌کنه:
تصویر رو توی چند تا اندازه مختلف نگاه می‌کنه (مثل ذره‌بین)
برای هر نقطه، جهت رو حساب می‌کنه
برای هر نقطه، یه بردار ۱۲۸ عددی می‌سازه
این کارها زمان بیشتری می‌بره
# 📌 مثال ساده:
Harris مثل اینه که یه عکس بگیری و بگی: «گوشه‌هاش کجاست؟»
SIFT مثل اینه که:
از ۵ تا فاصله مختلف عکس بگیری
از ۴ تا زاویه مختلف عکس بگیری
همه رو مقایسه کنی
بعد بگی: «گوشه‌هاش کجاست؟»
طبیعیه که SIFT کندتره! (چون کار بیشتری می‌کنه)
# 📌 کد SIFT
import cv2

# ۱. خواندن تصویر
img = cv2.imread("E:/cp-vision/face_test.jpg")

# ۲. تبدیل به خاکستری
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ۳. ساخت SIFT
sift = cv2.SIFT_create()

# ۴. پیدا کردن نقاط کلیدی
keypoints, descriptors = sift.detectAndCompute(gray, None)

# ۵. چاپ تعداد
print(f"تعداد نقاط کلیدی: {len(keypoints)}")

# ۶. رسم نقاط
img_kp = cv2.drawKeypoints(img, keypoints, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ۷. نمایش و ذخیره
cv2.imshow("SIFT", img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("E:/cp-vision/sift.jpg", img_kp)

# 📌 ۲. خط به خط:
خط ۱: import cv2
وارد کردن OpenCV
خط ۲: img = cv2.imread(...)
خوندن تصویر
خط ۳: gray = cv2.cvtColor(...)
تبدیل به خاکستری
خط ۴: sift = cv2.SIFT_create()
ساخت یه شیء SIFT (مثل یه کارآگاه که آماده‌ست)
خط ۵: keypoints, descriptors = sift.detectAndCompute(gray, None)
قلب ماجرا! SIFT می‌گرده و پیدا می‌کنه:
خروجی	چیه؟
keypoints	لیست نقاط کلیدی
descriptors	توصیفگر هر نقطه
خط ۶: print(f"تعداد: {len(keypoints)}")
چاپ تعداد نقاط
خط ۷: cv2.drawKeypoints(...)
رسم نقاط روی تصویر
پارامتر	چیه؟
img	تصویر اصلی
keypoints	نقاط پیدا شده
None	تصویر خروجی
flags=...	نحوه رسم
📌 ۳. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS چیه؟
نقاط رو با دایره‌های بزرگ رسم می‌کنه:
اندازه دایره = مقیاس نقطه
خط داخل دایره = جهت نقطه
پس فقط یه نقطه ساده نیست، اطلاعات بیشتری داره.
# 📌 چرا دایره؟
چون دایره دو تا اطلاعات رو نشون میده:
ویژگی دایره	معنی
اندازه دایره	مقیاس نقطه (چقدر بزرگ/کوچیک دیده شده)
خط داخل دایره	جهت نقطه (زاویه)

-------------------
# 📌 قسمت ۳: ORB
-------------------
# 📌 عنوان: ORB چیست؟
# 📌 ۱. توضیح کلی:
ORB یه روش سریع‌تر از SIFT برای پیدا کردن ویژگی‌هاست.
اسمش مخففه:
O = Oriented (جهت‌دار)
R = Rotated (چرخیده)
B = BRIEF (توصیفگر دودویی)
یعنی: ترکیب روش FAST (تشخیص سریع) و BRIEF (توصیفگر سریع)
# 📌 ۲. چرا ORB ساخته شد؟
مشکل SIFT:
دقیقه ولی کنده ❌
برای ویدیو زنده مناسبه نیست
راه‌حل ORB:
سریع‌تر از SIFT ✅
رایگان (SIFT پتنت داره) ✅
برای Real-time عالیه ✅
مقاوم به چرخش ✅
# 📌 ۴. تفاوت توصیفگر:
SIFT: توصیفگر اعشاری (۱۲۸ عدد float)
ORB: توصیفگر دودویی (۰ و ۱)
نتیجه: ORB سبک‌تره و سریع‌تره.
# ❓ سؤال:
ORB برای چه کاربردهایی مناسبه؟
ویدیو زنده، موبایل، رباتیک
# 📌 ۱. کد:
import cv2

# ۱. خواندن تصویر
img = cv2.imread("E:/cp-vision/face_test.jpg")

# ۲. تبدیل به خاکستری
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ۳. ساخت ORB ← این خط جا افتاده بود!
orb = cv2.ORB_create()

# ۴. پیدا کردن نقاط کلیدی
keypoints, descriptors = orb.detectAndCompute(gray, None)

# ۵. چاپ تعداد
print(f"تعداد نقاط کلیدی: {len(keypoints)}")

# ۶. رسم نقاط
img_kp = cv2.drawKeypoints(img, keypoints, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ۷. نمایش و ذخیره
cv2.imshow("ORB", img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("E:/cp-vision/orb.jpg", img_kp)
print("✅ تصویر ذخیره شد.")

# 📌 ۳. مقایسه کد SIFT و ORB:
SIFT	                    ORB
sift = cv2.SIFT_create()	orb = cv2.ORB_create()
sift.detectAndCompute(...)	orb.detectAndCompute(...)
فقط اسمشون فرق داره، بقیه کد یکیه!

# 📌 توضیح خط به خط (فقط چیزای جدید)

خط ۳: orb = cv2.ORB_create()
قسمت	توضیح
orb	اسم متغیر (خودت انتخاب می‌کنی)
cv2	کتابخانه OpenCV
.ORB_create()	تابعی که یه شیء ORB می‌سازه
کارش چیه؟
یه کارآگاه می‌سازه که آماده‌ست بره دنبال نقاط کلیدی بگرده.
فرقش با SIFT:
cv2.SIFT_create() → کارآگاه دقیق ولی کند
cv2.ORB_create() → کارآگاه سریع


خط ۴: keypoints, descriptors = orb.detectAndCompute(gray, None)
قسمت	توضیح
keypoints	لیست نقاط کلیدی (اسم دلخواه)
descriptors	توصیفگرها (اسم دلخواه)
orb	همون شیء ORB که ساختیم
.detectAndCompute(...)	تابعی که هم تشخیص میده هم توصیفگر می‌سازه
gray	تصویر خاکستری (ورودی)
None	ماسک (فعلاً None = همه‌ی تصویر رو بررسی کن)
کارش چیه؟
دو تا کار با هم:
detect = پیدا کردن نقاط کلیدی
compute = ساخت توصیفگر برای هر نقطه
چرا با هم؟
چون هر دو به هم وابسته‌ن و جدا کردنشون کند میشه.

خط ۵: print(f"تعداد نقاط کلیدی: {len(keypoints)}")
قسمت	توضیح
len(keypoints)	تعداد نقاط کلیدی


خط ۶: img_kp = cv2.drawKeypoints(img, keypoints, None, flags=...)
قسمت	توضیح
img_kp	تصویر خروجی با نقاط رسم شده
cv2.drawKeypoints(...)	تابع رسم نقاط
img	تصویر اصلی
keypoints	نقاطی که رسم میشن
None	تصویر خروجی (خودش می‌سازه)
flags=...	نحوه رسم
فقط یه چیز جدید داره:
flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS → نقاط رو با اطلاعات کامل رسم می‌کنه.


# 📌 کد کامل ORB روی ویدیو (وبکم)
import cv2

# ۱. ساخت ORB
orb = cv2.ORB_create()

# ۲. باز کردن وبکم
cap = cv2.VideoCapture(0)

while True:
    # ۳. خواندن یه فریم
    ret, frame = cap.read()
    if not ret:
        break
    
    # ۴. تبدیل به خاکستری
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ۵. پیدا کردن نقاط کلیدی
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # ۶. رسم نقاط روی فریم
    frame_kp = cv2.drawKeypoints(frame, keypoints, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # ۷. نمایش فریم
    cv2.imshow("ORB Video", frame_kp)
    
    # ۸. خروج با کلید q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ۹. آزاد کردن منابع
cap.release()
cv2.destroyAllWindows()

# 📌 خط به خط:
خط	کد	کارش
۱	orb = cv2.ORB_create()	     ساخت شیء ORB (قبل از حلقه)
۲	cap = cv2.VideoCapture(0)	         باز کردن وبکم
۳	ret, frame = cap.read()	            خواندن یه فریم
۴	if not ret: break	             اگه فریم نیومد، خارج شو
۵	gray = cv2.cvtColor(frame, ...)  	تبدیل به خاکستری
۶	keypoints, descriptors = orb.detectAndCompute(gray, None)	                                  پیدا کردن نقاط
۷	frame_kp = cv2.drawKeypoints(...)	رسم نقاط
۸	cv2.imshow("ORB Video", frame_kp)	نمایش
۹	if cv2.waitKey(1) & 0xFF == ord('q'): break	خروج با q
۱۰	cap.release()	                    آزاد کردن وبکم
۱۱	cv2.destroyAllWindows()           	بستن پنجره‌ها

# 📌 فرق گوشه و لبه:
گوشه                             	لبه
تغییر روشنایی در یه جهات	تغییر روشنایی در همه جهت
 	مثل خط صاف بین دو رنگ         مثل گوشه‌ی مربع

لبه = مرز بین دو ناحیه‌ی متفاوت.
مثال‌ها:

مرز دیوار و زمین  	روشنایی دیوار ≠ روشنایی زمین
مرز یه خودرو        	رنگ خودرو ≠ رنگ خیابون
مرز سایه و نور	         سایه تیره ≠ نور روشن

گوشه:
┌
│  ← تغییر در دو جهت

لبه:
───  ← تغییر در یه جهت

-------------------
# 📌 قسمت ۴: FAST
-------------------
📌  توضیح کلی:
FAST یه روش خیلی سریع برای پیدا کردن گوشه‌هاست.
اسمش مخففه:
F = Features (ویژگی‌ها)
A = from (از)
S = Accelerated (تسریع‌شده)
T = Segment (قطعه) Test
یعنی: یه روش سریع برای تشخیص گوشه‌ها.

# 📌  چرا FAST ساخته شد؟
مشکل SIFT و ORB:
نسبتاً کند هستن ❌
برای ویدیو با فریم بالا مناسبن نیستن
راه‌حل FAST:
بسیار سریع ✅
مناسب برای Real-time ✅
ساده و سبک ✅
# نکته
اگه چند تا پیکسل پیوسته روی دایره، روشن‌تر یا تیره‌تر از گوشه‌ست →  پیکسل مرکزی باشن  

# 📌  مزیت FAST:
خیلی سریع - ساده - ریل تایم -
# 📌 . عیب FAST:
بدون توصیفگره فقط گوشه پیدا میکنه
مقاوم به چرخش نیست اگر تصویر بچرخه مشکل  داره
حساس به نویزه نویز باعث اشتباهش میشه
# 📌 . کد FAST:
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(gray, None)
قسمت	توضیح
cv2.FastFeatureDetector_create()	ساخت شیء FAST
fast.detect(gray, None)	      فقط تشخیص (بدون توصیفگر)
keypoints	                    لیست نقاط کلیدی
نکته: FAST فقط detect داره، نه detectAndCompute (چون توصیفگر نمی‌سازه).
# ❓ سؤال:
 FAST برای چه کاربردهایی مناسبه؟
 وقتی سرعت خیلی مهمه، مثل ویدیو زنده با فریم بالا
 # 📌 فریم چیه؟ (خیلی ساده)
ویدیو = چند تا عکس پشت سر هم که سریع نشون داده میشن
هر عکس = یه فریم.
# 📌 فریم ریت (Frame Rate) چیه؟
معنی     فریم ریت	
۲۴ fps	سینمایی (فیلم)
۳۰ fps	ویدیو معمولی
۶۰ fps	ویدیو روون
۱۲۰ fps	ویدیو خیلی روون (بازی‌ها)
fps = Frames Per Second = فریم در ثانیه

فریم = یه عکس از ویدیو
فریم ریت = تعداد عکس در ثانیه

# کد کامل
import cv2

# ۱. خواندن تصویر
img = cv2.imread("E:/cp-vision/face_test.jpg")

# ۲. تبدیل به خاکستری
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ۳. ساخت FAST
fast = cv2.FastFeatureDetector_create()

# ۴. پیدا کردن نقاط کلیدی
keypoints = fast.detect(gray, None)

# ۵. چاپ تعداد
print(f"تعداد نقاط کلیدی: {len(keypoints)}")

# ۶. رسم نقاط
img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

# ۷. نمایش و ذخیره
cv2.imshow("FAST", img_kp)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("E:/cp-vision/fast.jpg", img_kp)
print("✅ تصویر ذخیره شد.")

۱. fast.detect() فقط تشخیص میده
نه توصیفگر می‌سازه (برخلاف SIFT و ORB که detectAndCompute دارن).
۲. color=(0, 255, 0) چیه؟
رنگ نقاط رسم شده. (0, 255, 0) = سبز (BGR).
۳. FAST نقطه‌ها رو ساده رسم می‌کنه
چون توصیفگر نداره (بدون اندازه و جهت).

----------------
# 📌 قسمت ۵: Blob Detection (آخرین قسمت ۰۴.۱)
----------------
📌 ۱. توضیح کلی:
Blob = لکه
Blob Detection = پیدا کردن لکه‌ها توی تصویر.
لکه چیه؟
یه ناحیه‌ی متمایز که روشنایی‌اش با اطرافش فرق داره.
مثلاً: یه دایره‌ی سفید روی پس‌زمینه‌ی سیاه.
# 📌 ۴. روش‌های Blob Detection:
روش	توضیح
SimpleBlobDetector	ساده و آماده در OpenCV
LoG	              Laplacian of Gaussian
DoG	              Difference of Gaussian
ما از SimpleBlobDetector استفاده می‌کنیم (آماده‌ست).
# 📌 ۵. کد Blob Detection:
detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(gray)
cv2.SimpleBlobDetector_create()	ساخت شیء Blob Detector
detector.detect(gray)	            پیدا کردن لکه‌ها
keypoints	                           لیست لکه‌ها   
# ❓ سؤال:
Blob Detection برای چه کاربردهایی مناسبه؟

تشخیص سلول	     توی تصاویر میکروسکوپی
تشخیص حباب                  	توی مایعات
تشخیص ستاره          	توی تصاویر نجومی
تشخیص لکه               	روی سطوح صنعتی
تشخیص غده               	توی تصاویر پزشکی
شمارش اشیا              	سکه، دکمه، مهره
عکس	         ✅
ویدیو	     ✅
وبکم زنده	✅
# 📌 کد کامل Blob Detection
# 📌 ۱. کد عکس:
import cv2
# ☝️ وارد کردن کتابخانه OpenCV

img = cv2.imread("E:/cp-vision/face_test.jpg")
# ☝️ خواندن تصویر

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ☝️ تبدیل به خاکستری

detector = cv2.SimpleBlobDetector_create()
# ☝️ ساخت شیء Blob Detector ← جدید

keypoints = detector.detect(gray)
# ☝️ پیدا کردن لکه‌ها ← جدید

print(f"تعداد لکه‌ها: {len(keypoints)}")
# ☝️ چاپ تعداد لکه‌ها

img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# ☝️ رسم لکه‌ها روی تصویر

cv2.imshow("Blob", img_kp)
# ☝️ نمایش تصویر

cv2.waitKey(0)
cv2.destroyAllWindows()
# ☝️ انتظار و بستن پنجره‌ها

cv2.imwrite("E:/cp-vision/blob.jpg", img_kp)
# ☝️ ذخیره تصویر

print("✅ تصویر ذخیره شد.")
# ☝️ پیام موفقیت

# 📌 توضیح چیزای جدید (عکس):

detector = cv2.SimpleBlobDetector_create()
ساخت یه شیء Blob Detector (مثل یه کارآگاه که دنبال لکه‌ها می‌گرده).

keypoints = detector.detect(gray)
پیدا کردن لکه‌ها توی تصویر خاکستری.
detector	شیء Blob Detector
.detect(gray)	تابع پیدا کردن لکه‌ها
gray	تصویر خاکستری
keypoints	لیست لکه‌های پیدا شده
# 📌 ۲. کد ویدیو:
import cv2
# ☝️ وارد کردن کتابخانه OpenCV

detector = cv2.SimpleBlobDetector_create()
# ☝️ ساخت شیء Blob Detector (بیرون حلقه)

cap = cv2.VideoCapture(0)
# ☝️ باز کردن وبکم

while True:
    ret, frame = cap.read()
    # ☝️ خواندن یه فریم
    
    if not ret:
        break
    # ☝️ اگه فریم نیومد، خارج شو
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ☝️ تبدیل فریم به خاکستری
    
    keypoints = detector.detect(gray)
    # ☝️ پیدا کردن لکه‌ها توی فریم
    
    frame_kp = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # ☝️ رسم لکه‌ها روی فریم
    
    cv2.imshow("Blob Video", frame_kp)
    # ☝️ نمایش فریم
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # ☝️ خروج با کلید q

cap.release()
cv2.destroyAllWindows()
# ☝️ آزاد کردن وبکم و بستن پنجره‌ها

# 📌 توضیح چیزای جدید (ویدیو):
detector = cv2.SimpleBlobDetector_create() (بیرون حلقه)
ساخت یه بار (نه توی حلقه) → سریع‌تره.

keypoints = detector.detect(gray) (داخل حلقه)
هر فریم، لکه‌ها رو پیدا می‌کنه.

if cv2.waitKey(1) & 0xFF == ord('q'): break
اگه کلید q زده شد، از حلقه خارج شو.
# 📌 ۳. تفاوت عکس و ویدیو:
عکس:
detect یبار
cv2.imread
یه بار نمایش
ویدیو:
توی حلقه، هر فریم detect
cap.read
توی حلقه نمایش
# 📊 خلاصه‌ی ۰۴.۱:

۱	Harris Corner	     cv2.cornerHarris()	
۲	SIFT	             cv2.SIFT_create()	
۳	ORB	                  cv2.ORB_create()	
۴	FAST	             cv2.FastFeatureDetector_create()	
۵	Blob Detection	     cv2.SimpleBlobDetector_create()	


✅ گوشه‌ها رو پیدا کنی (Harris, FAST)
✅ نقاط متمایز رو پیدا کنی (SIFT, ORB)
✅ لکه‌ها رو پیدا کنی (Blob Detection)
✅ روی عکس و ویدیو کار کنی

# 📌 مقایسه کلی الگوریتم‌های Feature Detection
# Harris Corner
سرعت: سریع
دقت: متوسط
مقاوم به چرخش: ✅ بله
مقاوم به مقیاس: ❌ خیر
توصیفگر: ❌ نداره
کاربرد: تشخیص گوشه، ویدیو ساده
# SIFT
سرعت: کند
دقت: بالا
مقاوم به چرخش: ✅ بله
مقاوم به مقیاس: ✅ بله
توصیفگر: ✅ داره (۱۲۸ عدد اعشاری)
کاربرد: تطابق دقیق، پانوراما، تشخیص شیء
# ORB
سرعت: متوسط
دقت: خوب
مقاوم به چرخش: ✅ بله
مقاوم به مقیاس: ⚠️ تا حدی
توصیفگر: ✅ داره (دودویی)
کاربرد: Real-time، موبایل، رباتیک
# FAST
سرعت: خیلی سریع
دقت: پایین
مقاوم به چرخش: ❌ خیر
مقاوم به مقیاس: ❌ خیر
توصیفگر: ❌ نداره
کاربرد: ویدیو با فریم بالا، رباتیک سریع
# Blob Detection
سرعت: متوسط
دقت: خوب
مقاوم به چرخش: ✅ بله
مقاوم به مقیاس: ✅ بله
توصیفگر: ✅ داره
کاربرد: تشخیص سلول، حباب، لکه، ستاره

📌 جمع‌بندی در یه خط:
ساده‌ترین: Harris
سریع‌ترین: FAST
دقیق‌ترین: SIFT
متعادل‌ترین: ORB
تشخیص لکه: Blob Detection

------------------پایان بخش 1 ------------------------------
----------------------------------
# ۰۴.۲ — Feature Matching (تطابق ویژگی‌ها)
----------------------------------
# نکته :
این بخش همیشه به دو تا عکس نیاز داره
# 📌 توضیح کلی:
Feature Matching = تطابق ویژگی‌ها بین دو تصویر.
داستان چیه؟
فرض کن دو تا عکس از یه ساختمون داری:
عکس اول: از زاویه‌ی چپ
عکس دوم: از زاویه‌ی راست
می‌خوای بفهمی کدوم نقاط این دو عکس، مربوط به یه چیز واحد هستن.
Feature Matching دقیقاً همین کار رو می‌کنه:
نقاط کلیدی عکس اول رو پیدا می‌کنه
نقاط کلیدی عکس دوم رو پیدا می‌کنه
نقاط مشترک رو تطابق میده
# 📌 چرا مهمه؟
چسباندن دو عکس به هم        پانوراما
پیدا کردن شیء توی تصویر     تشخیص شیء 
ساخت مدل ۳D     از دو عکسبازسازی ۳بعدی   
مقایسه دو تصویر         تشخیص جعل

# 📌 این زیرشاخه چند تا قسمت داره؟
۱	BFMatcher	   تطابق ساده ویژگی‌ها
۲	FLANN	       تطابق سریع
۳	Ratio Test	   فیلتر تطابق‌های اشتباه

------------------
# 📌 قسمت ۱: BFMatcher
------------------
# 📌 توضیح کلی:
BFMatcher = Brute-Force Matcher = تطابق‌گر زورکی
اسمش از کجا اومده؟
Brute = زور، خام
Force = نیرو
Matcher = تطابق‌دهنده
یعنی: یه روش ساده و خام که همه‌ی نقاط رو با همه‌ی نقاط مقایسه می‌کنه.

# 📌 چطور کار می‌کنه؟
فرض کن:
تصویر اول: ۱۰۰ تا نقطه کلیدی داره
تصویر دوم: ۱۰۰ تا نقطه کلیدی داره
BFMatcher میاد:
نقطه‌ی اول تصویر ۱ رو با همه‌ی ۱۰۰ نقطه‌ی تصویر ۲ مقایسه می‌کنه
نزدیک‌ترین نقطه رو انتخاب می‌کنه
همین کار رو برای همه‌ی ۱۰۰ نقطه انجام میده
نتیجه: ۱۰۰ تا تطابق.

# 📌 مزایا و معایب:
مزیت :
ساده و دقیق  / برای نقاط کم عالیه
عیب :
کند برای نقاط زیاد  / برای نقاط زیاد بهینه نیست

# 📌 کد BFMatcher:
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

cv2.BFMatcher()	     ساخت شیء BFMatcher
cv2.NORM_HAMMING	روش محاسبه فاصله (برای ORB)
crossCheck=True	    فقط تطابق‌های دوطرفه
bf.match(...)	    تطابق توصیفگرها

# 📌 نکته مهم:
cv2.NORM_HAMMING یا cv2.NORM_L2؟
روش               توصیفگر
SIFT (اعشاری)       cv2.NORM_L2
ORB (دودویی)      cv2.NORM_HAMMING
# ❓ سؤال:
 BFMatcher برای چه تصاویری مناسبه؟
 تصاویر با نقاط کم

# 📌 کد کامل BFMatcher
بریا اینکه دوتا عکس بازاویه مختلف پیدا کنیم 
https://github.com/opencv/opencv/tree/master/samples/data

# 📌 ۱. کد:
import cv2

# ۱. خواندن دو تصویر
img1 = cv2.imread("E:/cp-vision/left01.jpg")
img2 = cv2.imread("E:/cp-vision/left02.jpg")

# ۲. تبدیل به خاکستری
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ۳. ساخت ORB
orb = cv2.ORB_create()

# ۴. پیدا کردن نقاط و توصیفگر
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

print(f"نقاط تصویر ۱: {len(kp1)}")
print(f"نقاط تصویر ۲: {len(kp2)}")

# ۵. ساخت BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# ۶. تطابق توصیفگرها
matches = bf.match(des1, des2)

# ۷. مرتب‌سازی بر اساس فاصله
matches = sorted(matches, key=lambda x: x.distance)

# ۸. چاپ تعداد تطابق‌ها
print(f"تعداد تطابق‌ها: {len(matches)}")

# ۹. رسم تطابق‌ها (فقط ۵۰ تای اول)
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# ۱۰. نمایش و ذخیره
cv2.imshow("BFMatcher", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("E:/cp-vision/bfmatcher.jpg", img_matches)
print("✅ تصویر ذخیره شد.")


نقاط کلیدی تصویر اول رو پیدا کرده
نقاط کلیدی تصویر دوم رو پیدا کرده
نقاط مشترک رو بین دو تصویر تطابق داده
با خط سبز به هم وصل کرده
ORB — پیدا کردن نقاط
BFMatcher — تطابق نقاط
drawMatches — رسم تطابق‌ها

# 📌 مرحله به مرحله:
مرحله ۱: پیدا کردن نقاط (ORB)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)
مرحله ۲: ساخت تطابق‌گر (BFMatcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
→ یه کارآگاه که نقاط مشترک رو پیدا می‌کنه.
مرحله ۳: تطابق
matches = bf.match(des1, des2)
→ قلب ماجرا! تطابق بین دو تصویر.
مرحله ۴: رسم
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, ...)
→ نمایش تطابق‌ها.

----------------
# 📌 قسمت ۲: FLANN
----------------
# 📌 توضیح کلی:
FLANN = Fast Library for Approximate Nearest Neighbors
ترجمه: کتابخانه‌ی سریع برای پیدا کردن نزدیک‌ترین همسایه‌های تقریبی.

BFMatcher :
همه‌ی نقاط رو با همه مقایسه می‌کنه
دقیق ولی کند
برای نقاط کم
FLANN:
از ساختار درختی استفاده می‌کنه
تقریبی ولی سریع
برای نقاط زیاد
# 📌 چرا FLANN سریع‌تره؟
BFMatcher: مثل اینه که توی یه کتابخانه، همه‌ی کتاب‌ها رو ورق بزنی تا کتاب مورد نظر رو پیدا کنی.
FLANN: مثل اینه که از فهرست کتابخانه استفاده کنی → سریع‌تر.
# 📌 FLANN چطور کار می‌کنه؟
اول داده‌ها رو دسته‌بندی می‌کنه
بعد توی دسته‌ی مربوطه، جستجو می‌کنه
نتیجه: سریع‌تر از BFMatcher
# 📌 کد FLANN:
# برای SIFT (اعشاری)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
FLANN_INDEX_KDTREE	نوع ساختار درختی
trees=5	تعداد درخت‌ها
checks=50	تعداد بررسی‌ها
knnMatch(..., k=2)	برای هر نقطه، ۲ تا نزدیک‌ترین رو پیدا کن

# 📌 چرا k=2؟
چون برای Ratio Test (قسمت بعدی) نیاز به ۲ تا تطابق داریم:
بهترین تطابق
دومین تطابق
# ❓ سؤال:
 FLANN برای چه تصاویری مناسبه؟
 تصاویر با نقاط زیاد
 # 📌 کد کامل FLANN
# 📌 ۱. کد:
import cv2

# ۱. خواندن دو تصویر
img1 = cv2.imread("E:/cp-vision/left01.jpg")
img2 = cv2.imread("E:/cp-vision/left02.jpg")

# ۲. تبدیل به خاکستری
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ۳. ساخت SIFT
sift = cv2.SIFT_create()

# ۴. پیدا کردن نقاط و توصیفگر
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f"نقاط تصویر ۱: {len(kp1)}")
print(f"نقاط تصویر ۲: {len(kp2)}")

# ۵. تنظیمات FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# ۶. تطابق با knnMatch (k=2)
matches = flann.knnMatch(des1, des2, k=2)

# ۷. Ratio Test (فیلتر تطابق‌های اشتباه)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"تعداد تطابق‌های خوب: {len(good_matches)}")

# ۸. رسم تطابق‌ها
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# ۹. نمایش و ذخیره
cv2.imshow("FLANN", img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("E:/cp-vision/flann.jpg", img_matches)
print("✅ تصویر ذخیره شد.")




# 📌 ۲. خط به خط (فقط چیزای جدید):
# خط ۵: FLANN_INDEX_KDTREE = 1 
این یه کد عددی هست که به FLANN می‌گه از چه روشی برای جستجو استفاده کن.
OpenCV یه سری عدد ثابت تعریف کرده:
1	KD-Tree (روش درختی)
0	Linear (روش خطی، مثل BFMatcher)
KD-Tree	1
Linear	0
LSH	    6
# 📌 KD-Tree چیه؟ (ساده)
KD-Tree = یه ساختار درختی که داده‌ها رو دسته‌بندی می‌کنه.
مثال ساده:
فرض کن ۱۰۰۰ تا نقطه داری.
بدون KD-Tree: باید همه رو بررسی کنی → کند
با KD-Tree: داده‌ها توی دسته‌های مختلف تقسیم شدن → سریع

# خط ۶:
 index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
تنظیمات ساختار درختی رو تعیین می‌کنه.
index_params	       اسم متغیر (اسم دلخواه)
dict(...)	          ساخت یه دیکشنری (جدول کلید-مقدار)
algorithm=FLANN_INDEX_KDTREE	نوع الگوریتم (KD-Tree = 1)
trees=5	          تعداد درخت‌ها (بیشتر = دقیق‌تر ولی کندتر)
trees=5 یعنی چی؟
FLANN میاد ۵ تا درخت می‌سازه و داده‌ها رو توشون تقسیم می‌کنه.
هر چی بیشتر، جستجو دقیق‌تر ولی کندتر.

# خط ۷: 
search_params = dict(checks=50)
تنظیمات جستجو رو تعیین می‌کنه.
search_params	اسم متغیر
dict(...)	ساخت دیکشنری
checks=50	تعداد بررسی‌ها
checks=50 یعنی چی؟
توی هر جستجو، FLANN ۵۰ تا بررسی انجام میده.
هر چی بیشتر، دقیق‌تر ولی کندتر.

# خط ۸ :
 flann = cv2.FlannBasedMatcher(index_params, search_params)
ساخت شیء FLANN با تنظیمات مشخص.

flann	اسم متغیر
cv2.FlannBasedMatcher(...)	ساخت تطابق‌گر FLANN
index_params	تنظیمات درخت
search_params	تنظیمات جستجو
نتیجه: یه کارآگاه سریع آماده‌ست که نقاط مشترک رو پیدا کنه.

# خط ۹:
 matches = flann.knnMatch(des1, des2, k=2)
تطابق توصیفگرهای دو تصویر.

flann	شیء FLANN
.knnMatch(...)	تابع تطابق K-نزدیک‌ترین
des1	توصیفگرهای تصویر ۱
des2	توصیفگرهای تصویر ۲
k=2	برای هر نقطه، ۲ تا نزدیک‌ترین پیدا کن
matches	لیست تطابق‌ها

knnMatch چیه؟
= K-Nearest Neighbors Match
یعنی برای هر نقطه، k تا نزدیک‌ترین رو پیدا کن.
k=2 یعنی چی؟
برای هر نقطه، ۲ تا نزدیک‌ترین نقطه توی تصویر دوم رو پیدا کن.

# خط ۱۰: 
Ratio Test (فیلتر تطابق‌های اشتباه)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
تطابق‌های اشتباه رو حذف می‌کنه.

good_matches = []	لیست خالی برای تطابق‌های خوب
for m, n in matches	حلقه روی هر تطابق (m=بهترین، n=دومین)
m.distance	فاصله بهترین تطابق
n.distance	فاصله دومین تطابق
0.75	آستانه (معمولاً ۰.۷ تا ۰.۸)
good_matches.append(m)	اگه شرط درست بود، اضافه کن

چرا Ratio Test؟
فرض کن یه نقطه توی تصویر ۱ داری.
توی تصویر ۲:
بهترین تطابق: فاصله = ۱۰
دومین تطابق: فاصله = ۱۲
آیا بهترین تطابق واقعاً درسته؟
نه! چون دومین تطابق خیلی نزدیکه → احتمالاً اشتباه داره تطابق میده.
با Ratio Test:
10 < 0.75 * 12 → 10 < 9 ❌ → رد میشه
مثال درست:
بهترین تطابق: فاصله = ۵
دومین تطابق: فاصله = ۲۰
5 < 0.75 * 20 → 5 < 15 ✅ → قبول میشه

# خط ۱۱: 
img_matches = cv2.drawMatches(...)
رسم تطابق‌های خوب بین دو تصویر.

img1, kp1	تصویر اول و نقاطش
img2, kp2	تصویر دوم و نقاطش
good_matches	فقط تطابق‌های خوب
None	تصویر خروجی
flags=...	فقط نقاط تطابق‌دار

----------------
# 📌 Ratio Test 
----------------

این بخش رو خالی استفاده نمیشه فقط در بخش 2 استفاده میشه
حط 10 استفاده شد فیلتر کردن بود
# 📌 Ratio Test کجا استفاده میشه؟
BFMatcher	❌ لازم نیست
FLANN	    ✅ لازمه

BFMatcher:
از crossCheck=True استفاده می‌کنه
خودش تطابق‌های اشتباه رو حذف می‌کنه
پس نیازی به Ratio Test نیست
FLANN:
از knnMatch(des1, des2, k=2) استفاده می‌کنه
k=2 یعنی ۲ تا تطابق برمی‌گردونه
خودش تطابق‌های اشتباه رو حذف نمی‌کنه
پس باید Ratio Test بذاری

# Ratio Test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
# 📌 توضیح:

۱	good_matches = []	لیست خالی برای تطابق‌های خوب
۲	for m, n in matches:	حلقه روی هر جفت تطابق
۳	if m.distance < 0.75 * n.distance:	شرط Ratio Test
۴	good_matches.append(m)	اضافه کردن تطابق خوب

# 📌 معنی متغیرها:
m	بهترین تطابق
n	دومین تطابق
m.distance	فاصله بهترین تطابق
n.distance	فاصله دومین تطابق
0.75	آستانه (معمولاً ۰.۷ تا ۰.۸)

------------------پایان بخش 2 ------------------------------

------------------------------
# ۰۴.۳ — Geometric Transformations (تبدیلات هندسی)
------------------------------
# 📌 توضیح کلی:
Geometric Transformations = تغییرات هندسی روی تصویر.
یعنی جابه‌جا کردن، چرخوندن، بزرگ/کوچیک کردن، تغییر شکل دادن تصویر.
# 📌 داستان چیه؟
فرض کن یه عکس از یه ساختمون داری که کج گرفته شده.
می‌خوای:
ساختمون رو صاف کنی
زاویه‌اش رو درست کنی
مثل این که از روبه‌رو گرفتی
# 📌 چه کارهایی میشه کرد؟
چرخش        	چرخوندن عکس ۹۰ درجه
جابه‌جایی	حرکت دادن عکس به چپ/راست
بزرگ/کوچیک	                  زوم کردن
کج کردن          	صاف کردن عکس کج
آینه‌ای	              برعکس کردن عکس
Geometric Transformations دقیقاً همین کار رو می‌کنه.
# 📌 چرا مهمه؟
صاف کردن اسناد      	عکس کج از سند رو صاف می‌کنه
پانوراما             	عکس‌ها رو به هم می‌چسبونه
واقعیت افزوده	اشیای مجازی رو توی عکس واقعی می‌ذاره
تشخیص پلاک                  	پلاک کج رو صاف می‌کنه
# 📌 ۰۴.۳ شامل چیه؟
۱	Homography	                 تبدیل صفحه به صفحه
۲	Perspective Transform	     تغییر پرسپکتیو
۳	Affine Transform	         چرخش، مقیاس، انتقال
۴	Image Stitching	           چسباندن تصاویر (پانوراما)

---------------
# 📌 قسمت ۱: Homography
---------------
# 📌 توضیح کلی:
Homography = هوموگرافی = تبدیل صفحه به صفحه
یعنی یه صفحه‌ی کج رو به یه صفحه‌ی صاف تبدیل می‌کنه.
# 📌 داستان چیه؟
فرض کن از یه کتاب عکس گرفتی، ولی کج گرفته شده:
کتاب کج:

   ╱─────────╲
  ╱           ╲
 ╱             ╲
╱───────────────╲

می‌خوای کتاب رو صاف کنی:

کتاب صاف:
┌─────────────┐
│             │
│             │
│             │
└─────────────┘
Homography دقیقاً همین کار رو می‌کنه.

# 📌 Homography چطور کار می‌کنه؟
میاد ۴ نقطه‌ی کلیدی رو توی تصویر کج پیدا می‌کنه و به ۴ نقطه‌ی صاف تبدیل می‌کنه.

# 📌 نقاط کج (تصویر اصلی):
(x1,y1) ───────── (x2,y2)
   ╱                  ╲
  ╱                    ╲
 ╱                      ╲
(x4,y4) ───────── (x3,y3)
چهار گوشه: x1y1، x2y2، x3y3، x4y4

# 📌 نقاط صاف (تصویر خروجی):
(x1',y1') ───────── (x2',y2')
   │                     │
   │                     │
   │                     │
(x4',y4') ───────── (x3',y3')
چهار گوشه: x1'y1'، x2'y2'، x3'y3'، x4'y4'

نقطه‌ی صاف → نقطه کج
(x1,y1)	→	(x1',y1')
(x2,y2)	→	(x2',y2')
(x3,y3)	→	(x3',y3')
(x4,y4)	→	(x4',y4')

# 📌 کاربردهای Homography: 
کاربرد                                 توضیح
صاف کردن اسناد      	عکس کج از سند رو صاف می‌کنه
تشخیص پلاک                     	پلاک کج رو صاف می‌کنه
پانوراما                	عکس‌ها رو به هم می‌چسبونه
واقعیت افزوده	اشیای مجازی رو توی عکس واقعی می‌ذاره
تشخیص شیء   	          شیء رو توی تصویر پیدا می‌کنه

# 📌 کد Homography:
H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
warped = cv2.warpPerspective(img, H, (width, height))

	توضیح                          قسمت
cv2.findHomography(...)     	پیدا کردن ماتریس Homography
src_points	                    نقاط مبدأ (چهار گوشه‌ی کج)
dst_points	                    نقاط مقصد (چهار گوشه‌ی صاف)
cv2.RANSAC	                              روش مقاوم به نویز
5.0	                                           آستانه RANSAC
H	                                      ماتریس Homography
cv2.warpPerspective(...)                    	اعمال تبدیل

# Homography حداقل به ۴ نقطه نیاز داره
چون تصویر دو بعدی هست (ارتفاع و عرض).
برای صاف کردن دقیق یه صفحه، حداقل ۴ نقطه لازمه:
گوشه بالا-چپ
گوشه بالا-راست
گوشه پایین-راست
گوشه پایین-چپ

گوشه‌ی A → باید بره به A'
گوشه‌ی B → باید بره به B'
گوشه‌ی C → باید بره به C'
گوشه‌ی D → باید بره به D'
اگه فقط ۲ تا بدی:
Homography نمی‌دونه بقیه‌ی گوشه‌ها کجان.
اگه ۴ تا بدی:
Homography دقیقاً می‌دونه هر گوشه کجا بره → تصویر کامل صاف میشه.
# 📌 یه مثال دیگه:
فرض کن یه میز داری که ۴ تا پا داره.
اگه فقط ۲ تا پا رو تنظیم کنی، میز کج می‌مونه.
اگه ۴ تا پا رو تنظیم کنی، میز کامل صاف میشه.
# ✅ خلاصه:
۴ نقطه = ۴ گوشه‌ی تصویر
بدون ۴ گوشه، تصویر کامل صاف نمیشه.
# ❓ سؤال:
Homography برای چه کاربردهایی مناسبه؟
(راهنمایی: صاف کردن اسناد، تشخیص پلاک، پانوراما)

# 📌 کد Homography

دو مدل باید کد بنویسیم 

با روش دستی: 
1.کدی که باید مختصات 4 گوشه هدف رو با حدس بنویسیم مدام که صاف بشه
نقطه ۱: (100, 50)   ← حدس زدی
نقطه ۲: (400, 70)   ← حدس زدی
نتیجه: تصویر ناقص صاف میشه ❌
مختصات رو خودت حدس می‌زنی.
مشکل:
اگه مختصات اشتباه باشن → تصویر درست صاف نمیشه
باید چند بار امتحان کنی تا درست بشه

# کدش
src_points = np.float32([
    [100, 50],
    [400, 70],
    [420, 500],
    [80, 480]
])


با روش ماوس:
2. با کلیک روی 4گوشه مختصات صحیح رو پیدا کنیم توی بخش کد 1 بریم بنویسیم
نقطه ۱: (120, 45)   ← دقیق کلیک کردی
نقطه ۲: (398, 68)   ← دقیق کلیک کردی
نتیجه: تصویر کامل صاف میشه ✅
با کلیک روی گوشه‌ها، مختصات دقیق ثبت میشه.
مزیت:
مختصات دقیق روی گوشه‌ها ثبت میشه
تصویر کامل صاف میشه
نیازی به حدس زدن نیست

# کدش
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])


# ⚠️ نکته مهم:
نقاط باید به این ترتیب باشن:
بالا-چپ
بالا-راست
پایین-راست
پایین-چپ


# 📌 2 با ماوس. کد:
import cv2

img = cv2.imread("E:/cp-vision/left02.jpg")

points = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"{len(points)}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image", img)
        print(f"نقطه {len(points)}: ({x}, {y})")

cv2.imshow("Image", img)
cv2.setMouseCallback("Image", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n✅ نقاط کلیک شده:", points)



# مختصات بدست امده به ترتیب
نقطه ۱: (195, 43)   → بالا-چپ
نقطه ۲: (604, 121)  → بالا-راست
نقطه ۳: (466, 417)  → پایین-راست
نقطه ۴: (221, 364)  → پایین-چپ
# توضیحات

خط ۳: points = []
یه لیست خالی می‌سازه که مختصات کلیک‌ها توش ذخیره بشه

خط ۴: def click_event(event, x, y, flags, params):
تعریف تابع برای مدیریت کلیک ماوس
توضیح  پارامتر
event	نوع رویداد (کلیک، حرکت، ...)
x, y	مختصات کلیک
flags	پرچم‌های اضافی
params	پارامترهای اضافی

خط ۵: if event == cv2.EVENT_LBUTTONDOWN:
اگه کلیک چپ ماوس زده شد

خط ۶: points.append([x, y])
مختصات کلیک رو به لیست اضافه کن

خط ۷: cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
یه دایره قرمز روی نقطه‌ی کلیک شده بکش
img	تصویر
(x, y)	مرکز دایره
5	شعاع
(0, 0, 255)	رنگ قرمز
-1	پر کردن دایره

خط ۸: cv2.putText(img, f"{len(points)}", (x, y), ...)
شماره‌ی نقطه رو کنارش بنویس

خط ۹: cv2.imshow("Image", img)
تصویر رو دوباره نشون بده (با نقطه‌ی جدید)

خط ۱۰: print(f"نقطه {len(points)}: ({x}, {y})")
مختصات رو توی ترمینال چاپ کن

خط ۱۱: cv2.imshow("Image", img)
نمایش تصویر

خط ۱۲: cv2.setMouseCallback("Image", click_event)
اتصال تابع کلیک به پنجره‌ی تصویر

یعنی: هر کلیکی روی پنجره، تابع click_event رو صدا بزنه.

خط ۱۳: cv2.waitKey(0)
انتظار برای کلید (تا پنجره بسته نشه)

خط ۱۴: cv2.destroyAllWindows()
بستن پنجره‌ها

خط ۱۵: print("\n✅ نقاط کلیک شده:", points)
چاپ همه‌ی نقاط کلیک شده

# 📌 خلاصه:
کار      خط
۱-۲	   خوندن تصویر
۳	   لیست خالی
۴-۱۰	تابع کلیک
۱۱-۱۲	اتصال تابع به پنجره
۱۳-۱۴	نمایش و انتظار
۱۵	     چاپ نقاط


# 📌 حالا کد Homography بدون ماوس:
import cv2
import numpy as np

# ۱. خواندن تصویر
img = cv2.imread("E:/cp-vision/left02.jpg")

# ۲. نقاط مبدأ (۴ گوشه‌ی تخته شطرنج)
src_points = np.float32([
    [195, 43],     # بالا-چپ
    [604, 121],    # بالا-راست
    [466, 417],    # پایین-راست
    [221, 364]     # پایین-چپ
])

# ۳. نقاط مقصد (۴ گوشه‌ی صاف)
dst_points = np.float32([
    [0, 0],        # بالا-چپ
    [500, 0],      # بالا-راست
    [500, 500],    # پایین-راست
    [0, 500]       # پایین-چپ
])

# ۴. پیدا کردن ماتریس Homography
H, status = cv2.findHomography(src_points, dst_points)

# ۵. اعمال Homography
warped = cv2.warpPerspective(img, H, (500, 500))

# ۶. نمایش
cv2.imshow("Original", img)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ۷. ذخیره
cv2.imwrite("E:/cp-vision/warped.jpg", warped)
print("✅ تصویر ذخیره شد.")

حروجی:
هم عکس کج هم صاف رو نشان میده

# توضیح 📌

خط ۱: import cv2
وارد کردن کتابخانه OpenCV

خط ۲: import numpy as np
وارد کردن کتابخانه NumPy (برای کار با آرایه‌ها)

خط ۳: img = cv2.imread("E:/cp-vision/left02.jpg")
خوندن تصویر

خط ۴-۹: src_points = np.float32([...])
نقاط مبدأ = ۴ گوشه‌ی چیزی که می‌خوای صاف کنی.
نقاط مبدأ = ۴ گوشه‌ی چیزی که می‌خوای صاف کنی.
src_points = np.float32([
    [195, 43],     # بالا-چپ
    [604, 121],    # بالا-راست
    [466, 417],    # پایین-راست
    [221, 364]     # پایین-چپ
])

src_points	اسم متغیر
np.float32([...])	تبدیل به آرایه‌ی اعشاری
[195, 43]	مختصات گوشه‌ی بالا-چپ
چرا float32؟
چون findHomography فقط با اعشاری کار می‌کنه.

خط ۱۰-۱۵: dst_points = np.float32([...])
نقاط مقصد = ۴ گوشه‌ی صاف که می‌خوایم تصویر به اونجا بره.
dst_points = np.float32([
    [0, 0],        # بالا-چپ
    [500, 0],      # بالا-راست
    [500, 500],    # پایین-راست
    [0, 500]       # پایین-چپ
])
اینا یه مربع ۵۰۰×۵۰۰ می‌سازن.


خط ۱۶: H, status = cv2.findHomography(src_points, dst_points)
قلب ماجرا!
H	ماتریس Homography (۳×۳)
status	وضعیت نقاط (درست/غلط)
کارش چیه؟
ماتریس Homography رو حساب می‌کنه که می‌گه:
هر پیکسل از src_points کجا بره توی dst_points

خط ۱۷: warped = cv2.warpPerspective(img, H, (500, 500))
اعمال تبدیل روی تصویر.

قسمت	توضیح
warped	تصویر خروجی
cv2.warpPerspective(...)	تابع اعمال Homography
img	تصویر اصلی
H	ماتریس Homography
(500, 500)	ابعاد تصویر خروجی

خط ۱۸-۲۰: نمایش
cv2.imshow("Original", img)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
نمایش تصویر اصلی و تصویر صاف شده.

خط ۲۱: cv2.imwrite("E:/cp-vision/warped.jpg", warped)
ذخیره تصویر خروجی

# 📌 خلاصه:
کار         خط
۱-۲	وارد کردن کتابخانه‌ها
۳	خوندن تصویر
۴-۹	نقاط مبدأ (کج)
۱۰-۱۵	نقاط مقصد (صاف)
۱۶	محاسبه ماتریس Homography
۱۷	اعمال تبدیل
۱۸-۲۰	نمایش
۲۱	ذخیره

-----------------
# 📌 قسمت ۲: Perspective Transform
-----------------
# 📌 فرقش با Homography چیه؟
Homography و Perspective Transform تقریباً یکی هستن!

ویژگی	  Homography	             Perspective Transform
تبدیل صفحه به صفحه         تبدیل صفحه به صفحه   کارش
نقطه4                    نقطه4       نقاط لازم
دستور	cv2.findHomography	    cv2.getPerspectiveTransform
اعمال	cv2.warpPerspective	    cv2.warpPerspective

تفاوت اصلی:
Homography: خودش ماتریس رو پیدا می‌کنه (با RANSAC)
Perspective Transform: ماتریس رو از ۴ نقطه حساب می‌کنه (دقیق)

# 📌 کد Perspective Transform:
M = cv2.getPerspectiveTransform(src_points, dst_points)
warped = cv2.warpPerspective(img, M, (width, height))

cv2.getPerspectiveTransform(...)	محاسبه ماتریس از ۴ نقطه
src_points	نقاط مبدأ
dst_points	نقاط مقصد
M	ماتریس تبدیل (۳×۳)
cv2.warpPerspective(...)	اعمال تبدیل
# 📌 تفاوت findHomography و getPerspectiveTransform:
findHomography	                getPerspectiveTransform
cv2.findHomography(src, dst)	cv2.getPerspectiveTransform(src, dst)
با RANSAC                        	بدون RANSAC
برای نقاط دقیق                  برای نقاط تقریبی
خروجی: H, status	                 خروجی: M

# RANSAC چیه
📌 RANSAC چیه؟ (خیلی ساده)
🎯 RANSAC = یه فیلتر هوشمند
اسمش مخففه:
RA = RANdom (تصادفی)
SA = SAmple (نمونه)
C = Consensus (توافق)
یعنی: یه روش که نقاط اشتباه رو حذف می‌کنه.

# 📌 کد Perspective Transform
import cv2
import numpy as np

# ۱. خواندن تصویر
img = cv2.imread("E:/cp-vision/left02.jpg")

# ۲. نقاط مبدأ (۴ گوشه‌ی تخته شطرنج)
src_points = np.float32([
    [195, 43],     # بالا-چپ
    [604, 121],    # بالا-راست
    [466, 417],    # پایین-راست
    [221, 364]     # پایین-چپ
])

# ۳. نقاط مقصد (۴ گوشه‌ی صاف)
dst_points = np.float32([
    [0, 0],
    [500, 0],
    [500, 500],
    [0, 500]
])

# ۴. پیدا کردن ماتریس Perspective
M = cv2.getPerspectiveTransform(src_points, dst_points)

# ۵. اعمال تبدیل
warped = cv2.warpPerspective(img, M, (500, 500))

# ۶. نمایش
cv2.imshow("Original", img)
cv2.imshow("Warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ۷. ذخیره
cv2.imwrite("E:/cp-vision/warped_perspective.jpg", warped)
print("✅ تصویر ذخیره شد.")

این پیشرفته تر از قبلیه
# 📌 توضیح خط به خط:
خط ۱-۲: import cv2, numpy as np
وارد کردن کتابخانه‌ها

خط ۳: img = cv2.imread(...)
خوندن تصویر

خط ۴-۹: src_points = np.float32([...])
نقاط مبدأ = ۴ گوشه‌ی کج توی تصویر

خط ۱۰-۱۵: dst_points = np.float32([...])
نقاط مقصد = ۴ گوشه‌ی صاف

خط ۱۶: M = cv2.getPerspectiveTransform(src_points, dst_points)
قلب ماجرا! ماتریس تبدیل رو از ۴ نقطه حساب می‌کنه.
M	ماتریس تبدیل (۳×۳)
تفاوت با findHomography:
findHomography → H, status (با RANSAC)
getPerspectiveTransform → M (بدون RANSAC)

خط ۱۷: warped = cv2.warpPerspective(img, M, (500, 500))
اعمال تبدیل روی تصویر.
img	تصویر اصلی
M	ماتریس تبدیل
(500, 500)	ابعاد خروجی


---------------
# 📌 قسمت ۳: Affine Transform
---------------
# 📌 توضیح کلی:
Affine Transform = تبدیل آفین
یعنی چرخش، مقیاس، انتقال، کج کردن تصویر — همه با هم.
# 📌 فرقش با Homography و Perspective چیه؟


ویژگی	            Affine	              Perspective / Homography
 نقطه ۴                            نقطه۳          نقاط لازم
 ممکنه موازی بمونن یا نباشن            موازی می‌مونن        موازی‌ها    	پیچیده‌تر                             ساده تر        پیچیدگی
 صاف کردن کج                       چرخش، مقیاس         کاربرد
# 📌 Affine چیکار می‌کنه؟
مثال                کار 
چرخوندن تصویر ۳۰ درجه              چرخش
بزرگ/کوچیک کردن              مقیاس
جابه‌جا کردن            انتقال      
کج کردن (Shear)	  کج کردن افقی/عمودی
# 📌 چرا فقط ۳ نقطه؟
چون Affine ۳ تا کار انجام میده:
چرخش
مقیاس
انتقال
برای تعریف این ۳ تا، حداقل ۳ نقطه لازمه.
# 📌 مثال ساده:
فرض کن یه مثلث داری:

     A
    / \
   /   \
  B─────C
  با Affine می‌تونی:

بچرخونی
بزرگ/کوچیک کنی
جابه‌جا کنی
ولی نمی‌تونی موازی‌ها رو به هم بریزی.
# 📌 کد Affine Transform:
M = cv2.getAffineTransform(src_points, dst_points)
warped = cv2.warpAffine(img, M, (width, height))

cv2.getAffineTransform(...)	محاسبه ماتریس از ۳ نقطه
src_points	۳ نقطه‌ی مبدأ
dst_points	۳ نقطه‌ی مقصد
M	ماتریس تبدیل   (۲×۳)
cv2.warpAffine(...)	اعمال تبدیل
 
 # 📌 کد کامل Affine Transform
📌 هدف کد Affine Transform چی بود؟
هدفش صاف کردن نبود!
هدفش تغییر شکل تصویر بود:
چرخش
مقیاس
جابه‌جایی
کج کردن

import cv2
import numpy as np

# ۱. خواندن تصویر
img = cv2.imread("E:/cp-vision/left02.jpg")

# ۲. نقاط مبدأ (۳ گوشه)
src_points = np.float32([
    [195, 43],     # بالا-چپ
    [604, 121],    # بالا-راست
    [221, 364]     # پایین-چپ
])

# ۳. نقاط مقصد (۳ گوشه‌ی جدید)
dst_points = np.float32([
    [0, 0],
    [500, 0],
    [0, 500]
])

# ۴. پیدا کردن ماتریس Affine
M = cv2.getAffineTransform(src_points, dst_points)

# ۵. اعمال تبدیل
warped = cv2.warpAffine(img, M, (500, 500))

# ۶. نمایش
cv2.imshow("Original", img)
cv2.imshow("Affine", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ۷. ذخیره
cv2.imwrite("E:/cp-vision/affine.jpg", warped)
print("✅ تصویر ذخیره شد.")

# 📌 توضیح خط به خط:
خط ۲: src_points = np.float32([...])
۳ نقطه‌ی مبدأ توی تصویر اصلی.

خط ۳: dst_points = np.float32([...])
۳ نقطه‌ی مقصد که می‌خوایم تصویر به اونجا بره.

خط ۴: M = cv2.getAffineTransform(src_points, dst_points)
قلب ماجرا! ماتریس Affine رو از ۳ نقطه حساب می‌کنه.

خط ۵: warped = cv2.warpAffine(img, M, (500, 500))
اعمال تبدیل روی تصویر.
img	تصویر اصلی
M	ماتریس تبدیل
(500, 500)	ابعاد خروجی

-----------------
# 📌 قسمت ۴: Image Stitching
-----------------
# 📌 توضیح کلی:
Image Stitching = چسباندن تصاویر به هم.

یعنی دو یا چند تا عکس رو بگیریم و به هم بچسبونیم تا یه عکس بزرگ‌تر بسازیم.
 # 📌 مثال ساده:
فرض کن از یه منظره‌ی پانوراما ۳ تا عکس گرفتی:
عکس ۱: [چپ منظره]
عکس ۲: [وسط منظره]
عکس ۳: [راست منظره]
با Image Stitching، این ۳ تا رو به هم می‌چسبونی و یه پانوراما می‌سازی:
[چپ + وسط + راست] = یه عکس بزرگ
# 📌 Image Stitching چطور کار می‌کنه؟
۱	پیدا کردن نقاط مشترک بین دو عکس (Feature Matching)
۲	محاسبه Homography بین دو عکس
۳	اعمال Homography روی عکس دوم
۴	چسباندن دو عکس به هم
# 📌 چرا Homography لازمه؟
چون دو عکس از زاویه‌های مختلف گرفته شدن.
Homography میاد عکس دوم رو با عکس اول هم‌راستا می‌کنه
# 📌 کد Image Stitching:
stitcher = cv2.Stitcher_create()
status, result = stitcher.stitch([img1, img2])
توضیح:
cv2.Stitcher_create()	ساخت شیء Stitcher
stitcher.stitch([img1, img2])	چسباندن دو عکس
status	وضعیت (OK یا Error)
result	تصویر نهایی
# 📌 نکته مهم:
OpenCV یه کلاس آماده برای Stitching داره: cv2.Stitcher
یعنی نیازی نیست خودت Feature Matching و Homography رو انجام بدی.
فقط عکس‌ها رو میدی، خودش همه کار می‌کنه.
# ❓ سؤال:
برای چه کاربردهایی مناسبه؟
پانوراما، عکس‌های هوایی، نقشه‌برداری

# نکته
برای اینکه عکس هایی رو بخواهید تست کنید باید همپوشانی باشند و از یک صحنه اینو سرچ کنید در گوگل
OpenCV panorama stitching sample images
وقتی اینو سرچ میکنی طبیعتا سه عکس یا بیشتر میاره که تیکه های عکس بهم چسبیده یک عکس کامل رو تشکیل داده
# 📌 کد کامل Image Stitching

import cv2

img1 = cv2.imread("E:/cp-vision/car1.png")
img2 = cv2.imread("E:/cp-vision/car2.png")

# ساخت Stitcher با حالت پانوراما
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
#                                    ↑
#                            حالت پانوراما

status, result = stitcher.stitch([img1, img2])

if status == cv2.Stitcher_OK:
    print("✅ موفق!")
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("E:/cp-vision/stitched.jpg", result)
else:
    print(f"❌ خطا: {status}")

# توضیحات

خط ۲: img1 = cv2.imread("E:/cp-vision/car1.png")
خواندن عکس اول از مسیر مشخص

قسمت	توضیح
img1	اسم متغیر
cv2.imread(...)	تابع خواندن تصویر
"E:/cp-vision/car1.png"	مسیر عکس

خط ۳: img2 = cv2.imread("E:/cp-vision/car2.png")
خواندن عکس دوم

خط ۴: stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
ساخت چسباننده با حالت پانوراما

قسمت	توضیح
stitcher	اسم متغیر
cv2.Stitcher_create(...)	تابع ساخت Stitcher
cv2.Stitcher_PANORAMA	حالت پانوراما (بهترین حالت)
چرا پانوراما؟
چون برای چسباندن عکس‌های منظره طراحی شده.


خط ۵: status, result = stitcher.stitch([img1, img2])
قلب ماجرا! چسباندن دو عکس

قسمت	توضیح
status	وضعیت (موفق/ناموفق)
result	تصویر نهایی
stitcher.stitch(...)	تابع چسباندن
[img1, img2]	لیست عکس‌ها
خط ۶: if status == cv2.Stitcher_OK:
چک کردن وضعیت

اگه status برابر cv2.Stitcher_OK بود → موفق ✅

خط ۷: print("✅ موفق!")
چاپ پیام موفقیت

خط ۸: cv2.imshow("Result", result)
نمایش تصویر نهایی

قسمت	توضیح
"Result"	عنوان پنجره
result	تصویر نهایی
خط ۹: cv2.waitKey(0)
انتظار برای کلید (تا پنجره بسته نشه)

خط ۱۰: cv2.destroyAllWindows()
بستن همه‌ی پنجره‌ها

خط ۱۱: cv2.imwrite("E:/cp-vision/stitched.jpg", result)
ذخیره تصویر نهایی

قسمت	توضیح
"E:/cp-vision/stitched.jpg"	مسیر ذخیره
result	تصویری که ذخیره میشه
خط ۱۲: else:
اگه وضعیت موفق نبود

خط ۱۳: print(f"❌ خطا: {status}")
چاپ کد خطا
# 📌 خلاصه:
کار      خط 
۱	وارد کردن OpenCV
۲-۳	خواندن دو عکس
۴	ساخت Stitcher
۵	چسباندن
۶-۷	چک وضعیت و پیام
۸-۱۱	نمایش و ذخیره
۱۲-۱۳	چاپ خطا
# 📌 کد خطاها:
cv2.Stitcher_OK                       	موفق ✅
cv2.Stitcher_ERR_NEED_MORE_IMGS	       عکس کافی نیست
cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL	Homography محاسبه نشد
cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL	پارامترهای دوربین تنظیم نشد
📌 خطای کد ۱ چیه؟
کد خطای ۱ = cv2.Stitcher_ERR_NEED_MORE_IMGS
یعنی: عکس کافی نیست یا همپوشانی کافی ندارن.
# 📌 نکته مهم:
عکس‌ها باید:
از یه صحنه باشن
همپوشانی داشته باشن (نقاط مشترک)
نور یکنواخت داشته باشن
اگه عکس‌ها همپوشانی نداشته باشن → خطا میده.

------------------پایان بخش 3 ------------------------------

------------------------------
# 📌 بخش ۰۴.۴: Camera Calibration (کالیبراسیون دوربین)
------------------------------
Camera Calibration = کالیبراسیون دوربین = تنظیم دقیق دوربین

یعنی پیدا کردن پارامترهای دقیق دوربین تا بفهمیم چطور تصویر می‌گیره

📌 داستان چیه؟
فرض کن با دو تا دوربین مختلف از یه شیء عکس می‌گیری:
دوربین ۱: تصویر صاف و دقیق میده
دوربین ۲: تصویر کمی کج و اعوجاج داره (مثل چشم ماهی)
سؤال: چطور بفهمیم کدوم تصویر واقعی‌تره؟
جواب: با کالیبراسیون دوربین!
# 📌 اعوجاج (Distortion) چیه؟
اعوجاج = کج شدن تصویر به خاطر لنز دوربین.
دو نوع اعوجاج:
۱. Radial Distortion (شعاعی)
خطوط صاف به سمت بیرون یا داخل خم میشن.
واقعیت:        اعوجاج:

┌─────┐        ╭─────╮
│     │   →    │     │
│     │        │     │
└─────┘        ╰─────╯
خطوط صاف → خمیده میشن.
۲. Tangential Distortion (مماسی)
تصویر کمی می‌چرخه یا کج میشه.
واقعیت:         اعوجاج:

┌─────┐        ╱─────╲
│     │   →   ╱       ╲
│     │       ╲       ╱
└─────┘        ╲─────╱
✅ خلاصه:
اعوجاج = کج شدن تصویر به خاطر لنز
دو نوع: شعاعی (خم شدن) و مماسی (چرخش)
# 📌 چرا کالیبراسیون مهمه؟
توضیح                    کاربرد
فاصله‌ی واقعی اشیا    اندازه‌گیری دقیق
تصحیح لنز                  تشخیص 
ساخت مدل ۳D           بازسازی ۳بعدی  
موقعیت دقیق اشیا       بینایی رباتیک
فاصله‌ی دقیق موانع              خودران

این بخش  4 بخش داره 
Intrinsic Parameters	تئوری (فقط مفهوم)
Extrinsic Parameters	تئوری (فقط مفهوم)
Distortion Correction	عملی (کد داره)
Chessboard Calibration	عملی (کد داره)
----------------------
# 📌 زیرشاخه ۱: Intrinsic Parameters
----------------------
Intrinsic Parameters = پارامترهای داخلی دوربین
یعنی مشخصات داخلی دوربین که ثابت هستن و به موقعیت دوربین ربطی ندارن.
📌 این پارامترها چیا هستن؟
Focal Length (f)	فاصله کانونی لنز
Optical Center (cx, cy)	مرکز تصویر
Pixel Size	اندازه هر پیکسل
Skew	کجی پیکسل‌ها
# 📌 ماتریس دوربین (Camera Matrix):
این پارامترها توی یه ماتریس ۳×۳ ذخیره میشن:
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]
fx, fy	فاصله کانونی (به پیکسل)
cx, cy	مرکز تصویر (به پیکسل)
# 📌 مثال ساده:
فرض کن دوربینت:
فاصله کانونی: ۵۰ میلی‌متر
اندازه سنسور: ۳۶ میلی‌متر
عرض تصویر: ۱۹۲۰ پیکسل

🎯 اول: فوکال لنت (Focal Length) چیه؟
فاصله کانونی = فاصله‌ی بین لنز و سنسور دوربین
     لنز              سنسور
      │                 │
      │←── فاصله ──→│
      │   کانونی    │
      مثال:

اگه لنز دوربینت ۵۰ میلی‌متر باشه، یعنی فاصله‌ی لنز تا سنسور ۵۰ میلی‌متر هست.
🎯 دوم: سنسور چیه؟
سنسور = صفحه‌ای که تصویر روش میفته
مثل فیلم توی دوربین‌های قدیمی.
🎯 سوم: عرض تصویر چیه؟
عرض تصویر = تعداد پیکسل‌های افقی
مثلاً یه عکس ۱۹۲۰×۱۰۸۰:
عرض: ۱۹۲۰ پیکسل
ارتفاع: ۱۰۸۰ پیکسل
📌 حالا فرمول:
fx = (فوکال لنت ÷ اندازه سنسور) × عرض تصویر
fx = (50 / 36) × 1920 = 2666 پیکسل
cx = 1920 / 2 = 960 پیکسل
پس ماتریس دوربین میشه:
پس ماتریس دوربین میشه:
K = [ 2666    0   960 ]
    [    0 2666   540 ]
    [    0    0     1 ]
# 📌 چرا این پارامترها مهمن؟
چون ثابت هستن و میشه ازشون استفاده کرد برای:
اندازه‌گیری دقیق
تصحیح اعوجاج
بازسازی ۳D
# ❓ سؤال:
به نظرت این پارامترها برای هر دوربین متفاوت هستن یا یکسان؟
 هر دوربین لنز خودش رو داره

# ✅  این بخش تئوری هست.
📌 Intrinsic Parameters کد مستقیم نداره!
چون این پارامترها از کالیبراسیون به دست میان، نه اینکه خودت توی کد بنویسیشون.

# 📌 ولی برای دیدن ماتریس دوربین می‌تونی از این کد استفاده کنی:

import cv2
import numpy as np

# ماتریس دوربین (نمونه)
K = np.array([[2666, 0, 960],
              [0, 2666, 540],
              [0, 0, 1]], dtype=np.float32)

print("ماتریس دوربین:")
print(K)

print(f"\nفوکال لنت fx: {K[0, 0]}")
print(f"فوکال لنت fy: {K[1, 1]}")
print(f"مرکز تصویر cx: {K[0, 2]}")
print(f"مرکز تصویر cy: {K[1, 2]}")
---------------------
# 📌 زیرشاخه ۲: Extrinsic Parameters
---------------------
Extrinsic Parameters = پارامترهای خارجی دوربین
یعنی موقعیت و جهت دوربین توی دنیای واقعی.
📌 فرق با Intrinsic:
 Intrinsic           	      Extrinsic
 دوربین خارجی              دوربین داخلی
 موقعیت و جهت دوربین    مشخصات لنز و سنسور
 متغیر (بسته به موقعیت)   ثابت برای هر دوربین
# 📌 این پارامترها چیا هستن؟
Rotation (R)	چرخش دوربین
Translation (T)	جابه‌جایی دوربین
# 📌 مثال ساده:
فرض کن یه دوربین روی سه‌پایه داری.
اگه سه‌پایه رو بچرخونی:
Intrinsic: تغییری نمی‌کنه
Extrinsic: عوض میشه (چرخش)
اگه سه‌پایه رو جابه‌جا کنی:
Intrinsic: تغییری نمی‌کنه
Extrinsic: عوض میشه (انتقال)
# 📌 ماتریس خارجی:
[R | T]
R	ماتریس چرخش (۳×۳)
T	بردار انتقال (۳×۱)
# 📌 چرا این پارامترها مهمن؟
بازسازی ۳D       	موقعیت دوربین رو می‌دونیم
موقعیت ربات رو می‌دونیم      بینایی رباتیک
موقعیت خودرو رو می‌دونیم             خودران
SLAM	            حرکت دوربین رو می‌دونیم

-------------------
# 📌 زیرشاخه ۳: Distortion Correction
-------------------
Distortion Correction = تصحیح اعوجاج
یعنی درست کردن تصویر کج که به خاطر لنز دوربین ایجاد شده.
خطوط صاف میشن
تصویر واقعی‌تر میشه
# 📌 دو نوع اعوجاج که تصحیح میشن:
Radial	خطوط به بیرون/داخل خم میشن
Tangential	تصویر کمی می‌چرخه
# 📌 کد Distortion Correction:
# ۱. ماتریس دوربین
K = np.array([[1000, 0, 320],
              [0, 1000, 240],
              [0, 0, 1]])

# ۲. ضرایب اعوجاج
dist = np.array([0.1, -0.05, 0, 0, 0])

# ۳. تصحیح
undistorted = cv2.undistort(img, K, dist)

K	  ماتریس دوربین (Intrinsic)
dist	          ضرایب اعوجاج
cv2.undistort(...)	تابع تصحیح
# 📌 ضرایب اعوجاج (dist) چیا هستن؟
یه آرایه با ۵ عدد:
dist = [k1, k2, p1, p2, k3]
k1, k2, k3	اعوجاج شعاعی
p1, p2	    اعوجاج مماسی  

# 📌 اول: k1, k2, k3 (اعوجاج شعاعی)
شعاعی = از مرکز تصویر به سمت بیرون.
چیکار می‌کنه؟
خطوط صاف رو خم می‌کنه
k1 > 0 (مثلاً ۰.۱)	خطوط به داخل خم میشن
k1 < 0 (مثلاً -۰.۱)	خطوط به بیرون خم میشن
k1 = 0	خطوط صاف می‌مونن

مثال 
واقعیت:        با k1 > 0:        با k1 < 0:

┌─────┐        ╭─────╮          ╰─────╯
│     │   →    │     │     →    │     │
└─────┘        ╰─────╯          ╭─────╮

# 📌 دوم: p1, p2 (اعوجاج مماسی)
مماسی = چرخش یا کجی تصویر.
چیکار می‌کنه؟
تصویر رو کمی می‌چرخونه یا کج می‌کنه.
p1 > 0	تصویر کمی به یه طرف کج میشه
p2 > 0	تصویر کمی به طرف دیگه کج میشه
p1 = p2 = 0	تصویر کج نمیشه

مثال
واقعیت:        با p1 > 0:

┌─────┐        ╱─────╲
│     │   →   ╱       ╲
└─────┘        ╲─────╱

📌 سوم: k3
k3 = یه ضریب اضافی برای اعوجاج شعاعی.
معمولاً ۰ هست (اگه لنز خیلی اعوجاج داشته باشه، ازش استفاده میشه).

# 📌 حالا مثال ما:
dist = np.array([0.1, -0.05, 0, 0, 0])
k1	۰.۱  	خطوط به داخل خم میشن
k2	-۰.۰۵	تصحیح خم
p1	۰	    بدون چرخش
p2	۰	    بدون کجی
k3	۰	    بدون اعوجاج اضافی

# 📌 این مقادیر از کجا میان؟
از کالیبراسیون (Chessboard Calibration).
توی قسمت ۴، این مقادیر رو خودکار به دست میاریم.
# ❓ سؤال:
به نظرت اگه dist = [0, 0, 0, 0, 0] باشه، تصویر تغییر می‌کنه؟
 صفر یعنی هیچ اعوجاجی نیست

# 📌 کد ۱: Radial Distortion (اعوجاج شعاعی)
import cv2
import numpy as np

# ۱. خواندن تصویر
img = cv2.imread("E:/cp-vision/left01.jpg")
h, w = img.shape[:2]

# ۲. ماتریس دوربین
K = np.array([[1000, 0, w/2],
              [0, 1000, h/2],
              [0, 0, 1]], dtype=np.float32)

# ۳. فقط اعوجاج شعاعی
dist_radial = np.array([0.5, 0, 0, 0, 0])
#                         ↑
#                        k1

# ۴. تصحیح
undist_radial = cv2.undistort(img, K, dist_radial)

# ۵. نمایش
cv2.imshow("Radial Distortion", undist_radial)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("E:/cp-vision/radial.jpg", undist_radial)

# 📌 کد ۲: Tangential Distortion (اعوجاج مماسی)
import cv2
import numpy as np

# ۱. خواندن تصویر
img = cv2.imread("E:/cp-vision/left01.jpg")
h, w = img.shape[:2]

# ۲. ماتریس دوربین
K = np.array([[1000, 0, w/2],
              [0, 1000, h/2],
              [0, 0, 1]], dtype=np.float32)

# ۳. فقط اعوجاج مماسی
dist_tangential = np.array([0, 0, 0.1, 0, 0])
#                              ↑  ↑
#                             p1 p2

# ۴. تصحیح
undist_tangential = cv2.undistort(img, K, dist_tangential)

# ۵. نمایش
cv2.imshow("Tangential Distortion", undist_tangential)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("E:/cp-vision/tangential.jpg", undist_tangential)

# 📌 کد ۳: ترکیبی (هر دو)
import cv2
import numpy as np

# ۱. خواندن تصویر
img = cv2.imread("E:/cp-vision/left01.jpg")
h, w = img.shape[:2]

# ۲. ماتریس دوربین
K = np.array([[1000, 0, w/2],
              [0, 1000, h/2],
              [0, 0, 1]], dtype=np.float32)

# ۳. ترکیبی
dist_both = np.array([0.3, 0, 0.05, 0.05, 0])
#                      ↑     ↑    ↑
#                     k1    p1   p2

# ۴. تصحیح
undist_both = cv2.undistort(img, K, dist_both)

# ۵. نمایش
cv2.imshow("Both", undist_both)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("E:/cp-vision/both.jpg", undist_both)


۱	k1 = 0.5	خطوط خم میشن
۲	p1 = 0.1	تصویر کج میشه
۳	k1 + p1 + p2	هم خم، هم کج


--------------------
# 📌 زیرشاخه ۴: Chessboard Calibration
--------------------
Chessboard Calibration = کالیبراسیون با صفحه شطرنجی
یعنی با استفاده از یه صفحه شطرنجی، پارامترهای دوربین رو پیدا می‌کنیم.

# 📌 چرا صفحه شطرنجی؟
چون صفحه شطرنجی:
گوشه‌های واضح داره
قابل تشخیص هست
دقیق هست
# 📌 چطور کار می‌کنه؟
۱	از صفحه شطرنجی چند تا عکس می‌گیریم (از زوایای مختلف)
۲	گوشه‌های صفحه شطرنجی رو توی هر عکس پیدا می‌کنیم
۳	با این گوشه‌ها، پارامترهای دوربین رو حساب می‌کنیم
# 📌 خروجی چیه؟
K	ماتریس دوربین (Intrinsic)
dist	ضرایب اعوجاج
rvecs	بردارهای چرخش
tvecs	بردارهای انتقال
# 📌 کد Chessboard Calibration:
# ۱. پیدا کردن گوشه‌های صفحه شطرنجی
ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

# ۲. کالیبراسیون
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    object_points, image_points, gray.shape[::-1], None, None)

cv2.findChessboardCorners(...)	پیدا کردن گوشه‌ها
(7, 6)	تعداد گوشه‌ها       (۷ ستون، ۶ ردیف)
cv2.calibrateCamera(...)	    کالیبراسیون
K                      	       ماتریس دوربین
dist	                       ضرایب اعوجاج 
# 📌 نکته مهم:
صفحه شطرنجی باید:
تخت باشه
واضح باشه
از زوایای مختلف عکس گرفته بشه (حداقل ۱۰-۱۵ عکس)
# ❓ سؤال:
به نظرت چرا باید از زوایای مختلف عکس بگیریم؟
🎯 دلیل: برای پیدا کردن پارامترهای مختلف دوربین
# 📌 مثال ساده:
فرض کن می‌خوای ابعاد یه اتاق رو اندازه بگیری:
اگه فقط از یه گوشه نگاه کنی → فقط ۲ تا دیوار رو می‌بینی
اگه از چند گوشه نگاه کنی → همه‌ی دیوارها رو می‌بینی
کالیبراسیون هم همینه.
# 📌 پارامترهایی که با زوایای مختلف پیدا میشن:


زاویه       پارامتری که پیدا میشه
---------------------------------------
روبه‌رو     → مرکز تصویر (cx, cy)
کج    → اعوجاج شعاعی (k1, k2)
چرخیده          → چرخش (p1, p2)
دور       → فوکال لنت (fx, fy)
نزدیک          → مقیاس
هر زاویه، یه پارامتر رو بهتر نشون میده.

# نکته
📌 بله! باید صفحه شطرنجی باشه.
♟️ صفحه شطرنجی:

⬛⬜⬛⬜⬛⬜⬛
⬜⬛⬜⬛⬜⬛⬜
⬛⬜⬛⬜⬛⬜⬛
⬜⬛⬜⬛⬜⬛⬜
⬛⬜⬛⬜⬛⬜⬛
⬜⬛⬜⬛⬜⬛⬜

و 14 الی 15 تا عکس باید باشه

# 📌 کد کامل Chessboard Calibration
import cv2
import numpy as np
import glob

# ۱. تنظیمات صفحه شطرنجی
CHESSBOARD = (7, 6)
# ☝️ تعداد گوشه‌های شطرنجی (۷ ستون، ۶ ردیف)
#    اگه صفحه‌ات فرق داره، این عدد رو عوض کن

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# ☝️ معیار توقف الگوریتم
#    30 = حداکثر تکرار
#    0.001 = دقت مورد نظر

# ۲. نقاط 3D فرضی
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
# ☝️ آرایه‌ی صفر به اندازه تعداد گوشه‌ها (۴۲ گوشه)

objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
# ☝️ پر کردن مختصات (x, y) گوشه‌ها
#    z همیشه صفر (چون صفحه تخت)

# ۳. لیست‌های ذخیره‌سازی
objpoints = []
# ☝️ نقاط 3D (دنیای واقعی)

imgpoints = []
# ☝️ نقاط 2D (توی تصویر)

# ۴. خواندن عکس‌ها
images = glob.glob("E:/cp-vision/chessboard/*.jpg")
# ☝️ پیدا کردن همه عکس‌های jpg توی پوشه

print(f"تعداد عکس‌ها: {len(images)}")

# ۵. چک کردن عکس‌ها
if len(images) == 0:
    print("❌ هیچ عکسی توی پوشه پیدا نشد!")
    print("لطفاً عکس‌های شطرنجی رو توی پوشه E:/cp-vision/chessboard/ بذار.")
else:
    # ۶. حلقه روی عکس‌ها
    for fname in images:
        img = cv2.imread(fname)
        # ☝️ خواندن عکس
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ☝️ تبدیل به خاکستری
        
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
        # ☝️ پیدا کردن گوشه‌های شطرنجی
        #    ret = True/False
        #    corners = مختصات گوشه‌ها
        
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # ☝️ بهبود دقت گوشه‌ها (زیرپیکسلی)
            
            objpoints.append(objp)
            # ☝️ ذخیره نقاط 3D
            
            imgpoints.append(corners2)
            # ☝️ ذخیره نقاط 2D
            
            cv2.drawChessboardCorners(img, CHESSBOARD, corners2, ret)
            # ☝️ رسم گوشه‌ها روی عکس
            
            cv2.imshow("Chessboard", img)
            cv2.waitKey(300)
            # ☝️ نمایش هر عکس ۳۰۰ میلی‌ثانیه
    
    cv2.destroyAllWindows()
    
    # ۷. کالیبراسیون
    if len(objpoints) > 0:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        # ☝️ کالیبراسیون دوربین
        #    ورودی: نقاط 3D و 2D
        #    خروجی:
        #    K = ماتریس دوربین
        #    dist = ضرایب اعوجاج
        #    rvecs = بردارهای چرخش
        #    tvecs = بردارهای انتقال
        
        print("\n✅ کالیبراسیون موفق بود!")
        print(f"\nماتریس دوربین:\n{K}")
        print(f"\nضرایب اعوجاج:\n{dist}")
        
        # ۸. ذخیره پارامترها
        np.savez("E:/cp-vision/calibration.npz", K=K, dist=dist)
        # ☝️ ذخیره پارامترها توی فایل npz
        print("\n✅ پارامترها ذخیره شدند: calibration.npz")
    else:
        print("❌ هیچ گوشه‌ای پیدا نشد!")
# 📌 توضیح خط به خط:
۱	CHESSBOARD = (7, 6)	تعداد گوشه‌ها
۲	criteria = (...)	معیار توقف
۳	objp = np.zeros(...)	نقاط 3D فرضی
۴	objp[:, :2] = ...	پر کردن مختصات
۵	objpoints = []	لیست نقاط 3D
۶	imgpoints = []	لیست نقاط 2D
۷	glob.glob(...)	پیدا کردن عکس‌ها
۸	cv2.findChessboardCorners(...)	پیدا کردن گوشه‌ها
۹	cv2.cornerSubPix(...)	بهبود دقت
۱۰	cv2.drawChessboardCorners(...)	رسم گوشه‌ها
۱۱	cv2.calibrateCamera(...)	کالیبراسیون
۱۲	np.savez(...)	ذخیره پارامترها
# 📌 import glob چیه؟
🎯 glob = یه کتابخانه برای پیدا کردن فایل‌ها
📌 کارش چیه؟
میاد توی یه پوشه می‌گرده و فایل‌های مورد نظر رو پیدا می‌کنه.
# 📌 ساختار پوشه:
E:/cp-vision/
├── chessboard/           ← عکس‌های شطرنجی
│   ├── img01.jpg
│   ├── img02.jpg
│   ├── ...
│   └── img15.jpg
├── test.py               ← این کد
└── calibration.npz       ← خروجی (ذخیره خودکار)


---------------------
# 📌 ۰۴.۴ (قسمت ۴): Chessboard Calibration
---------------------
Chessboard Calibration = کالیبراسیون با صفحه شطرنجی
یعنی با استفاده از صفحه شطرنجی، پارامترهای دوربین رو پیدا می‌کنیم.
📌 چرا صفحه شطرنجی؟
چون گوشه‌های واضح و دقیق داره.
OpenCV می‌تونه این گوشه‌ها رو دقیق پیدا کنه.
🎯 پارامترهای دوربین = مشخصات دوربین
مثل وقتی می‌خوای یه دوربین رو بشناسی:
لنزش چیه؟
چقدر بزرگ هست؟
اعوجاج داره یا نه؟
کجا نصب شده؟
# 📌 چطور کار می‌کنه؟
مرحله	کار
۱	از صفحه شطرنجی ۱۰-۱۵ عکس از زوایای مختلف می‌گیریم
۲	OpenCV گوشه‌ها رو توی هر عکس پیدا می‌کنه
۳	با این گوشه‌ها، پارامترهای دوربین رو حساب می‌کنه
# 📌 خروجی چیه؟
خروجی	توضیح
K	ماتریس دوربین
dist	ضرایب اعوجاج
rvecs	بردارهای چرخش
tvecs	بردارهای انتقال
📌 بعد از کالیبراسیون چیکار می‌کنیم؟
از این پارامترها برای هر عکس جدید با همون دوربین استفاده می‌کنیم:
تصحیح اعوجاج
اندازه‌گیری دقیق
بازسازی ۳D
# 📌 کد (خلاصه):
# ۱. پیدا کردن گوشه‌ها
ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

# ۲. کالیبراسیون
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

# 📌 کد کامل Chessboard Calibration
باز 15 تا عکس از زاویه های مختلف نیازه

import cv2
import numpy as np
import glob

# ۱. تنظیمات
CHESSBOARD = (7, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ۲. نقاط 3D
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# ۳. خواندن عکس‌ها
images = glob.glob("E:/cp-vision/chessboard/*.jpg")
print(f"تعداد عکس‌ها: {len(images)}")

if len(images) == 0:
    print("❌ هیچ عکسی توی پوشه پیدا نشد!")
else:
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
        
        if ret == True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            cv2.drawChessboardCorners(img, CHESSBOARD, corners2, ret)
            cv2.imshow("Chessboard", img)
            cv2.waitKey(300)
    
    cv2.destroyAllWindows()
    
    # ۴. کالیبراسیون
    if len(objpoints) > 0:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        print("\n✅ کالیبراسیون موفق بود!")
        print(f"\nماتریس دوربین:\n{K}")
        print(f"\nضرایب اعوجاج:\n{dist}")
        
        # ۵. ذخیره
        np.savez("E:/cp-vision/calibration.npz", K=K, dist=dist)
        print("\n✅ پارامترها ذخیره شدند.")
    else:
        print("❌ هیچ گوشه‌ای پیدا نشد!")

    
# 📌 توضیح خط به خط کد Chessboard Calibration
خط ۱: import cv2
وارد کردن OpenCV

خط ۲: import numpy as np
وارد کردن NumPy (برای کار با آرایه‌ها)

خط ۳: import glob
وارد کردن glob (برای پیدا کردن فایل‌ها)

خط ۴: CHESSBOARD = (7, 6)
تعداد گوشه‌های صفحه شطرنجی

عدد	معنی
۷	تعداد گوشه‌ها در جهت افقی
۶	تعداد گوشه‌ها در جهت عمودی
⚠️ نکته: این عدد رو باید دقیق بذاری. اگه صفحه‌ات ۸×۶ هست، بذار (8, 6).

خط ۵: criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
معیار توقف الگوریتم

قسمت	توضیح
TERM_CRITERIA_EPS	توقف بر اساس دقت
TERM_CRITERIA_MAX_ITER	توقف بر اساس تعداد تکرار
30	حداکثر ۳۰ تکرار
0.001	دقت مورد نظر
یعنی: الگوریتم تا ۳۰ بار یا تا دقت ۰.۰۰۱ تکرار کن.

خط ۶: objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
ساخت آرایه‌ی صفر برای نقاط 3D

قسمت	توضیح
CHESSBOARD[0] * CHESSBOARD[1]	تعداد کل گوشه‌ها (۷×۶ = ۴۲)
3	سه بعد (x, y, z)
np.float32	نوع داده اعشاری
نتیجه: آرایه‌ی ۴۲×۳ پر از صفر.

خط ۷: objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
پر کردن مختصات (x, y) گوشه‌ها

قسمت	توضیح
np.mgrid[0:7, 0:6]	ساخت شبکه‌ی مختصات
.T	ترانهاده
.reshape(-1, 2)	تبدیل به آرایه‌ی n×۲
نتیجه: مختصات گوشه‌ها مثل (0,0)، (1,0)، (2,0)، ...

نکته: z همیشه صفر می‌مونه (چون صفحه تخت هست).

خط ۸: objpoints = []
لیست خالی برای نقاط 3D (دنیای واقعی)

خط ۹: imgpoints = []
لیست خالی برای نقاط 2D (توی تصویر)

خط ۱۰: images = glob.glob("E:/cp-vision/chessboard/*.jpg")
پیدا کردن همه عکس‌های jpg توی پوشه

قسمت	توضیح
glob.glob(...)	تابع پیدا کردن فایل
*.jpg	همه فایل‌های jpg
خط ۱۱: print(f"تعداد عکس‌ها: {len(images)}")
چاپ تعداد عکس‌های پیدا شده

خط ۱۲: if len(images) == 0:
اگه عکسی پیدا نشد

خط ۱۳: print("❌ هیچ عکسی توی پوشه پیدا نشد!")
پیام خطا

خط ۱۴: for fname in images:
حلقه روی همه عکس‌ها

خط ۱۵: img = cv2.imread(fname)
خواندن هر عکس

خط ۱۶: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
تبدیل به خاکستری (چون findChessboardCorners با خاکستری کار می‌کنه)

خط ۱۷: ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
قلب ماجرا! پیدا کردن گوشه‌های صفحه شطرنجی

خروجی	توضیح
ret	True/False (پیدا شد یا نه)
corners	مختصات گوشه‌ها
خط ۱۸: if ret == True:
اگه گوشه‌ها پیدا شدن

خط ۱۹: corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
بهبود دقت گوشه‌ها (زیرپیکسلی)

قسمت	توضیح
(11, 11)	اندازه پنجره
(-1, -1)	ناحیه‌ی صفر
criteria	معیار توقف
چرا؟ دقت رو از پیکسل به زیرپیکسل می‌بره.

خط ۲۰-۲۱: objpoints.append(objp) و imgpoints.append(corners2)
ذخیره‌ی نقاط 3D و 2D

خط ۲۲: cv2.drawChessboardCorners(img, CHESSBOARD, corners2, ret)
رسم گوشه‌ها روی عکس

خط ۲۳-۲۴: cv2.imshow(...) و cv2.waitKey(300)
نمایش هر عکس ۳۰۰ میلی‌ثانیه

خط ۲۵: cv2.destroyAllWindows()
بستن پنجره‌ها

خط ۲۶: if len(objpoints) > 0:
اگه حداقل یه عکس گوشه‌ها رو داشته

خط ۲۷: ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(...)
کالیبراسیون نهایی!

خروجی	توضیح
K	ماتریس دوربین
dist	ضرایب اعوجاج
rvecs	بردارهای چرخش
tvecs	بردارهای انتقال
خط ۲۸-۳۰: print(...)
چاپ نتایج

خط ۳۱: np.savez("E:/cp-vision/calibration.npz", K=K, dist=dist)
ذخیره پارامترها توی فایل .npz

چرا؟ که بعداً بتونی ازشون استفاده کنی.

خط ۳۲: else: print("❌ هیچ گوشه‌ای پیدا نشد!")
اگه هیچ گوشه‌ای پیدا نشد



------------------پایان بخش 4 ------------------------------
----------------------------------
# 📌 ۰۴.۵ — Object Tracking (ردیابی اشیا)
----------------------------------
Object Tracking = ردیابی اشیا توی ویدیو.
یعنی دنبال کردن یه شیء توی فریم‌های ویدیو.
# 📌 ۴ زیرشاخه داره:
۱	Optical Flow	ردیابی حرکت پیکسل‌ها
۲	MeanShift	ردیابی بر اساس رنگ
۳	CamShift	نسخه بهبودیافته MeanShift
۴	KLT Tracker	ردیابی نقاط کلیدی
# 📌 زیرشاخه ۱: Optical Flow
Optical Flow = جریان نوری = ردیابی حرکت پیکسل‌ها
یعنی میاد حرکت هر پیکسل رو بین دو فریم حساب می‌کنه
📌 مثال ساده:
فرض کن یه توپ توی ویدیو حرکت می‌کنه:
فریم ۱:
⚽ ← اینجا
فریم ۲:
    ⚽ ← حالا اینجا
Optical Flow میاد
جهت حرکت رو حساب می‌کنه (راست)
سرعت حرکت رو حساب می‌کنه (مثلاً ۱۰ پیکسل)
# 📌 چرا مهمه؟
ردیابی اشیا  	دنبال کردن اشیا
تشخیص حرکت	  فهمیدن جهت حرکت
خودران          	حرکت موانع
ورزش	تحلیل حرکت بازیکنان
ویدیو           	فشرده‌سازی
# 📌 دو نوع Optical Flow:
Sparse	فقط بعضی نقاط (سریع)
Dense	همه پیکسل‌ها (کند)
# 📌 الگوریتم‌ها:
Lucas-Kanade	Sparse
Farneback	    Dense
Horn-Schunck	Dense
-----------
# 📌 روش Lucas-Kanade
----------
# 📌 عنوان: Lucas-Kanade Optical Flow
یعنی فقط بعضی نقاط رو ردیابی می‌کنه (نه همه پیکسل‌ها).
# 📌 چطور کار می‌کنه؟
۱	گوشه‌های مهم رو توی فریم اول پیدا می‌کنه
۲	همون گوشه‌ها رو توی فریم دوم پیدا می‌کنه
۳	حرکت هر گوشه رو حساب می‌کنه
# 📌 چرا گوشه؟
چون گوشه‌ها متمایز هستن و راحت پیدا میشن.
# 📌 کد Lucas-Kanade:
# ۱. پیدا کردن گوشه‌ها
corners = cv2.goodFeaturesToTrack(gray1, 100, 0.3, 7)

# ۲. محاسبه Optical Flow
new_corners, status, err = cv2.calcOpticalFlowPyrLK(
    gray1, gray2, corners, None)
cv2.goodFeaturesToTrack(...)	پیدا کردن گوشه‌ها
cv2.calcOpticalFlowPyrLK(...)	محاسبه‌ی جریان نوری
gray1	فریم اول
gray2	فریم دوم
corners	گوشه‌های فریم اول
new_corners	گوشه‌های فریم دوم
status	وضعیت (پیدا شد یا نه)
gray1	تصویر خاکستری
100	حداکثر ۱۰۰ گوشه
0.3	کیفیت (۰-۱)
7	حداقل فاصله بین گوشه‌ها
# 📌 کد کامل Lucas-Kanade Optical Flow
با عکس : دوتا عکس شبیه هم با جابه جا شده اشیا و... هاش
import cv2
import numpy as np

# ۱. خواندن دو عکس
img1 = cv2.imread("E:/cp-vision/left01.jpg")
img2 = cv2.imread("E:/cp-vision/left02.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ۲. پیدا کردن گوشه‌ها توی عکس اول
corners1 = cv2.goodFeaturesToTrack(gray1, 100, 0.3, 7)

# ۳. محاسبه Optical Flow
corners2, status, err = cv2.calcOpticalFlowPyrLK(
    gray1, gray2, corners1, None)

# ۴. انتخاب گوشه‌های موفق
good_new = corners2[status == 1]
good_old = corners1[status == 1]

# ۵. رسم خطوط حرکت
for new, old in zip(good_new, good_old):
    a, b = new.ravel()
    c, d = old.ravel()
    cv2.line(img2, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
    cv2.circle(img2, (int(a), int(b)), 5, (0, 0, 255), -1)

# ۶. نمایش
cv2.imshow("Optical Flow", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("E:/cp-vision/optical_flow.jpg", img2)
print("✅ تصویر ذخیره شد.")


خطوط سبز	جهت حرکت هر نقطه
دایره‌های قرمز	موقعیت جدید نقاط
خطوط	مسیر حرکت
گوشه‌ها توی فریم اول پیدا شدن
حرکت اون‌ها توی فریم دوم حساب شد
خطوط مسیر حرکت رو نشون میدن
# 📌 کد کامل Lucas-Kanade Optical Flow روی ویدیو
import cv2
import numpy as np

# ۱. باز کردن ویدیو
cap = cv2.VideoCapture("E:/cp-vision/gushe.mp4")

# ☝️ چک کن ویدیو باز شده
if not cap.isOpened():
    print("❌ ویدیو باز نشد!")
    exit()

# ☝️ اطلاعات ویدیو
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"✅ ویدیو باز شد")
print(f"FPS: {fps}")
print(f"تعداد فریم: {total}")
print(f"ابعاد: {width}×{height}")

# ۲. خواندن فریم اول
ret, frame1 = cap.read()

if not ret:
    print("❌ فریم اول خونده نشد!")
    cap.release()
    exit()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
corners1 = cv2.goodFeaturesToTrack(gray1, 100, 0.3, 7)
mask = np.zeros_like(frame1)

while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    corners2, status, err = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, corners1, None)
    
    if corners2 is not None:
        good_new = corners2[status == 1]
        good_old = corners1[status == 1]
        
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                            (0, 255, 0), 2)
            frame2 = cv2.circle(frame2, (int(a), int(b)), 5, 
                                 (0, 0, 255), -1)
        
        output = cv2.add(frame2, mask)
        cv2.imshow("Optical Flow", output)
    
    gray1 = gray2.copy()
    corners1 = good_new.reshape(-1, 1, 2)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
نتیجه:
✅ ویدیو باز شد
FPS: 25.0
تعداد فریم: 271
ابعاد: 720×720

-----------------
# MeanShift
-----------------
MeanShift = جابه‌جایی میانگین
یعنی یه روش ردیابی که بر اساس رنگ کار می‌کنه.
📌 داستان چیه؟
فرض کن می‌خوای یه توپ قرمز رو توی ویدیو ردیابی کنی.
MeanShift چیکار می‌کنه؟
۱. هیستوگرام رنگ توپ رو حساب می‌کنه (چقدر قرمز داره)
۲. توی هر فریم، دنبال اون رنگ می‌گرده
۳. موقعیت جدید توپ رو پیدا می‌کنه
# 📌 چرا MeanShift؟
راحت پیاده‌سازی میشه  ساده
سریع	      Real-time
شکل شیء مهم نیست   مقاوم به شکل
چرخش مهم نیست   مقاوم به چرخش
# 📌 عیب MeanShift:
اندازه شیء عوض بشه، مشکل داره     اندازه ثابت
نور عوض بشه، مشکل داره   حساسیت به نور
اگه رنگ مشابه باشه، گیج میشه  پس زمینه مشابه
# 📌 کد MeanShift (خلاصه):
# ۱. هیستوگرام رنگ
roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])

# ۲. نرمال‌سازی
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# ۳. ردیابی
ret, track_window = cv2.meanShift(back_proj, track_window, term_crit)

cv2.calcHist(...)	محاسبه هیستوگرام
هیستوگرام = نمودار توزیع رنگ
cv2.normalize(...)	نرمال‌سازی
نرمال‌سازی = هم‌مقیاس کردن اعداد
cv2.meanShift(...)	ردیابی
ناحیه‌ی قبلی توپ رو نگاه می‌کنه
هیستوگرامش رو حساب می‌کنه
دنبال ناحیه‌ای می‌گرده که هیستوگرامش شبیه‌ترین باشه
مرکز اون ناحیه رو پیدا می‌کنه → موقعیت جدید توپ
# خلاصه
calcHist	حساب کردن توزیع رنگ
normalize	هم‌مقیاس کردن اعداد
meanShift	پیدا کردن موقعیت جدید
 برای این بخش تنها عکس کافی نیست یا وبکم یا ویدیو باید باشه


 اگر وبکمتون گیر کرد اینو بزنین
 win + r = taskkill /F /IM python.exe

 وقتی اجرا میشه دور یک شی با ماوس مستطیل بکش بعد اینتر بزن  بعد اروم تکون بده تا دنبالش بکنه

 # کد کامل

import cv2
import numpy as np
import time

# ۱. باز کردن وبکم
cap = cv2.VideoCapture(0)
time.sleep(2)

# ۲. خواندن فریم اول
ret, frame = cap.read()
if not ret:
    print("❌ وبکم فریم نداد!")
    cap.release()
    exit()

# ۳. انتخاب ناحیه (ROI) با ماوس
r = cv2.selectROI("Select Object", frame, False)
cv2.destroyWindow("Select Object")

# ۴. مختصات ناحیه
x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])

# ۵. ناحیه انتخاب شده
roi = frame[y:y+h, x:x+w]

# ۶. تبدیل به HSV
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# ۷. ماسک
mask = cv2.inRange(roi_hsv, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

# ۸. هیستوگرام
roi_hist = cv2.calcHist([roi_hsv], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# ۹. معیار توقف
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

# ۱۰. پنجره ردیابی
track_window = (x, y, w, h)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # ۱۱. تبدیل به HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # ۱۲. back projection
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    
    # ۱۳. MeanShift
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    
    # ۱۴. رسم مستطیل
    x, y, w, h = track_window
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # ۱۵. نمایش
    cv2.imshow("MeanShift", frame)
    
    # ۱۶. خروج با q
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

--------------
# 📌 زیرشاخه ۳: CamShift
--------------
CamShift = Continuously Adaptive MeanShift
یعنی نسخه بهبودیافته MeanShift که اندازه پنجره رو خودکار تنظیم می‌کنه.
# 📌 فرقش با MeanShift چیه؟
MeanShift	                              CamShift
اندازه تطبیقی                          اندازه ثابت
خودکار تنظیم میشه   اگه شیء دور/نزدیک بشه، مشکل داره
موقعیت + اندازه + چرخش                     فقط موقعیت 
# 📌 مثال ساده:
فرض کن یه توپ داری که داره دور میشه:
MeanShift: پنجره ثابت می‌مونه ❌
CamShift: پنجره کوچیک‌تر میشه ✅
# 📌 کد CamShift:
ret, track_window = cv2.CamShift(dst, track_window, term_crit)
dst	back projection
track_window	پنجره ردیابی
term_crit	معیار توقف
ret	اطلاعات چرخش
track_window	پنجره جدید

# 📌 تفاوت رسم:
MeanShift:
x, y, w, h = track_window
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

CamShift:
pts = cv2.boxPoints(ret)
pts = np.int0(pts)
cv2.polylines(frame, [pts], True, (0, 255, 0), 2)


# 📌 نسخه استاندارد (با selectROI):
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

# انتخاب ناحیه با ماوس
r = cv2.selectROI(frame, False)

# مختصات
x, y, w, h = int(r[0]), int(r[1]), int(r[2]), int(r[3])

roi = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
track_window = (x, y, w, h)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    
    pts = cv2.boxPoints(ret)
    pts = pts.astype(int)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    
    cv2.imshow("CamShift", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# 📌 کد نهایی CamShift (بدون selectROI):
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

# ناحیه ثابت (وسط تصویر)
h, w = frame.shape[:2]
x = w // 3
y = h // 3
w_roi = w // 3
h_roi = h // 3

roi = frame[y:y+h_roi, x:x+w_roi]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
track_window = (x, y, w_roi, h_roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
    
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)
    
    pts = cv2.boxPoints(ret)
    pts = pts.astype(int)
    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
    
    cv2.imshow("CamShift", frame)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


------------------------------
# 📌 زیرشاخه ۴: KLT Tracker (آخرین قسمت ۰۴.۵)
------------------------------
# 📌 توضیح کلی:
KLT = Kanade-Lucas-Tomasi
ترکیب دو روش:
Lucas-Kanade (Optical Flow)
Tomasi (گوشه‌یابی)
# 📌 کارش چیه؟
ردیابی نقاط کلیدی توی ویدیو.
یعنی:
۱. یه سری نقاط کلیدی انتخاب می‌کنه
۲. اون‌ها رو توی فریم‌های بعدی ردیابی می‌کنه
# 📌 فرقش با Optical Flow چیه؟
Optical Flow	KLT Tracker
ردیابی مداوم      فقط حرکت
نقاط جدید اضافه میشن  نقاط گم میشن
پایدارتر               ساده 
# 📌 کد KLT Tracker:
# ۱. پیدا کردن گوشه‌ها
p0 = cv2.goodFeaturesToTrack(gray1, 100, 0.3, 7)

# ۲. ردیابی
p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None)

# ۳. انتخاب نقاط موفق
good_new = p1[st == 1]
good_old = p0[st == 1]

# 📌 کد کامل KLT Tracker
import cv2
import numpy as np

# ۱. باز کردن وبکم
cap = cv2.VideoCapture(0)

# ۲. خواندن فریم اول
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# ۳. پیدا کردن گوشه‌ها
p0 = cv2.goodFeaturesToTrack(old_gray, 100, 0.3, 7)

# ۴. ماسک برای رسم
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ۵. محاسبه Optical Flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
    
    # ۶. انتخاب نقاط موفق
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    # ۷. رسم خطوط و نقاط
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
    
    # ۸. ترکیب
    img = cv2.add(frame, mask)
    cv2.imshow("KLT Tracker", img)
    
    # ۹. آپدیت
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    
    # ۱۰. خروج با q
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

نتیجه :
گوشه‌های مهم تصویر رو پیدا می‌کنه (با goodFeaturesToTrack)
روی هر گوشه یه نقطه قرمز می‌ذاره
با خط سبز مسیر حرکتش رو نشون میده

# 📌 توضیح خط به خط:
۱	cv2.VideoCapture(0)	باز کردن وبکم
۲	cv2.goodFeaturesToTrack(...)	پیدا کردن گوشه‌ها
۳	mask = np.zeros_like(old_frame)	ماسک مشکی
۴	cv2.calcOpticalFlowPyrLK(...)	محاسبه Optical Flow
۵	good_new = p1[st == 1]	نقاط موفق
۶	cv2.line(...)	رسم خط
۷	cv2.circle(...)	رسم نقطه
۸	cv2.add(frame, mask)	ترکیب
۹	old_gray = frame_gray.copy()	آپدیت فریم
۱۰	p0 = good_new.reshape(-1, 1, 2)	آپدیت نقاط
