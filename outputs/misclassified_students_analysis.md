# Characteristics of Misclassified Students
This report presents an in-depth error profile analysis of the **Logistic Regression** model, which achieved the best classification performance on the test set.

## Prediction Category Breakdown
- **True Positives (TP)**: 265 students (Actual Dropouts correctly identified)
- **True Negatives (TN)**: 409 students (Actual Graduates correctly identified)
- **False Positives (FP)**: 33 students (Actual Graduates predicted as Dropouts)
- **False Negatives (FN)**: 19 students (Actual Dropouts predicted as Graduates — *High Risk*)

## Mean Profile Comparison of Key Features

| Feature | TP (N=265) | TN (N=409) | FP (N=33) | FN (N=19) |
| --- | --- | --- | --- | --- |
| **Age at enrollment** | 26.35 | 21.71 | 20.24 | 24.11 |
| **Scholarship holder** | 9.1% | 34.5% | 21.2% | 10.5% |
| **Debtor** | 22.6% | 4.6% | 9.1% | 10.5% |
| **Tuition fees up to date** | 63.4% | 99.3% | 90.9% | 100.0% |
| **Gender** | 55.8% | 28.1% | 57.6% | 31.6% |
| **Displaced** | 41.9% | 61.6% | 84.8% | 63.2% |
| **Curricular units 1st sem (approved)** | 1.98 | 6.39 | 2.79 | 7.16 |
| **Curricular units 1st sem (grade)** | 6.50 | 12.83 | 7.61 | 12.14 |
| **Curricular units 2nd sem (approved)** | 1.38 | 6.37 | 2.82 | 6.79 |
| **Curricular units 2nd sem (grade)** | 5.26 | 12.96 | 7.46 | 12.54 |

*Note: Binary flags (Scholarship holder, Debtor, Tuition fees up to date, Gender, Displaced) are represented as percentages of the group.*

## Qualitative Insights & Educational Implications

### 1. Profil False Negatives (FN) — Mahasiswa Dropout yang Lolos dari Prediksi
Mahasiswa dalam kelompok False Negative adalah kasus paling berisiko karena sistem gagal menandai mereka untuk intervensi. Dari data profil di atas:
- **Performa Akademik yang 'Menipu'**: Mahasiswa FN memiliki rata-rata unit semester 2 yang disetujui sebanyak **6.79** unit. Ini jauh lebih tinggi daripada kelompok TP (**1.38** unit) dan hampir menyamai kelompok TN (**6.37** unit). Hal ini menunjukkan bahwa secara akademis pada semester awal, mereka tampak aman, namun ada faktor lain yang memicu keputusan untuk keluar. Asosiasi misklasifikasi ini kemungkinan berkaitan dengan faktor-faktor yang tidak terobservasi (*unobserved factors*) yang tidak tercakup dalam dataset ini.
- **Keterbatasan Variabel Finansial**: Kelompok FN memiliki tingkat tunggakan (Debtor) sebesar **10.5%** dan ketepatan pembayaran SPP (Tuition fees up to date) sebesar **100.0%**. Meskipun catatan administrasi pembayaran SPP mereka 100% lancar, kelompok FN tampaknya dipengaruhi oleh faktor-faktor lain yang tidak direpresentasikan secara kuat oleh variabel akademis dan finansial yang tersedia di dataset.
- **Rekomendasi Intervensi**: Mahasiswa dengan performa akademik baik tetapi tetap berisiko dropout memerlukan mekanisme pemantauan tambahan di luar indikator akademik tradisional. Temuan ini menunjukkan bahwa sistem peringatan dini berbasis data akademik perlu dilengkapi dengan sumber informasi lain yang dapat menangkap faktor-faktor yang tidak tersedia dalam dataset.

### 2. Profil False Positives (FP) — Mahasiswa Graduate yang Diprediksi Dropout
Mahasiswa dalam kelompok False Positive diprediksi akan dropout, namun pada kenyataannya mereka berhasil lulus. Kelompok ini mewakili mahasiswa yang memiliki daya tahan (*resilience*) tinggi:
- **Resiliensi Akademik**: Meskipun mahasiswa FP memiliki performa akademik semester awal yang marjinal or bermasalah (rata-rata unit semester 2 disetujui = **2.82** unit, jauh di bawah kelompok lulusan TN), mereka mampu bertahan dan menyelesaikan studinya.
### 3. Pengaruh Variabel 'Displaced' (Mahasiswa Rantau)
- **Tingginya Proporsi Displaced pada FP**: Kelompok False Positive (FP) memiliki proporsi mahasiswa displaced (perantau) yang sangat tinggi, yaitu sebesar **84.8%**, dibandingkan dengan kelompok True Negative (TN) yang sebesar **61.6%**. Perbedaan besar ini menunjukkan bahwa status displaced sering kali ditafsirkan oleh model sebagai faktor risiko dropout yang kuat.
- **Faktor Risiko vs Mekanisme Koping**: Temuan ini menunjukkan bahwa status displaced mungkin berfungsi sebagai faktor risiko awal yang dipertimbangkan oleh model. Namun, tingginya proporsi mahasiswa displaced pada kelompok lulusan mengindikasikan bahwa status tersebut tidak selalu berujung pada dropout dan kemungkinan berinteraksi dengan faktor-faktor lain yang tidak tercakup dalam dataset.

### 4. Dampak Penerima Beasiswa (Scholarship Holder)
- **Proporsi Penerima Beasiswa yang Kontras**: Hanya **9.1%** dari kelompok TP dan **10.5%** dari kelompok FN yang merupakan penerima beasiswa, dibandingkan dengan **34.5%** pada kelompok lulusan (TN). Pola yang sangat kuat ini menegaskan bahwa penerima beasiswa secara signifikan lebih condong masuk ke dalam kelompok lulusan.
- **Mekanisme Beasiswa**: Tingginya proporsi penerima beasiswa pada kelompok lulusan menunjukkan adanya hubungan positif antara status penerima beasiswa dan keberhasilan studi. Temuan ini mengindikasikan bahwa beasiswa dapat berperan sebagai faktor pendukung keberlangsungan studi, meskipun mekanisme spesifik yang mendasari hubungan tersebut tidak dapat ditentukan secara langsung dari dataset yang tersedia.

### 5. Ringkasan Sintesis untuk Pembahasan Tesis
> The error analysis suggests that student dropout is strongly associated with early academic performance; however, a small subset of students with seemingly satisfactory academic progress still withdraw from university. These students are difficult to identify using conventional academic and financial indicators, indicating the possible influence of additional unobserved factors beyond those available in the dataset. Consequently, institutions should not rely exclusively on academic performance metrics when designing early-warning systems.
