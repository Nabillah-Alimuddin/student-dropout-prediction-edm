# Post-Ablation Research Review & Next Steps

## Student Dropout Prediction using XGBoost, SMOTE-ENN, and SHAP

---

# Executive Summary

Ablation Study berhasil menjadi titik balik penelitian.

Sebelumnya terdapat asumsi bahwa kombinasi:

```text
XGBoost + SMOTE-ENN + Optuna + Threshold Optimization
```

akan menghasilkan performa terbaik.

Namun hasil eksperimen menunjukkan bahwa asumsi tersebut tidak sepenuhnya benar.

Temuan ini bukan kegagalan penelitian.

Sebaliknya, temuan ini merupakan hasil ilmiah yang valid karena memberikan bukti empiris mengenai efektivitas setiap komponen pada dataset yang digunakan.

---

# Current Findings

## Ablation Study Results

| Configuration          | F1-Dropout |
| ---------------------- | ---------: |
| XGBoost Default        |     0.9081 |
| XGBoost + Optuna       |     0.9043 |
| XGBoost + SMOTE        |     0.9007 |
| XGBoost + SMOTE-ENN    |     0.8878 |
| Full Proposed Pipeline |     0.8917 |

---

## Key Observation

Model baseline justru menghasilkan performa terbaik.

Secara umum:

* Optuna tidak memberikan peningkatan berarti.
* SMOTE menurunkan performa.
* SMOTE-ENN menurunkan performa lebih jauh.
* Threshold Optimization tidak menghasilkan perubahan signifikan.
* Full Pipeline tidak mampu mengungguli baseline.

---

# Research Question Status

## RQ1

### Research Question

Apakah SMOTE-ENN meningkatkan performa XGBoost pada prediksi dropout mahasiswa?

### Current Answer

Tidak pada dataset ini.

Hasil eksperimen menunjukkan bahwa:

```text
XGBoost Default
>
XGBoost + SMOTE-ENN
```

baik dari sisi F1 maupun beberapa metrik utama lainnya.

---

## Interpretation

Temuan ini menunjukkan bahwa:

* Dataset kemungkinan tidak cukup imbalanced untuk memperoleh manfaat besar dari SMOTE-ENN.
* XGBoost telah mampu menangani distribusi kelas dengan baik tanpa resampling tambahan.
* Resampling berlebihan berpotensi mengubah distribusi alami data sehingga mengurangi kemampuan generalisasi model.

---

## Research Value

Hasil negatif tetap merupakan hasil penelitian yang valid.

Kontribusi penelitian tidak harus berupa peningkatan performa.

Menunjukkan bahwa suatu teknik tidak efektif pada kondisi tertentu juga merupakan temuan ilmiah yang bernilai.

---

# RQ2 Status

### Research Question

Faktor apa yang paling memengaruhi prediksi dropout mahasiswa berdasarkan SHAP?

### Current Answer

SHAP berhasil mengidentifikasi fitur paling penting.

Contoh fitur dominan:

* Curricular units 2nd sem approved
* Curricular units 1st sem approved
* Curricular units 2nd sem grade
* Tuition fees up to date
* Scholarship holder

---

# Current Weaknesses

Meskipun SHAP telah menghasilkan ranking fitur yang baik, interpretasi akademisnya masih terbatas.

Saat ini penelitian baru menjawab:

```text
Fitur apa yang penting?
```

Tetapi belum menjawab:

```text
Mengapa fitur tersebut penting?
Apa implikasinya bagi universitas?
Apakah sesuai dengan literatur sebelumnya?
Bagaimana fitur tersebut dapat digunakan untuk intervensi dini?
```

---

# Highest Priority Moving Forward

## Priority 1 — Deep SHAP Interpretation

### Objective

Mengubah hasil SHAP dari sekadar ranking fitur menjadi insight akademis.

---

### Required Tasks

Untuk minimal 10 fitur teratas:

#### Importance

Mengapa fitur tersebut penting?

#### Direction

Pastikan arah pengaruh SHAP benar.

Perlu dilakukan validasi khusus karena terdapat indikasi bahwa beberapa interpretasi arah pengaruh masih perlu diverifikasi.

---

#### Educational Interpretation

Jelaskan hubungan fitur dengan risiko dropout.

Contoh:

```text
Tuition Fees Up To Date
```

Pertanyaan yang perlu dijawab:

* Apakah mahasiswa yang menunggak pembayaran lebih berisiko dropout?
* Apakah faktor ekonomi memiliki pengaruh signifikan terhadap keberlangsungan studi?

---

#### Literature Comparison

Bandingkan dengan penelitian sebelumnya.

Cari:

* Persamaan
* Perbedaan
* Penjelasan kemungkinan penyebab

---

#### Institutional Implications

Jelaskan tindakan yang dapat dilakukan universitas berdasarkan temuan tersebut.

---

### Expected Deliverable

Subsection:

```text
Educational Implications of SHAP Findings
```

Bagian ini berpotensi menjadi kontribusi terkuat penelitian.

---

# Priority 2 — CatBoost Benchmark

## Why

Literatur menunjukkan bahwa CatBoost sering menghasilkan performa yang sangat kompetitif pada dataset pendidikan.

Saat ini belum diketahui apakah:

```text
CatBoost
>
XGBoost
```

atau sebaliknya.

---

## Tasks

Tambahkan:

* CatBoost

Opsional:

* LightGBM

Gunakan:

* Train/Test Split yang sama
* Optuna Tuning
* Cross Validation yang sama
* Evaluasi yang sama

---

## Expected Outcome

Mengetahui apakah baseline terbaik sebenarnya adalah:

* Logistic Regression
* XGBoost
* CatBoost

---

# Priority 3 — Calibration Analysis

## Why

Saat ini evaluasi berfokus pada klasifikasi.

Namun untuk sistem early warning:

```text
Probabilitas
```

sering lebih penting daripada label prediksi.

---

## Tasks

Hitung:

* Brier Score
* Calibration Curve
* Reliability Diagram

Bandingkan:

* Logistic Regression
* XGBoost Baseline
* Proposed Model

---

## Expected Outcome

Menentukan model mana yang menghasilkan probabilitas paling terpercaya.

---

# Priority 4 — Error Analysis

## Why

Confusion Matrix belum cukup menjelaskan perilaku model.

---

## Tasks

Analisis:

### False Positive

Mahasiswa diprediksi dropout tetapi sebenarnya graduate.

### False Negative

Mahasiswa diprediksi graduate tetapi sebenarnya dropout.

---

## Investigate

Cari pola pada:

* Usia
* Beasiswa
* Pembayaran kuliah
* Nilai akademik
* Karakteristik pendaftaran

---

## Expected Outcome

Subsection:

```text
Characteristics of Misclassified Students
```

---

# What Is No Longer a Priority

## Nested Cross Validation

Saat ini bukan prioritas.

Alasan:

* Pipeline sudah leakage-free.
* Cross-validation sudah dilakukan.
* McNemar test sudah dilakukan.
* Ablation Study sudah menjawab pertanyaan yang jauh lebih penting.

Tambahan effort Nested CV kemungkinan tidak memberikan peningkatan nilai ilmiah sebesar SHAP Interpretation atau CatBoost Benchmark.

---

# Final Recommended Roadmap

## Phase 1 (Must Do)

1. Deep SHAP Interpretation
2. Validate SHAP Direction
3. Literature-Based SHAP Discussion

---

## Phase 2 (Strongly Recommended)

4. CatBoost Benchmark
5. Calibration Analysis
6. Error Analysis

---

## Phase 3 (Optional)

7. Nested Cross Validation

---

# Reviewer Verdict

Setelah Ablation Study selesai, kualitas penelitian meningkat secara signifikan.

Sebelumnya penelitian hanya menunjukkan performa model.

Sekarang penelitian berhasil menunjukkan:

* Kontribusi masing-masing komponen pipeline.
* Keterbatasan SMOTE-ENN pada dataset dengan tingkat imbalance moderat.
* Efektivitas XGBoost baseline.
* Faktor-faktor utama yang memengaruhi risiko dropout mahasiswa.

Fokus berikutnya bukan lagi mengejar kenaikan F1 beberapa persen, melainkan memperdalam interpretasi hasil agar menghasilkan kontribusi akademis yang lebih kuat dan lebih bernilai untuk publikasi.
