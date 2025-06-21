import pandas as pd
import numpy as np

# --- LANGKAH 1: PERSIAPAN DATA ---
# Berdasarkan studi kasus yang telah kita definisikan sebelumnya.

# a. Daftar Alternatif (Nama Laptop)
alternatives = [
    "Apple MacBook Pro 14 (M3 Pro)", "Apple MacBook Air 15 (M3)", "Dell XPS 15 (Ultra 7)",
    "Dell XPS 13 (Ultra 7)", "Lenovo ThinkPad P1 Gen 6", "Lenovo ThinkPad X1 Carbon G12",
    "Lenovo Yoga Slim 7 Pro (Ryzen 7)", "HP Spectre x360 14", "HP ZBook Firefly 14 G10",
    "Asus Zenbook Duo (2024)", "Asus ROG Zephyrus G14", "Razer Blade 14",
    "Microsoft Surface Laptop Studio 2", "Samsung Galaxy Book4 Pro", "Dell Alienware m16 R2",
    "Apple MacBook Pro 16 (M3 Max)", "Lenovo Legion Slim 5", "HP Omen Transcend 14",
    "Asus ProArt Studiobook 16", "Gigabyte AERO 14 OLED"
]

# b. Nama Kriteria
criteria_names = [
    'Harga (C1)', 'Performa CPU (C2)', 'Kapasitas RAM (C3)',
    'Daya Tahan Baterai (C4)', 'Portabilitas/Bobot (C5)', 'Kualitas Layar/Keyboard (C6)'
]

# c. Matriks Keputusan (X)
# Data fiktif yang realistis, dengan urutan kolom sesuai kriteria di atas:
# [Harga (juta Rp), Skor CPU (Geekbench Multi-Core), RAM (GB), Baterai (Jam), Bobot (kg), Kualitas (skala 1-10)]
decision_matrix = np.array([
    [35.0, 12500, 18, 15, 1.55, 9],  # A1: MacBook Pro 14
    [25.0, 10500, 16, 16, 1.51, 8],  # A2: MacBook Air 15
    [33.0, 11800, 32, 10, 1.90, 9],  # A3: Dell XPS 15
    [26.0, 11500, 16, 12, 1.19, 8],  # A4: Dell XPS 13
    [40.0, 13000, 32, 8,  1.78, 9],  # A5: ThinkPad P1
    [38.0, 12100, 32, 14, 1.09, 10], # A6: ThinkPad X1 Carbon
    [22.0, 11000, 16, 13, 1.45, 7],  # A7: Yoga Slim 7 Pro
    [29.0, 11600, 16, 11, 1.36, 9],  # A8: HP Spectre x360
    [34.0, 12200, 32, 9,  1.40, 8],  # A9: HP ZBook
    [36.0, 13200, 32, 9,  1.65, 9],  # A10: Zenbook Duo
    [30.0, 13500, 16, 10, 1.60, 8],  # A11: ROG Zephyrus G14
    [42.0, 13800, 16, 8,  1.84, 9],  # A12: Razer Blade 14
    [45.0, 13400, 32, 7,  1.98, 9],  # A13: Surface Laptop Studio 2
    [28.0, 11700, 16, 12, 1.23, 8],  # A14: Galaxy Book4 Pro
    [48.0, 14500, 32, 6,  2.50, 8],  # A15: Alienware m16
    [60.0, 15000, 36, 18, 2.10, 10], # A16: MacBook Pro 16
    [24.0, 12800, 16, 9,  2.20, 7],  # A17: Legion Slim 5
    [27.0, 12400, 16, 10, 1.63, 8],  # A18: Omen Transcend 14
    [55.0, 14800, 32, 7,  2.40, 10], # A19: ProArt Studiobook
    [29.5, 12600, 16, 11, 1.49, 9]   # A20: AERO 14 OLED
])

# d. Bobot Kriteria (W) - Total harus 1.0
weights = np.array([0.25, 0.30, 0.20, 0.10, 0.10, 0.05])

# e. Tipe Kriteria ('cost' atau 'benefit')
# 'cost' -> semakin kecil semakin baik
# 'benefit' -> semakin besar semakin baik
criteria_type = ['cost', 'benefit', 'benefit', 'benefit', 'cost', 'benefit']

# Membuat DataFrame untuk menampilkan data awal dengan rapi
df_data = pd.DataFrame(decision_matrix, index=alternatives, columns=criteria_names)
print("--- Matriks Keputusan Awal (X) ---")
print(df_data)
print("\n" + "="*50 + "\n")

# --- LANGKAH 2: NORMALISASI MATRIKS KEPUTUSAN (R) ---

# Membuat matriks kosong untuk menampung hasil normalisasi
normalized_matrix = np.zeros_like(decision_matrix)

# Melakukan iterasi per kolom (kriteria) untuk proses normalisasi
for j in range(len(criteria_names)):
    column = decision_matrix[:, j]
    if criteria_type[j] == 'benefit':
        # Rumus normalisasi untuk kriteria benefit: x_ij / max(x_i)
        max_value = np.max(column)
        normalized_matrix[:, j] = column / max_value
    elif criteria_type[j] == 'cost':
        # Rumus normalisasi untuk kriteria cost: min(x_i) / x_ij
        min_value = np.min(column)
        normalized_matrix[:, j] = min_value / column

df_normalized = pd.DataFrame(normalized_matrix, index=alternatives, columns=[f"{c} (Norm)" for c in criteria_names])
print("--- Matriks Ternormalisasi (R) ---")
print(df_normalized.round(4)) # Dibulatkan 4 desimal untuk kerapian
print("\n" + "="*50 + "\n")


# --- LANGKAH 3: MENGHITUNG NILAI PREFERENSI (V) ---

# Mengalikan matriks ternormalisasi (R) dengan vektor bobot (W)
# V_i = Σ (w_j * r_ij)
scores = np.dot(normalized_matrix, weights)


# --- LANGKAH 4: PERANKINGAN DAN HASIL AKHIR ---

# Membuat DataFrame untuk menampilkan hasil akhir
df_result = pd.DataFrame({
    'Nama Laptop': alternatives,
    'Skor Akhir (V)': scores
})

# Mengurutkan DataFrame dari skor tertinggi ke terendah
df_result_sorted = df_result.sort_values(by='Skor Akhir (V)', ascending=False).reset_index(drop=True)

# Menambahkan kolom 'Peringkat'
df_result_sorted['Peringkat'] = df_result_sorted.index + 1

# Mengatur ulang urutan kolom
df_result_sorted = df_result_sorted[['Peringkat', 'Nama Laptop', 'Skor Akhir (V)']]

print("--- Hasil Akhir Perankingan Laptop Menggunakan Metode SAW ---")
print(df_result_sorted.to_string()) # .to_string() agar semua baris ditampilkan
print("\n" + "="*50 + "\n")

# Menampilkan rekomendasi terbaik
rekomendasi_terbaik = df_result_sorted.iloc[0]
print(">> REKOMENDASI TERBAIK <<")
print(f"Peringkat 1: {rekomendasi_terbaik['Nama Laptop']}")
print(f"Dengan Skor Akhir: {rekomendasi_terbaik['Skor Akhir (V)']:.4f}")

