# Laporan Proyek Machine Learning - Muhamad Alhadid Fadillah

## Domain Proyek

Beton adalah salah satu bahan konstruksi yang paling penting dan banyak digunakan di dunia. Kekuatan tekan beton adalah salah satu sifat mekanis utama yang menentukan daya dukungnya, stabilitas struktural, dan ketahanannya terhadap beban. Kekuatan tekan didefinisikan sebagai kemampuan beton untuk menahan beban tekan tanpa mengalami keruntuhan. Pengujian kekuatan tekan beton dilakukan dengan menggunakan silinder beton standar yang diberi beban hingga hancur, dan nilai kekuatan tekan dihitung dari beban maksimum yang diterima oleh silinder dibagi dengan luas penampang silinder tersebut (Neville, 1996).

Berbagai faktor mempengaruhi kekuatan tekan beton, termasuk rasio air-semen, jenis dan kualitas bahan penyusun (semen, agregat, dan air), serta kondisi curing. Penelitian telah menunjukkan bahwa rasio air-semen yang lebih rendah umumnya menghasilkan kekuatan tekan yang lebih tinggi karena berkurangnya porositas dalam matriks beton (Mehta & Monteiro, 2014). Selain itu, penggunaan aditif dan bahan tambah seperti fly ash, silica fume, dan superplasticizer dapat meningkatkan kekuatan tekan beton dengan memperbaiki struktur mikro dan meningkatkan hidrasi semen (Mindess et al., 2003).

Perkembangan teknologi dalam desain campuran beton dan metode curing juga berkontribusi pada peningkatan kekuatan tekan beton. Misalnya, penggunaan beton berkekuatan tinggi (high-strength concrete) telah menjadi umum dalam konstruksi bangunan tinggi dan struktur jembatan karena kemampuannya untuk menahan beban berat dan meningkatkan efisiensi desain (Feldman, 2008).

Dalam konteks penelitian dan aplikasi praktis, pemahaman yang mendalam tentang faktor-faktor yang mempengaruhi kekuatan tekan beton sangat penting untuk memastikan kualitas dan keamanan struktur beton. Penelitian terus dilakukan untuk mengoptimalkan campuran beton dan metode pengujian, serta untuk mengembangkan beton dengan kekuatan tekan yang lebih tinggi dan ketahanan yang lebih baik terhadap kondisi lingkungan ekstrem (Neville, 2011).

## Business Understanding

### Problem Statements

- Dari ciri-ciri atau karakteristik individu yang telah dikumpulkan, karakteristik apa yang paling berpengaruh terhadap kuat tekan beton (Compressive Strength of Concrete)?
- Berapa kuat tekan beton pada karakteristik atau fitur tertentu?

### Goals

- Mengetahui karakteristik apa yang paling berpengaruh terhadap kuat tekan beton.
- Membuat model machine learning yang dapat memprediksi kuat tekan beton berdasarkan fitur-fitur yang ada.

### Solution statements
- Permasalahan yang harus diselesaikan pada proyek ini adalah permasalahan regresi karena proyek ini bertujuan untuk memprediksi kuat tekan beton berdasarkan karakteristik tertentu.
- Model yang digunakan pada proyek ini adalah K-Nearest Neighbour, Random Forest, dan Adaptive Boosting.
- Metriks yang digunakan untuk memecahkan masalah regresi adalah Mean Squared Error (MSE).

## Data Understanding
[Dataset Kuat Tekan Beton](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).

### Data Loading
![Data Loading](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/data_loading.jpg?raw=true)
Setelah melakukan data loading, dihasilkan bahwa banyaknya jumlah sampel data yaitu 1030 baris dengan 9 kolom atau kategori yang terdiri dari Cement, BlastFurnaceSlag, FlyAsh, Water, Superplasticizer, CoarseAggregate, FineAggregate, Age, dan CompressiveStrength.

### Variabel-Variabel pada Dataset Concrete
Berikut adalah variabel-variabel yang ada pada dataset Concrete:
- Cement: Bubuk mineral yang digiling halus, biasanya berwarna abu-abu. Bahan baku terpenting untuk produksi semen adalah batu kapur, tanah liat, dan napal. Ketika dicampur dengan air, semen berfungsi sebagai perekat untuk mengikat pasir, kerikil, dan batuan keras pada beton.
- BlastFurnaceSlag: Produk sampingan dari produksi besi di tanur sembur, yang diumpankan oleh campuran bijih besi, kokas, dan batu kapur. Dalam prosesnya, bijih besi direduksi menjadi besi sementara semua bahan sisanya membentuk terak, yang disadap sebagai cairan cair dan didinginkan.
- FlyAsh: Residu halus berbentuk tepung yang tercipta saat bubuk batu bara dibakar di pembangkit listrik tenaga batu bara. Material ini terdiri dari partikel berukuran lumpur yang biasanya berbentuk bola dan ukurannya berkisar antara 10–100 mikron. Fly ash sering kali dikumpulkan dari gas buang menggunakan alat pengendap elektrostatis, baghouse, atau alat pengumpul mekanis seperti siklon.
- Water: Air
- Superplasticizer: dikenal sebagai pengurang air kisaran tinggi, adalah bahan tambahan yang digunakan untuk membuat beton berkekuatan tinggi atau untuk menempatkan beton yang dapat memadat sendiri. Plasticizer adalah senyawa kimia yang memungkinkan produksi beton dengan kandungan air sekitar 15% lebih sedikit.
- CoarseAggregate: Agregat kasar merupakan komponen konstruksi yang terbuat dari batuan yang digali dari endapan tanah. Contoh endapan tanah semacam ini termasuk kerikil sungai, batu pecah dari tambang batu, dan beton bekas. Agregat kasar umumnya dikategorikan sebagai batuan yang lebih besar dari standar No.
- FineAggregate: Agregat halus merupakan bahan baku yang terdiri dari partikel pasir alam atau batu pecah yang berukuran lebih kecil dari 5 milimeter, atau dapat lolos saringan 3/8 inci. Agregat halus seringkali berbentuk lebih bulat dibandingkan agregat kasar, yaitu partikel yang berukuran lebih besar dari 5 milimeter dan tidak dapat melewati saringan 3/8 inci.
- Age: Usia beton (dalam satuan hari).
- CompressiveStrength: Variabel target yang akan digunakan, yaitu kekuatan tekan pada beton.

### Exploratory Data Analysis
#### Univariate Analysis
![Univariate Analysis](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/uni1.jpg?raw=true)
Univariate Analysis dapat dilakukan dengan menampilkan visualisasi data histogram untuk fitur numerik. Dari visualisasi histogram, kita bisa mendapatkan beberapa informasi sebagai berikut:
- Usia beton kebanyakan berkisar di 30 hari.
- Rata-rata kuat tekan beton berkisar di 30-40 MPa.
- Cement yang berfungsi sebagai perekat untuk mengikat pasir, kerikil, dan batuan keras pada beton kebanyakan memiliki nilai di kisaran 200-300 kg/m^3.

#### Multivariate Analysis
Multivariate EDA menunjukkan hubungan antara dua atau lebih variabel pada data. Multivariate EDA yang menunjukkan hubungan antara dua variabel biasa disebut sebagai bivariate EDA. Untuk mengamati hubungan antara fitur numerik, kita akan menggunakan fungsi pairplot(). Kita juga akan mengobservasi korelasi antara fitur numerik dengan fitur target menggunakan fungsi corr().
![Multivariate Analysis](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/multi1.jpg?raw=true)

![Multivariate Analysis](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/multi2.jpg?raw=true)

Setelah mengobservasi korelasi antara fitur numerik dengan fitur target, beberapa fitur memiliki korelasi positif terhadap fitur target yang kita gunakan (CompressiveStrength) seperti Superplasticizer, Age, dan Cement. Sedangkan fitur lainnya memiliki korelasi negatif terhadap fitur target.

#### Memeriksa Outliers
![Outliers](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/outlier1.jpg?raw=true)
![Outliers](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/outlier2.jpg?raw=true)
![Outliers](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/outlier3.jpg?raw=true)

Untuk menangani outliers, kita perlu memvisualisasikan tiap fitur apakah terdapat outlier atau tidak dengan menggunakan visualisasi data boxplot. Setelah melakukan visualisasi, dapat kita lihat bahwa terdapat outliers pada beberapa fitur numerik dalam dataset Concrete.

## Data Preparation
Pada bagian ini kita akan melakukan beberapa tahapan persiapan data, yaitu:

### Menangani Missing Value
![Missing Value](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/missing_value.jpg?raw=true)
Untuk menangani missing value, kita dapat mencari tahu variabel atau fitur apa yang ada missing value pada sampel data dengan menggunakan fungsi isnull().sum(). Didapatkan bahwa pada dataset kuat tekan beton, tidak ada missing value pada masing-masing fitur atau variabel.

### Menangani Outliers
Selanjutnya, kita perlu mengatasi outliers ini dengam metode IQR. Kita akan menggunakan metode IQR untuk mengidentifikasi outlier yang berada di luar Q1 dan Q3. Nilai apa pun yang berada di luar batas ini dianggap sebagai outlier. Hal pertama yang perlu kita lakukan adalah membuat batas bawah dan batas atas. Untuk membuat batas bawah, kurangi Q1 dengan 1,5 * IQR. Kemudian, untuk membuat batas atas, tambahkan 1.5 * IQR dengan Q3.
![Outliers](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/outlier4.jpg?raw=true)
![Data Loading](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/outlier5.jpg?raw=true)
Dataset kita sekarang sudah bersih dan memiliki 941 sampel.

### Train-Test Split
Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Kita perlu mempertahankan sebagian data yang ada untuk menguji seberapa baik generalisasi model terhadap data baru. Pada proyek ini kita membagi data training sebesar 80% dari keseluruhan sampel data dan 20% untuk data uji.
![Train Test Split](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/train-test.jpg?raw=true)

### Standarisasi
Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. 

Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita akan menggunakan teknik StandarScaler dari library Scikitlearn, 

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.
![Standar Scaler](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/standarscaler1.jpg?raw=true)
![Standar Scaler](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/standarscaler2.jpg?raw=true)

## Model Development
Model yang akan kita gunakan pada proyek ini adalah K-Nearest Neighbour, Random Forest, dan Boosting.
![Model](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/modeldev.jpg?raw=true)

### K-Nearest Neighbour
KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif).
![KNN](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/KNN.jpg?raw=true)
Kita menggunakan k = 10 tetangga dan metric Euclidean untuk mengukur jarak antara titik. Pada tahap ini kita hanya melatih data training dan menyimpan data testing untuk tahap evaluasi.

### Random Forest
Algoritma random forest adalah salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest juga merupakan algoritma yang sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni. Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning.
![RF](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/RF.jpg?raw=true)
Mula-mula, kita mengimpor RandomForestRegressor dari library scikit-learn. Kita juga sudah mengimpor mean_squared_error sebelumnya sebagai metrik untuk mengevaluasi performa model. Lalu, kita membuat variabel RF dan memanggil RandomForestRegressor dengan beberapa nilai parameter. Berikut adalah parameter-parameter yang digunakan:

- n_estimator: jumlah trees (pohon) di forest. Di sini kita set n_estimator=50.
- max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
- random_state: digunakan untuk mengontrol random number generator yang digunakan. 
- n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.

### Boosting
Algoritma Boosting bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti linear regression dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa. Dilihat dari caranya memperbaiki kesalahan pada model sebelumnya, algoritma boosting terdiri dari dua metode, yaitu Adaptive Boosting dan Gradient Boosting. Pada proyek ini, kita akan menggunakan Adaptive Boosting.
![Boosting](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/Boosting.jpg?raw=true)
Berdasarkan potongan code di atas, berikut adalah parameter-parameter yang digunakan:
- learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting.
- random_state: digunakan untuk mengontrol random number generator yang digunakan.

## Evaluasi Model
Metrik yang biasa  digunakan pada prediction model atau model regresi adalah ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.
![MSE](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/MSE.jpg?raw=true)

Keterangan:
- N = jumlah dataset
- yi = nilai sebenarnya
- y_pred = nilai prediksi

Namun, sebelum menghitung nilai MSE dalam model, kita perlu melakukan proses scaling fitur numerik pada data uji. Sebelumnya, kita baru melakukan proses scaling pada data latih untuk menghindari kebocoran data. Sekarang, setelah model selesai dilatih dengan 3 algoritma, yaitu KNN, Random Forest, dan Adaboost, kita perlu melakukan proses scaling terhadap data uji. Hal ini harus dilakukan agar skala antara data latih dan data uji sama dan kita bisa melakukan evaluasi.
![Scaling](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/scalingeval.jpg?raw=true)
Selanjutnya, kita evaluasi ketiga model kita dengan metrik MSE.
![Evaluation](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/eval1.jpg?raw=true)
![Evaluation](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/eval2.jpg?raw=true)

![Evaluation](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/eval3.jpg?raw=true)

![Evaluation](https://github.com/alhadidmhmd/project_picture/blob/main/Project%20Picture/eval4.jpg?raw=true)

Setelah melakukan evaluasi, dapat disimpulkan bahwa model Random Forest (RF) memberikan nilai eror yang paling kecil. Dengan demikian, model RF akan kita pilih sebagai model terbaik untuk melakukan prediksi kuat tekan beton.

## Kesimpulan
Dengan pemodelan ini, kita sudah bisa menyelesaikan permasalahan yang sudah kita tentukan sebelumnya. Untuk karakteristik atau ciri-ciri yang paling berpengaruh terhadap kuat tekan beton adalah Age (usia beton) dan Cement (perekat untuk mengikat pasir, kerikil, dan batuan keras pada beton) jika diliat dari correlation matrix. Lalu, pemodelan yang bisa digunakan atau bisa dibilang cukup baik untuk memprediksi kuat tekan beton adalah pemodelan Random Forest karena memiliki nilai eror yang paling kecil dan cukup mendekati nilai kuat tekan beton sebenarnya ketika diuji.

## Referensi

- Neville, A. M. (1996). Properties of Concrete. John Wiley & Sons.
- Mehta, P. K., & Monteiro, P. J. M. (2014). Concrete: Microstructure, Properties, and Materials (4th ed.). McGraw-Hill Education.
- Mindess, S., Young, J. F., & Darwin, D. (2003). Concrete (2nd ed.). Prentice Hall.
- Feldman, R. F. (2008). High-Strength Concrete. In Concrete Construction Engineering Handbook (2nd ed.). CRC Press.
- Neville, A. M. (2011). Properties of Concrete (5th ed.). Pearson Education.
