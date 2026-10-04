# FrustX — kinaz selektivite çalışması için devir belgesi

Bu dosya, web Claude ortamına kopyalanmak üzere yazıldı. İçeriği: (1) FrustX'in ne
hesapladığı ve hangi kararların nedenle sabitlendiği, (2) `frustx/` ve `scripts/`
altındaki her dosyanın ne iş yaptığı, (3) bugüne kadar yürütülen tüm veri
çalışmalarının sonuçları — pozitif ve negatif, (4) COX-1/COX-2 selektivite
çalışmasının tam kaydı ve oradan çıkan tuzaklar, (5) kinaz ailesi için veri seti
üretim planı ve selektiviteyi farklı açılardan inceleme yolları.

Durum tarihi: 2026-09-21. Rakamlar `docs/method.md` ve `results/cox_selectivity/`
içinden alındı; bir sayıyı tekrar kullanmadan önce `docs/method.md` içinde
"CORRECTED / withdrawn / supersedes" araması yapılmalı — o dosya append-only ve
erken bölümlerinin bir kısmı sonradan geri çekildi.

---

## 1. Bilimsel çerçeve: FrustX ne hesaplıyor

Chen et al., *Nat Commun* 11:5944 (2020) protokolünün yeniden uygulanması. Klasik
frustratometer (AWSEM / frustratometeR) kaba-taneli bir enerji fonksiyonu kullanır;
burada **tam-atom Rosetta REF2015** kullanılıyor.

**Eq. 1 (frustrasyon indeksi, temas başına):**

```
F_ij = ( <E_decoy> - E_native ) / sigma(E_decoy)
```

- İşaret, makaleye göre **ters çevrilmiş**; böylece çıktı alanın geri kalanıyla
  (frustratometeR) uyumlu: **yüksek pozitif = minimally frustrated** (iyi, uyumlu
  temas), negatif = highly frustrated.
- Sınıflandırma eşikleri frustratometeR'dan ödünç: `minimally >= 0.78`,
  `highly <= -1.0` (`frustx/config.py`). **Bunlar AWSEM üzerinde kalibre edilmiş,
  REF2015 için doğrulanmadı.** Yüzde-eşleme ile yeniden kalibrasyon denendi ve
  reddedildi (gerekçe `docs/method.md`, "Classification thresholds are borrowed").
- Decoy = diziyi karıştır (shuffle), omurgayı sabit tut, yan zincirleri yeniden
  paketle, gevşet. Yani bu bir **mutational** decoy'dur (kimlik rastgele, geometri
  native) — frustratometeR'ın `configurational` modu ile kıyaslanmamalı.
- `native_reference()` native diziyi **aynı** repack+relax işleminden geçirir. Ham
  girdi yapısının e_ij'si E0 olarak kullanılamaz (geçersiz karşılaştırma).

**Eq. 2 (çok-cisimli fon terimi) cebirsel olarak çöküyor:**

```
E_ij = e_ij + 1/2 sum_{k!=j} e_ik + 1/2 sum_{l!=i} e_jl   ==   1/2 (R_i + R_j)
```

Doğrudan `e_ij` terimi **tam olarak** sadeleşiyor (`R_i = sum_k e_ik`). FrustX bunu
bir ağırlıkla açıyor: `E_ij = e_ij + w * 1/2(R_i+R_j)`, varsayılan **w = 0**.

- `w=1` makalenin literal hali ama indeks dejenere oluyor: 1UBQ'da sıfır
  highly-frustrated temas, ve sinyalin %86-95'i artık temas-özgü değil,
  residü-düzeyi bir nicelik.
- Ligandla birlikte dejenerasyon daha da keskin: bir ligandın tüm protein
  temasları aynı `0.5*B_ligand` terimini paylaşır, yani fon bütün bağlanma
  bölgesine tek bir sabit ekler. Bağımsız bir uygulama (EGFR atomfrust dalı) aynı
  dejenerasyonu R²=1.000000 ile yeniden buldu. **w=0 kalıyor.**

**One-body / specific ayrıştırması (`additivity.py`):** her per-contact indeks
`c + a_i + a_j` biçiminde bir residü-düzeyi kısma (gömülülük, maruz kalma,
paketlenme) ve temasa özgü bir artığa bölünüyor. `contacts.csv` her ikisini de
yazıyor (`frustration_index_onebody`, `frustration_index_specific`), `run.json`
ise `additive_r2`'yi. İkisi de gerçek nicelikler; residü sorusu birinciyi,
belirli bir temas hakkındaki iddia ikinciyi istiyor.

**Kritik kimlik (selektivite işinde kafa karıştırdı, tekrar edecek):**
`frustration_index_specific` bir OLS artığı olduğundan **tek bir residünün tüm
temasları üzerinde tam olarak sıfıra toplanır**. Ligand tek bir residü olduğu için
ligand arayüzünde `mean(specific) = 0` (1.5e-14 ölçüldü) ve
`mean(onebody) = mean(frustration_index)` makine hassasiyetinde. Ligand arayüzünde
sadece **yayılım (std)** bilgi taşır.

**Doğrulama durumu (dürüst hali):** boru hattı uçtan uca çalışıyor, ama
**per-contact doğrulanmadı**. 1UBQ üzerinde frustratometeR'a karşı residü düzeyinde
rho = +0.48, temas düzeyinde +0.32 (mutational mod). Daha derin sorun:
frustratometeR'ın indeksleri büyük ölçüde residü-toplanabilir (%56 configurational,
%95 mutational), FrustX'in %23. Yani **frustratometeR, FrustX'in var olma nedeni
olan per-contact özgüllüğü doğrulayamaz.** Chen makalesi de tek bir per-contact
değer yayımlamıyor — o yol da kapalı.

---

## 2. `frustx/` paketi: hesap zinciri

Bağımlılık sırası önemli:

| dosya | işi |
|---|---|
| `contacts.py` | Geometrik temas haritası: Cα–Cα ≤ 10 Å (ligand temasları için min. ağır-atom mesafesi, varsayılan 6 Å). PyRosetta import etmez; çıplak koordinatlar üzerinde birim-test edilebilir. "Temas" kelimesinin anlamını aşağıdaki her şey için sabitler. |
| `energies.py` | `e_ij`: tek bir poz için ikili REF2015 etkileşim enerjisi. İki bariz olmayan çıkarım adımı taşıyıcı: bb–bb hidrojen bağları varsayılan energy graph'ta **yok** (elle eklenmeli), `rama_prepro` uzun-menzilli bir konteynerde. Ayrıştırmanın tamlığı doğrulandı. |
| `decoys.py` | Eq. 1'in referans topluluğu: diziyi karıştır → yan zincirleri repack et → gevşet. `native_reference()` aynı işlemi karıştırmadan yapar. Ligand varsa donmuş kalır. |
| `frustration.py` | Eq. 1, işaret çevrimi, `E_ij = e_ij + w*½(R_i+R_j)`, sınıflandırma, paralel decoy üretimi. En büyük dosya (724 satır). |
| `additivity.py` | one-body / contact-specific ayrıştırması. |
| `output.py` | `FrustrationResult` → `contacts.csv` (asıl sonuç), `residues.csv` (FrustX'in kendi toplaması; makalenin niceliği değil), B-faktörüne boyanmış `frustration.pdb`. |
| `cli.py` | argparse girişi; her sonucun yanına tam köken bilgisiyle `run.json` yazar. |
| `config.py` | Tüm ayarlanabilir varsayılanlar, her biri "ne denendi, neden bu kazandı" yorumuyla. Kasten import-hafif (numpy/Bio/pyrosetta yok) ki `frustx --help` 2.5 s PyRosetta bedelini ödemesin. |
| `provenance.py` | İçerik-hash anahtarları: bir ara ürün (decoy yapıları, tensörler) hâlâ geçerli mi? Ayarlar aşamaları bağımsız geçersiz kılar (ör. `--readout` skorlamayı geçersiz kılar, decoy yapılarını kılmaz). |
| `ligand_params.py` | Rastgele bir SDF'i Rosetta residü tipine çevirir (PyRosetta'nın kendi SDF okuyucusu `core::chemical::sdf::convert_to_ResidueTypes`), `fa_standard` üzerine katmanlanan bir `PoseResidueTypeSet`'e poz-başına kaydeder. `molfile_to_params.py` gerekmiyor — bu sürpriz olmuştu. |
| `awsem.py` | AWSEM temas enerjisinin bağımsız yeniden uygulaması (frustratometeR'ın LAMMPS kaynağından transkripsiyon, 1e-3'e kadar tam doğrulandı). Sadece "farklı kuvvet alanı" ile "farklı protokol"ü ayırmak için var. |
| `progress.py` | stderr'e ilerleme çubuğu; terminal değilse %10'luk düz satırlar. |

**Önemli CLI bayrakları** (tamamı README'de): `-n/--decoys` (1000), `--protocol`
(`relax` varsayılan / `min` ~10× hızlı), `--cutoff` (10 Å), `--background-weight`
(0), `--readout` (`pair`/`neighbourhood`), `--contact-atom` (CA/CB),
`--packing-frustration`, `-j/--jobs`, ligand tarafı: `--ligand` (yapının içindeki
HETATM'ın kimyasını SDF'den al), `--ligand-placed` (ligand yapıda yok, kendi
dosyasının koordinatlarıyla eklenir), `--ligand-name`, `--ligand-name3`,
`--ligand-cutoff`, `--no-freeze-ligand`.

**Karşılaştırılabilirlik kuralı:** `--protocol`, `n_decoys`, `background_weight`,
`readout`, `cutoff`, `ligand_cutoff`, `contact_atom`, `min_seq_sep` — bunlar Eq. 1
Z-skorunun **ölçeğini** belirler. Bu ayarlarda anlaşmayan iki koşu arasındaki fark
bir sayı değildir. Toplama script'i bu yüzden uyuşmazlıkta hata verip duruyor.

---

## 3. `scripts/` — her script ne yapıyor

`scripts/` paketin parçası değil; `docs/method.md`'deki sonuçları üreten tek
seferlik sürücüler. Gruplayarak:

### 3.1 Hazırlık / yardımcı
- **`prep_structure.py`** — PDB'yi tek protein zincirine indirir. Nedeni: PyRosetta
  MG/GDP/GTP'yi *tanıyor* ve poza residü olarak sokuyor, sonra `ca_coords_from_pose`
  "ResidueType MG does not have an atom CA" ile ölüyor. Dosya düzeyinde atmak poz
  numaralandırmasını 1:1 korur. Ligand sorusu için **yanlış** araç.
- **`renumber_to_reference.py`** — Dizisi birebir aynı olan bir PDB'yi referansa
  göre pozisyonel yeniden numaralandırır. 2RAP'ta Glu123'ün 222 numaralanması ve
  sonraki her residünün bir kayması yüzünden yazıldı; dizi eşleşmezse sert hata
  verir (best-effort hizalama yapmaz). **Kinaz ailesinde bu yaklaşım yetmez** —
  bkz. §5.2(a).
- **`dump_decoy_samples.py`** — Tam per-decoy enerji tensörünü (.npz:
  `decoys (N, n, n)`, `native`, residü etiketleri) diske yazar. Üretim kodu decoy'u
  atar; burada dağılımın **şekli** sorulabiliyor. Eşik tartışmasının tamamı bu
  tensöre dayanıyor.
- **`artefact_payload.py`** — Explainer sayfasının JSON'unu diskteki koşulardan
  üretir; gerçek decoy histogramları çizildiği için kaynak, bitmiş tablo değil
  tensördür.

### 3.2 frustratometeR'a karşı doğrulama
- **`compare_frustratometer.py`** — Aynı yapı üzerinde FrustX vs frustratometeR:
  sayılar değil **örüntü** (hangi temas minimally/highly) ve işaret/eşik aktarımı
  test edilir. `mutational` tablosuna bakılmalı.
- **`compare_singleresidue.py`** — Residü düzeyi. İki FrustX niceliği ayrı ayrı:
  (A) `residues.csv`'deki `mean_frustration` (temas indekslerinin ortalaması — bir
  residü Z-skoru *değil*), (B) tensörden hesaplanan gerçek residü Z-skoru
  `Z_i = (<R_i> - R_i_native)/sd(R_i)`. Fark önemli: rho +0.41 vs +0.48.
- **`compare_forcefields.py`** — 2×2'nin eksik hücresini doldurur: AWSEM enerjisini
  FrustX makinesinden geçirerek **kuvvet alanı** etkisini **protokol** etkisinden
  ayırır. Ayrıca her indeksin ne kadar one-body olduğunu ölçer.
- **`awsem_frustration.py`** — Yukarıdaki hücrenin üreticisi: enerji fonksiyonu
  dışında her şey (temas seti, aynı tohumlu shuffle decoy'lar, Eq. 1) birebir
  REF2015 koşusuyla aynı tutulur.
- **`plot_comparison.py`** — Karşılaştırma figürleri (sınıf oranları vb.).
  `configurational` değil `mutational` tablosuna yöneltilmeli.
- **`threshold_report.py`** — `docs/method.md`'deki eşik tablolarını yeniden üretir;
  yüzde-eşlemenin ne talep ettiğini gösterir. Hiçbir şeye karar vermez.
- **`test_decoy_normality.py`** — Decoy enerji dağılımı bir Z-eşiğinin anlam
  taşıyacağı kadar normal mi? (i) çarpıklık/basıklık + D'Agostino-Pearson K²
  (scipy'sız), (ii) asıl önemli olan **kalibrasyon**: native'i geçen decoy'ların
  ampirik oranı, normal öngörü Φ(-index) ile karşılaştırılıyor.
- **`category_ratios.py`** — Chen'in yayımladığı tek toplu sayı ailesi (arayüz
  minimally oranları %14.2 / %26.0 ve su-aracılı ~%24). Sonuç: sistemler
  isimlendirilmediği için bu iki sayı yeniden üretilemez; sayısız iki iddia ise
  test edilebilir.

### 3.3 Chen Fig. 2 (residü profili) hattı
- **`vicinity_profile.py`** — Makalenin tanımlamadığı "vicinity"yi
  frustratometeR'ın `XAdens()` fonksiyonundan geri kazanır ve **bit-bit** doğrular:
  her temas iki etkileşen atomun orta noktasına yerleştirilir; residü i için, orta
  noktası i'nin CA'sından **kesin küçük** `radius` (varsayılan 5 Å) içinde kalan
  temaslar sayılır. "i'yi içeren temaslar" *değildir* — uzamsal bir yoğunluk.
- **`fig2_profile.py`** — tensör → Eq. 1 → vicinity profili → CSV + grafik. Profil
  **şekli** yayımlanmış figürle karşılaştırılabilir; mutlak sayılar değil (makale ne
  eşiklerini ne yarıçapını veriyor).
- **`plot_fig2.py`** — Fig. 2 sağ kolon stili çizim (yeşil minimally, kırmızı highly).
- **`fig2_significance.py`** — İki kontrol: (1) gürültü tabanı *ölçülerek* —
  tek yapının 500 decoy'u iki yarıya bölünüp iki profil çıkarılır, aralarındaki fark
  saf örnekleme gürültüsüdür; (2) bölge seçimi — P-loop/switch bölgeleri bitişik ve
  profil uzamsal olarak özilintili olduğu için doğru null, aynı uzunlukta kaydırılan
  pencere (etiket permütasyonu değil, o özilintiyi yok edip null'ı hafife alır).
- **`gtpase_replication.py`** — Rheb bulgusunun bağımsız konformer çiftlerinde
  tekrarı: 1KAO/2RAP (Rap2A, dizi birebir ama çözünürlük farkı), 1OIV/1OIW (Rab11A,
  çözünürlük yakın ama 1OIW switch II içinde Q70L taşıyor). Zayıflıklar kasten
  tümleyici seçildi.

### 3.4 Decoy tasarımı ve okuma kapsamı deneyleri (hepsi ana hatta bağlı değil)
- **`sweep_background.py`** — w taramasını **tek** decoy topluluğundan kapalı
  formda yapar (`E_ij(w) = e_ij + w*B_ij` w'de doğrusal olduğundan ilk iki moment
  analitik). Orijinal tarama her w için ayrı 500-decoy koşusu yapmış ve yanlış moda
  (configurational) karşı skorlanmıştı.
- **`eq2_literal.py`** — Literal Eq. 2'yi terim terim doğrular (dejenerasyonun
  varsayım değil ölçüm olması için) ve FrustX'in `w` parametresinin literal formdan
  tam olarak `e_ij` kadar farklı olduğunu gösterir.
- **`readout_scope.py`** — En büyük tespit edilen uyuşmazlık kaynağı. Önceki tüm
  karşılaştırmalar bizim çıplak `e_ij`'imizi onların **komşuluk toplamına** karşı
  koyuyordu (AWSEM mutational decoy enerjisi `water(i,j) + burial_i + burial_j +
  sum_k water(i,k) + sum_k water(j,k)`). Bu script sadece okuma kapsamını değiştirir;
  PyRosetta gerektirmez, diskteki tensörler yeterli.
- **`contact_definition.py`** — CB–CB temas tanımı Cα–Cα'nın kaçırdığını düzeltir mi?
  GTPase koşularında temasların %23.7'sinde decoy sigma ≤ 0.02 (Eq. 1 ~0/0). CB
  temas setini düzeltiyor ama frustratometeR uyumunu iyileştirmiyor → varsayılan CA
  kaldı, bayrak olarak açık.
- **`local_decoy.py`** + **`pair_local_run.py`** + **`pair_local_analysis.py`** +
  **`bench_local_decoy.py`** — Pair-local decoy: sadece temas eden iki residüyü
  rastgeleleştir, sadece 10 Å kabuğu repack et. Uç noktalar koşu uçarken
  **ön-kaydedildi** (rho, additive R², artık rho, AUC) çünkü tek bir dağılım grafiği
  "daha iyi ölçüyor" ile "daha residü-toplanabilir oldu"yu ayırt edemiyor.
  Sonuç: mekanizma ateşlenmedi; eski "ayrımı neredeyse ikiye katlıyor" iddiası bir
  ölçek yapaylığı çıktı. Ana hatta alınmadı.
- **`packer_noise.py`** — Yukarıdaki fikrin geçit kontrolü: decoy enerji
  varyansının ne kadarı indirgenemez paketleyici gürültüsü? Aynı diziyi iki kez
  kurmak farklı rotamer atamaları veriyor. Ölçüm: havuzlanmış **%5.4** paketleyici
  gürültüsü — yani geri kazanılacak gerçek bağlam bilgisi var (ama §yukarıda
  görüldüğü gibi kazanılamadı).
- **`rao_blackwell.py`** — Decoy çekim olasılıkları tam olarak bilindiği için
  (`q(a,b) = n_a n_b / (L(L-1))`) örnek ortalaması yerine tam ağırlıklarla
  post-stratifikasyon. Aynı estimand, daha düşük varyans, yeni Rosetta maliyeti yok.
  Doğru çalıştı ama devreye alınmadı (kazanç maliyetini karşılamıyor).

### 3.5 Ligand hattı
- **`egfr_ligand_check.py`** — Yeni-ligand parametrizasyonunun uçtan uca kontrolü,
  **dört EGFR kinaz domeni kompleksi** üzerinde: 1M17/erlotinib, 2ITY/gefitinib,
  1XKK/lapatinib, 3POZ/TAK-285. Dördü de Rosetta'nın tanımadığı ilaç-benzeri
  ligandlar. Sonuçlar: maksimum koordinat sapması **0.0000 Å**, protein-ligand temas
  sayıları çıplak PDB metninden yapılan bağımsız numpy hesabıyla **birebir** (25/27/37/37),
  donmuş ligand 0.0000 Å oynuyor, dondurma kaldırılınca 1.8–3.3 Å oynuyor (kontrolün
  anlamlı olmasını sağlayan kolon), seri = paralel bit-bit aynı. Bir artık:
  3POZ'da 37 ligand temasından 36'sı sonlu indeks veriyor (`std > 0` koruması, sigma=0
  olan bir temas). Ayrıca çapraz-kontrolün ilk sürümü 1M17'de 25'e 26 anlaşmazlık
  verdi ve **FrustX haklı çıktı**: CYS A 751 iki yarım-doluluk konformerinde
  modellenmiş; altloc filtrelenmeden yapılan kontrol aynı şeyi ölçmüyor.
  **Bu, kinaz işi için zaten elde olan doğrulama.**

### 3.6 COX selektivite hattı (kinaz çalışmasının şablonu)
- **`run_cox_batch.sh`** — Toplu decoy üretimi. 49 inhibitör × 2 hedef = 98 koşu.
  Her hedef için ayrı docking yapıldığından *kimya* paylaşılıyor ama *poz*
  paylaşılmıyor; iki PDB de protein-only olduğundan her koşu `--ligand-placed`
  kullanıyor. Özellikleri: `run.json`'daki `n_decoys` mevcut ayarla eşleşiyorsa
  atlama (kesintiden sonra devam edebilir), koşu başına duvar-saati `timeout`
  (tek bir patolojik PDB-SDF atom eşlemesi 98-koşuluk seriyi kilitlemesin),
  dosya adı normalizasyonu (`cox1/GSK-644784.sdf` ile `cox2/gsk-644784-.sdf` aynı
  bileşik; çıktı dizini normalize edilmiş isimle anahtarlanıyor ki aşağı akıştaki
  eşleştirme çalışsın), yarım kalmış çıktı dizininin geçerli checkpoint sanılmaması
  için `rm -rf`.
- **`cox_selectivity.py`** — Toplama. Her koşunun protein-ligand temaslarını bir
  arayüz betimleyici kümesine indirger, sonra inhibitör başına COX2 − COX1 farkını
  alır. Betimleyiciler: `n_ligand_contacts`, `mean_frustration`, `sum_frustration`,
  `std_frustration_specific`, `n_minimally`, `n_neutral`, `n_highly`,
  `frac_minimally`, `sum_native_energy` (Rosetta'nın kendi arayüz etkileşim enerjisi
  — docking skorunun ev-içi analogu ve her frustrasyon sonucunun yenmesi gereken
  kontrol). Ayarlar uyuşmazsa **reddediyor**. Yarım biten çift için ΔΔF üretmiyor
  (bir sayı değil, eksik değerdir). Etiketler ve docking skoru
  `data/COX_Docking_Selectivity_Scores_Table.xlsx`'ten.
- **`cox_selectivity_stats.py`** — İstatistik ayrı adım. Her betimleyici için:
  rank **AUC** (Mann-Whitney U'nun [0,1]'e ölçeklenmişi, ortalama rank ile),
  20000 çekimli **etiket permütasyon** p-değeri (n=32 vs 17'de normal yaklaşım ince;
  ayrıca scipy bağımlılığı yok), **Benjamini-Hochberg q**. Ve asıl karar veren soru:
  betimleyici **docking skoru regresyonla çıkarıldıktan sonra** da ayırıyor mu?
  İki ayarlama seti ayrı ayrı raporlanıyor (`dock`, `dock + contact count`) —
  temas sayısı tartışmalı bir confound, çünkü COX-2'nin daha büyük cebi gerçekten
  daha çok temas kabul ediyor; bu sinyal de olabilir. `--exclude` ile duyarlılık
  koşusu.
- **`cox_pocket_analysis.py`** — Toplu analizin cevaplayamadığı soru: **nerede?**
  Pozisyon başına iki ayrı test: (i) **presence** — ligand bu pozisyona hiç dokunuyor
  mu (pozisyonlar üzerinden toplandığında bu *zaten* temas sayısıdır, yani bilinen
  etkinin ayrıştırması), (ii) **delta-F** — **iki hedefte de** temas eden ligandlar
  arasında, temas COX-2'de daha mı az frustre? Temas varlığına koşullu olduğu için
  sayıdan bağımsız; `contacts.py`'den elde edilemeyecek tek sonuç bu. Numaralandırma:
  3LN1 resnum + 14 = COX-1 resnum; ofset varsayılmıyor, **sekiz çapa** üzerinde
  doğrulanıyor (513 HIS/ARG ve 523 ILE/VAL dahil), biri düşerse script çıkıyor.
  `SIDE_POCKET = [523, 513, 518, 352, 90]` sonuçlara bakılmadan **önce** ilan edildi,
  43-pozisyonluk taramadan ayrı raporlanıyor (hipotez testi ile tarama karışmasın).
  BH düzeltmesi presence ve delta-F aileleri için **ayrı**.
- **`plot_cox_selectivity.py`** — Beş figür, kasten **negatif sonucu gösterecek**
  şekilde kurulu: `paired_shift.png` (dumbbell — ölçüm eşlenik, eşleniksiz özet bunu
  çöpe atar; COX2−COX1 kaymasına göre sıralı), `delta_by_class.png`, `roc.png`,
  `confound.png`, `auc_ladder.png`. Son ikisi taşıyıcı: ham ayrım (AUC ~0.87) ikna
  edici tek panel olurdu ama kontrolleri geçmiyor. Renkler: mavi = Selective,
  turuncu = Non-Selective (her yerde ve sadece bunu); kontrol aşamaları için ayrı
  tek-ton menekşe rampası (bir renk iki şey kodlamasın). Stereo şüpheli iki bileşik
  atılmıyor, `*` ile işaretleniyor.

---

## 4. Bugüne kadarki tüm veri çalışmaları — sonuç dökümü

### 4.1 Enerji çıkarımı (temel)
REF2015 ayrıştırması tam: bb–bb hidrojen bağları energy graph'a elle eklenerek ve
`rama_prepro` uzun-menzilli konteynerinden okunarak `e_ij` toplamının toplam skora
eşitliği doğrulandı. Ham girdi yapısı E0 olarak kullanılamaz.

### 4.2 1UBQ, frustratometeR'a karşı
- Residü düzeyi: rho = **+0.48** (p 1e-5, n 76) gerçek Z-skoruyla; CLI'nin
  `mean_frustration`'ı ile +0.41.
- Temas düzeyi: rho = **+0.32** (p 1e-9, n 344) `mutational` moda karşı; sadece
  doğrudan temaslarda +0.40.
- `configurational` moda karşı per-contact karşılaştırma **geri çekildi**: o mod
  decoy dağılımını protein başına **bir kez** hesaplıyor, yani indeksi bir per-contact
  Z-skoru değil, yeniden ölçeklenmiş bir enerji. Ayrıca yanlış mod: FrustX'in decoy'u
  mutational.
- Ayrıştırma: kuvvet alanı tek başına rho = +0.18, protokol tek başına +0.27. Kuvvet
  alanı daha büyük fark ve one-body içerik çıkarıldıktan sonra pozitif kalan tek
  karşılaştırma.
- Toplanabilirlik: frustratometeR %56 (configurational) / %95 (mutational) vs FrustX
  %23. O %95 onların **protokolünün** özelliği, AWSEM'in değil — aynı enerji FrustX
  protokolünden geçince %39.
- Bilinen yapaylık: glisin pozisyonları minimally frustrated okuyor.
- σ = 0 temasları cutoff kenarında (`std > 0` koruması hâlâ **açık soru**).

### 4.3 Chen Fig. 2 / küçük GTPase'ler
- "Vicinity" kuralı geri kazanıldı ve bit-bit doğrulandı; profil hattı altı yapıda
  makul profiller üretiyor. **Makine residü çözünürlüğünde doğrulanmış durumda.**
- İlk biyolojik okuma (GTPase fonksiyonel elemanları aktivasyonla frustrasyona
  kayıyor; Rheb'de birleşik kontrast p = 0.010, gürültü tabanının 3.6–4.1 katı)
  **iki ek konformer çiftinde tekrarlanmadı** (havuzlanmış p = 0.86 / 0.19, işaretler
  tutarsız). **İddia geri çekildi**, makine ayakta. Metodolojik ders gerçek sonuç:
  tek çift + marjinal anlamlılık + seçilmiş bölgeler = tekrarlanmayan bulgu.

### 4.4 Decoy tasarımı denemeleri — hepsi negatif ama pahalı öğrenmeler
| deney | sonuç |
|---|---|
| w taraması | w=1 dejenere; w=0 kalıyor. Eski tarama yanlış hedefe karşı yapılmıştı. |
| okuma kapsamı (pair vs neighbourhood) | frustratometeR uyuşmazlığının en büyük tek tanımlı sürücüsü; ama "daha iyi indeks" lisansı vermiyor (daha toplanabilir hale getiriyor). |
| CB temas atomu | ölü ikilileri ayırıyor, benchmark'ı iyileştirmiyor → varsayılan CA. |
| pair-local decoy | mekanizma ateşlenmedi; eski iddia ölçek yapaylığı; devreye alınmadı. |
| Rao-Blackwell | matematiksel olarak doğru, varyans düşüyor, maliyeti karşılamıyor → alınmadı. |
| paketleyici gürültüsü | varyansın sadece %5.4'ü → bağlam bilgisi gerçek. |
| decoy normalliği | Z-eşiğinin olasılıksal okumasını sınayan kalibrasyon testi mevcut. |
| checkpointing | son uzun koşu her şeyini kaybettiği için yazıldı; `provenance.py` aşama-bazlı geçersiz kılıyor. |
| paralel decoy | ~3.2× @ `-j 4`; düşmanca incelemede iki gerçek hata bulundu, kontroller istatistiksel değil tam. |

### 4.5 Ligandlar
- Rastgele SDF'ten parametrizasyon çalışıyor; `molfile_to_params.py` gerekmiyor.
- Kayıt **poz-başına** (`PoseResidueTypeSet`), bu da fork/paralelizasyonu mümkün kılıyor.
- İki indeks var, bir değil: `name` ve `name3`.
- Hidrojenler **zorunlu**: hidrojensiz veya sadece polar-H'li dosya reddediliyor
  (Rosetta atomları bağlantıdan tipler; soyulmuş ligand sessizce yanlış tip/yük alır).
- EGFR dörtlüsünde uçtan uca doğrulama geçti (§3.5).
- Ligand **seyirci**: temas haritasına giriyor, asla mutasyona uğramıyor / repack
  edilmiyor / hareket etmiyor. Ölçülen donma sapması 0.0000 Å, dondurma kalkınca
  1.8–3.3 Å.
- Kopyalanmaması gerekenler (EGFR dalından): kristal-params yolunda H ve formal yük
  işlemesi (lapatinib'in sekonder amini `Nhis` tipleniyor — kimyaca donör, orada
  akseptör; CCD formal yük kolonu kimse tarafından okunmuyor → yüklü inhibitör nötr
  parametrize ediliyor). İkisi de sessiz ve tam ölçmek istediğimiz temasların
  elektrostatiğini değiştiriyor.
- Hiçbir şeye temas etmeyen bir ligand, özelliğin kendisi kullanılırken bulundu
  (`--ligand-placed` yanlış kareden koordinat aldığında; RCSB `_ideal.sdf` orijin
  yakınında üretilmiş bir konformer, docking pozu değil).

### 4.6 COX-1/COX-2 selektivite — tam kayıt

**Veri seti.** 49 NSAID/coxib, her biri 1EQG (COX-1) ve 3LN1 (COX-2) içine ayrı ayrı
docking'lenmiş (MolModa). Etiketler `data/COX_Docking_Selectivity_Scores_Table.xlsx`:
**32 Selective / 17 Non-Selective**, ayrıca `COX1_Best_Score_kcal_mol`,
`COX2_Best_Score_kcal_mol`, `Selectivity (COX2_minus_COX1)`. Hidrojenler önceden
Open Babel 3.1.0 ile: `obabel <lig>.sdf -osdf -O <lig>_withH.sdf -h -p 7.4`.
**Uyarı:** iki bayrak birlikte verildiğinde Open Babel `-p`'yi sessizce yok sayıyor,
yani bu ligandlar **nötr** (karboksilik asitler protonlu). `-p 7.4` *tek başına*
deprotone ediyor ve `M CHG` yazıyor — ibuprofen üzerinde doğrulandı (nötr C13H18O2 vs
−1 yükte C13H17O2). 49 ligandın hiçbiri formal yük taşımıyor.

**Pilot koşular.** `results/1eqg_A_min20/`, `results/3ln1_A_min20/` (ligandsız,
`--protocol min -n 20`, temas haritası ve boru hattı kontrolü) ve
`results/pilot_*` (tek ligandlı pilotlar: celecoxib/COX-1, bromfenac, bms-347070 üç
farklı yükleme yoluyla, l-804600'ün bozuk ve düzeltilmiş halleri). Bunlar
ölçeklemeden önce yolun doğrulandığı yer.

**Deneme 1 (12–15 Eylül, terk edildi).** Docking'lenmiş **kompleks PDB**'ler
(`data/docking/COX-1|COX-2/<ligand>.pdb`) üzerinde `--ligand` (topology) modu,
`--protocol min`, 8 shard. 98'in ancak 46'sı bitti. İki ölümcül sorun:
1. Docking programları her hit'i `UNL`/`LIG` yazıyor; bu yer-tutucu kodlar Rosetta'nın
   PDB bileşen aramasında özel işleniyor ve özel poz residü tipi eşleşmeden **düşüyor**.
2. Atom adları çıplak element sembolü olarak yazılıyor (cis-stilbene'in 14 karbonu da
   "C"), Rosetta da HETATM↔residü-tipi eşleşmesini sadece geometriden kurmak zorunda
   kalıyor (`remap_pdb_atom_names`) ve dejenerasyon o aramaya tutunacak hiçbir şey
   bırakmıyor. 98 girdide ölçülen maliyet: 4-6 farklı ad → ~1.5 s; 3 farklı ad →
   `RuntimeError: too many tries in fill_missing_atoms!`; 1-2 farklı ad → hiç bitmiyor,
   1800 s'de öldürüldü. **98 girdinin hepsi yinelenen ad taşıyordu — docking çıktısı
   için bu normal durum, kenar durum değil.**

   Düzeltme (şu an `frustx/ligand_params.py`'de **commit'lenmemiş**): topology modundaki
   `UNL`/`LIG` ligandlara bellek-içi PDB'de ve kayıtlı tipte özel çakışmasız kod
   (`X01`, `X02`…) verilmesi; `uniquify_ligand_atom_names()` ile her HETATM atomuna
   element+sayaç ile tek ad verilmesi (kopya başına sayaç sıfırlanıyor, element kolonu
   boşsa addeki alfabetik kısma düşülüyor); ve `pose_from_pdbstring` kullanımı
   (normal init CCD yüklemesini kapattığı için HETATM'lar düşüyordu). 5 yeni birim testi
   eklendi. **Bu iş `docs/method.md`'de henüz belgelenmedi.**

**Deneme 2 (16–18 Eylül, analiz edilen koşu).** Protein-only PDB + hedefe özgü
docking pozu SDF'i → `--ligand-placed`, `--protocol relax`, `-n 1000`, `w=0`,
`--ligand-cutoff 6.0`, ligand donmuş. **98/98 koşu tamam.** Çıktı:
`data/docking/cox_results/decoy_results/{cox1,cox2}_<ligand>/`. Tüm koşuların
ölçek-belirleyen 8 ayarda anlaştığı doğrulandı.

**Veri hijyeni — ilk yapılması gereken kontrol.** 49 inhibitörden **üçü iki hedefte
kimyasal olarak farklı moleküllerdi**, çünkü MolModa COX-1 pozu için geçersiz
bağlantı yazmış:

```
l-804600  cox1 C21H24N2O4S vs cox2 C21H22N2O4S  sülfonil kopmuş (S-H@1.36, tek O, O-O@1.24)
l-768277  cox1 C17H13N3O2S2 vs cox2 C17H16N2O2S2  bir azot kayıp
sc-58125  cox1 C17H14F4N2O2S vs cox2 C17H12F4N2O2S  pirazol aromatikliği kayıp
```

Sadece `l-804600` **yüksek sesle** çöktü (`Cannot reroot a disconnected ResidueType`).
Diğer ikisi **sonuna kadar koştu ve makul sayılar üretti** — iki hedefte iki farklı
molekülü karşılaştırırken. Çökme şanslı durum. Üçünü de yakalayan ucuz kontrol:
bir inhibitörün cox1 ve cox2 dosyaları **aynı canonical SMILES**'ı vermek zorunda.
Formül eşitliği daha zayıf (bağ-derecesi ve stereo kusurlarını kaçırır); kopuk parça
taraması tek başına `sc-58125`'i kaçırır. İki çift (`pd-138387`, `sulindac-sulfide`)
sadece stereo/bağ-derecesi algısında farklı; taşınıyor ve `--exclude` duyarlılık
koşusu var.

**Toplu sonuç** (`results/cox_selectivity/stats.csv`, 21 Eylül'de yeniden üretilen
hali; `docs/method.md`'deki tablo bir önceki üretimden ve ondalıklarda birkaç binde
fark taşıyor).

```
n = 49 (32 Selective / 17 Non-Selective)
BASELINE  docking ddG                          AUC 0.823  p 0.0001

betimleyici           ham AUC   -dock              -dock-contacts
d_n_minimally           0.862   0.716 (p 0.015)    0.611 (p 0.224)
d_n_ligand_contacts     0.856   0.702 (p 0.023)    --
d_sum_frustration       0.853   0.716 (p 0.015)    0.639 (p 0.122)
d_std_frustration_spec  0.829   0.663 (p 0.072)    0.637 (p 0.131)
d_mean_frustration      0.825   0.688 (p 0.035)    0.631 (p 0.147)
d_frac_minimally        0.823   0.690 (p 0.036)    0.605 (p 0.247)
d_n_highly              0.736   0.635 (p 0.137)    0.569 (p 0.451)
d_sum_native_energy     0.159   0.329 (p 0.058)    0.435 (p 0.487)
d_n_neutral             0.372   0.349 (p 0.093)    0.331 (p 0.058)
```

Yön hipotezle uyumlu (selektif bileşikler COX-2 tarafında ortalama +3.8 minimally
frustrated temas kazanıyor, non-selektifler −2.1 kaybediyor) ve ham ayrım güçlü.
**Kontrolleri geçmiyor.** Docking ΔΔG zaten 0.82'lik bir yordayıcı; hem o hem ligand
temas sayısı regresyonla çıkarıldığında her betimleyici 0.61–0.64'e düşüyor ve hiçbiri
anlamlı değil (q ≥ 0.12). Stereo-uyuşmaz iki çift atıldığında n=47'de aynı tablo.

Belirleyici karşılaştırma: `d_n_ligand_contacts` **tek başına** ham 0.856, docking
ayarlamasından sonra 0.702 — herhangi bir frustrasyon betimleyicisi kadar iyi. Yani tüm
sinyal *COX-2-selektif inhibitörler COX-2'de COX-1'den daha fazla protein teması
yapıyor* ifadesine indirgeniyor; bu daha büyük yan cep hakkında bir cümledir ve tek bir
decoy üretmeden `contacts.py` ile elde edilir. **98 × 1000 REF2015 decoy ölçülebilir
hiçbir şey eklemiyor.**

Bu, **bir ligand arayüzü üzerinde toplanmış frustrasyon** hakkında negatif bir sonuçtur;
selektivite hakkında değil. Toplama, hangi temasın değiştiğini atıyor — makalenin birimi
ise temastır.

Arayüz betimleyicilerinin çıplak dağılımı (per_run_interface.csv): ligand teması
COX-1'de ortalama 22.8 (16–33), COX-2'de 28.0 (22–35); `mean_frustration` 0.43 vs 0.46
ama COX-1'in yayılımı iki kat geniş (sd 0.34 vs 0.14); `sum_native_energy` −25.4 vs
−32.7 REU.

**Residü çözünürlüklü analiz (21 Eylül, `pocket_positions.csv` — `docs/method.md`'ye
henüz yazılmadı).** 43 pozisyon iki hedefte de temas ediliyor, 21'i delta-F testi için
yeterli eşlenik liganda sahip.

Ön-kaydedilmiş yan cep:
```
pos  cox1/cox2  n1  n2  presence_auc q   n_paired dF_sel  dF_non  dF_auc  dF_p   dF_q   dF_adj_auc adj_p
523  ILE/VAL    41  49  0.580 (0.319)    41       +0.903  -0.024  0.807   0.001  0.010  0.623      0.196
518  PHE/PHE    22  49  0.832 (0.000)    22       -1.512  -0.706  0.181   0.018  0.093  0.248      0.066
352  LEU/LEU    22  49  0.832 (0.000)    22       +0.841  +0.920  0.486   0.918  0.918  0.514      0.946
513  HIS/ARG     5  40  0.642 (0.088)     5       (yeterli eşlenik ligand yok)
```
Taramanın en güçlüsü: **Tyr355** (49/49 ligandda temas), dF_sel +1.976 vs
dF_non −1.124, dF_auc 0.903, ham p < 0.001, q 0.001; docking+temas sayısı çıkarıldıktan
sonra 0.676 (p 0.044, ama tarama q'su 0.62).

Okuma: **Val523 (COX-1'de Ile523) ve Tyr355'te, iki hedefte de temas eden ligandlar
arasında, selektif bileşiklerin teması COX-2'de belirgin şekilde daha az frustre**;
bu, temas sayısına koşullu **olmayan** bir gözlem, yani toplu analizin öldürdüğü sinyalin
yaşadığı yer. Ancak ayarlanmış q değerleri tarama genelinde anlamlılığa ulaşmıyor
(n_paired 41 ve 49 — güç sınırlı). Bu, kinaz veri setinin **asıl hedefi** olmalı:
aynı testi çok daha büyük bir n ile yapmak.

---

## 5. Kinaz veri seti: ne kopyalanır, ne değişmeli

### 5.1 Doğrudan kopyalanabilir olanlar
- `run_cox_batch.sh` iskeleti: resume, timeout, isim normalizasyonu, shard mantığı.
- `cox_selectivity.py`'nin arayüz betimleyici indirgemesi ve **ayar uyuşmazlığında
  reddetme** davranışı.
- `cox_selectivity_stats.py`'nin AUC + permütasyon + BH + **kovaryans ayarlaması**
  iskeleti. Bu dosyanın asıl değeri istatistik değil, *hangi kontrolün zorunlu
  olduğunu* kodlaması: docking skoru ve temas sayısı çıkarılmadan hiçbir sayı
  rapor edilmiyor.
- `cox_pocket_analysis.py`'nin presence / delta-F ikiliği ve ön-kayıt disiplini.
- Ligand tarafı: EGFR dörtlüsünde zaten doğrulanmış (kinaz domeni!), yani
  parametrizasyon riski COX'a göre daha düşük.

### 5.2 Kinazda kaçınılmaz olarak farklı olan üç şey

**(a) Hizalama artık bir sabit ofset değil.** COX'ta 3LN1 + 14 = 1EQG yetiyordu ve
sekiz çapa ile doğrulanıyordu. Kinaz ailesinde iki farklı kinaz arasında böyle bir
ofset yok. Çözüm: **KLIFS'in 85-residülük cep numaralandırması** (veya Pfam/HMM
kinaz profili ile hizalama) kanonik ortak eksen olarak kullanılmalı — hinge,
gatekeeper, DFG motifi, β3-lizin, αC-glutamat, katalitik HRD hepsi bu eksende sabit
indeks. `cox_pocket_analysis.py`'deki `OFFSET`/`ANCHORS` mekanizması, "çapa
doğrulaması" fikri korunarak KLIFS indeks eşlemesine dönüştürülmeli (çapalar:
DFG-Asp, gatekeeper, K of VAIK, E of αC, HRD-Asp). Ofseti varsayma, doğrula, düşerse
çık — bu desen kalmalı.

**(b) İki hedef değil, bir panel.** COX bir *çift* karşılaştırmasıydı (ΔΔF = COX2 −
COX1). Kinazda doğru yapı bir **ligand × kinaz matrisi**. Bu, analizi eşlenik farktan
şunlara taşıyor: ligand başına hedefler arasında **profil** (vektör), ve selektivite
metriği olarak Gini/entropi ya da "en iyi hedefe göre ΔΔ". Deneysel etiket olarak
ikili Selective/Non-Selective yerine sürekli bir nicelik kullanılabilir (aşağıda).

**(c) Konformasyon selektivite değişkeni.** Kinazlarda selektivite büyük ölçüde
*konformasyondan* geliyor (DFG-in/out, αC-in/out, tip I / I½ / II / III allosterik).
Aynı kinazın iki konformerine aynı ligandı yerleştirmek, COX çalışmasında hiç
olmayan bir eksen açıyor ve FrustX'in gerçekten iyi olabileceği yer burası: frustrasyon
"bu temas bu yapıda ne kadar uyumlu" sorusunu ölçüyor, ve tip II inhibitörlerinin
DFG-out cebindeki temasları tam olarak bu tür bir sorunun konusudur.

### 5.3 Deneysel etiket kaynakları (COX'taki docking-skoru bağımlılığından çıkmak için)
COX çalışmasının en zayıf yeri, "selektivite" etiketinin yanında **aynı docking
koşusundan gelen bir skor** olması ve o skorun tek başına 0.82 AUC vermesiydi.
Kinazda ölçülmüş, docking'den bağımsız afinite panelleri var: Davis et al. 2011
(Kd, 72 inhibitör × 442 kinaz), Metz et al. 2011 (pKi), Karaman et al. 2008,
ChEMBL/BindingDB, ve PKIS setleri. Bunları kullanmak iki şeyi birden çözüyor:
(i) etiket docking'den bağımsız olur, (ii) ikili sınıf yerine sürekli ΔpKd
hedefi olur, yani AUC yerine korelasyon ve kısmi korelasyon raporlanabilir.
**Zorunlu kontrol aynı kalıyor:** docking skoru ve temas sayısı çıkarıldıktan sonra
frustrasyon hâlâ bir şey söylüyor mu?

### 5.4 Hesap bütçesi (COX'tan ölçülmüş)
- COX: 98 kompleks × 1000 decoy, `--protocol relax`, 8 shard × 4 worker →
  tahmin 8.8 saat (gerçekleşen: 12–18 Eylül aralığında, birkaç yeniden deneme ile).
- `--protocol min` ~10× hızlı ama **Eq. 1'in denominatörünü değiştirir**, yani min ile
  relax sonuçları karşılaştırılamaz. Pilot/iterasyon için min, nihai tablo için relax
  — ve ikisi asla aynı tabloda toplanmaz (`cox_selectivity.py` bunu reddediyor).
- Kinaz matrisi hızla büyüyor: 20 kinaz × 30 ligand = 600 kompleks ≈ COX'un 6 katı
  ≈ 2-3 gün. **Önce 4 kinaz × 10 ligand ile pilot** (min protokolü, 200 decoy),
  hizalama ve hijyen kontrollerini orada doğrula, sonra ölçekle.

### 5.5 Kinaza özgü tuzaklar (COX'ta öğrenilmişlerin uzantısı)
1. **Canonical SMILES kontrolü ilk adım olmalı.** Kinazda aynı ligand N hedefe
   docking'leniyor → N dosya → N kez bozulma şansı. Kontrol: bir ligandın tüm hedef
   dosyaları aynı canonical SMILES vermeli. Sessiz başarısızlık burada COX'tan daha
   olasıdır.
2. **Protonasyon.** COX'ta `obabel -h -p 7.4` sessizce nötr ligand üretti. Kinaz
   inhibitörlerinin çoğu bazik amin taşıyor (imatinib'in piperazini, pek çok tip I'in
   morfolini) ve pH 7.4'te **yüklüdür**. Bu bir tercihtir ve **bilinçli** yapılmalı:
   `-p 7.4` tek başına kullanılırsa yükler oluşur; o zaman Rosetta'nın o yükü doğru
   tipleyip tiplemediği kontrol edilmeli (EGFR dalında formal yük kolonunun kimse
   tarafından okunmaması kaydı var).
3. **Kofaktör: ATP/ADP ve Mg²⁺.** Rosetta bunları zaten tanıyor (bayrak gerekmez),
   ama `prep_structure.py` onları atıyor. Kinaz için karar verilmeli: ATP sahası
   inhibitörünün komşusu olarak Mg²⁺ bırakılacak mı? Bırakılırsa temas haritasına
   girer ve ΔΔF'i etkiler; atılırsa cebin elektrostatiği eksik kalır. Hangisi olursa
   **tüm koşularda aynı** olmak zorunda.
4. **Fosforilasyon durumu.** Aktivasyon halkası fosforlu/fosforsuz yapılar karışırsa
   protein-tarafı frustrasyon sistematik olarak kayar. Yapı seçiminde kaydedilmeli.
5. **Eksik aktivasyon halkası / eksik loop'lar.** Kinaz kristal yapılarında sık; 10 Å
   temas haritası eksik residüleri sessizce atlar ve `n_contacts` hedefler arasında
   karşılaştırılamaz hale gelir. Hedef başına eksik-residü raporu tutulmalı.
6. **Yinelenen atom adları / UNL kodu.** Kompleks PDB (topology modu) ile
   çalışılacaksa `ligand_params.py`'deki commit'lenmemiş düzeltmeler **şart**.
   `--ligand-placed` yolu bu sorunu hiç görmüyor — COX'ta çalışan yol buydu, kinazda
   da varsayılan bu olmalı.
7. **σ = 0 temasları** (`std > 0` koruması) cutoff kenarında sonlu olmayan indeks
   üretiyor; 3POZ'da 37 ligand temasından 36'sı sonlu çıktı. Ligand arayüzünde bu
   kayıp sayılmalı, sessizce düşürülmemeli.

---

## 6. Selektiviteyi farklı açılardan inceleme yolları

Her madde: **soru → gereken veri → hangi mevcut script'in genellemesi → confound.**
Sıralama, COX sonucundan sonra beklenen getiri sırasına göre.

### 6.1 Residü çözünürlüklü delta-F (en yüksek öncelik)
- **Soru:** İki hedefte de temas eden ligandlar arasında, belirli cep pozisyonlarının
  teması bir hedefte sistematik olarak daha mı az frustre?
- **Neden:** COX'ta toplu sinyal öldü ama pozisyon çözünürlüğünde Val523 ve Tyr355
  ham p ≤ 0.001 verdi. Bu, temas sayısına **koşullu** bir testtir, yani
  `contacts.py`'den elde edilemez — FrustX'in ekleyebileceği tek şey bu.
- **Veri:** ligand × kinaz matrisi + KLIFS pozisyon eşlemesi.
- **Genelleme:** `cox_pocket_analysis.py` → `kinase_pocket_analysis.py`, ofset yerine
  KLIFS indeksi, ikili etiket yerine sürekli ΔpKd, ön-kayıtlı pozisyon listesi
  (gatekeeper, DFG-Asp/Phe, hinge, β3-Lys, αC-Glu, katalitik HRD).
- **Confound:** docking skoru + temas sayısı mutlaka çıkarılmalı; BH tarama genelinde;
  ön-kayıtlı liste taramadan ayrı raporlanmalı.

### 6.2 Protein tarafı: apo vs holo frustrasyon farkı
- **Soru:** Ligand bağlanması cep residülerinin **protein-protein** temaslarının
  frustrasyonunu nasıl değiştiriyor? Selektif ligandlar cebi "rahatlatıyor" mu?
- **Neden:** Şimdiye kadar hep ligand arayüzü toplandı. Protein-protein tarafı tamamen
  el değmemiş ve orada makalenin kendi niceliği (residü profili) zaten doğrulanmış
  durumda (`vicinity_profile.py` bit-bit).
- **Veri:** aynı yapının apo ve holo koşusu (ligandlı/ligandsız aynı PDB).
- **Genelleme:** `fig2_profile.py` + `vicinity_profile.py`, hedef başına ΔF profili.
- **Confound:** Ligandın varlığı temas haritasını değiştirir; apo koşusunda ligandı
  atmak yeterli, yapı aynı kalmalı (aynı PDB, `--ligand` bayrağı var/yok). Decoy
  gürültü tabanı `fig2_significance.py`'nin yarı-bölme kontrolüyle ölçülmeli.

### 6.3 Direnç mutasyonları (kinazın en güçlü yanı)
- **Soru:** Bilinen direnç mutasyonu (EGFR T790M, ABL T315I, ALK L1196M, KIT D816V)
  bir inhibitörün temaslarının frustrasyonunu nasıl değiştiriyor? Duyarlı/dirençli
  inhibitör çiftleri ayrışıyor mu?
- **Neden:** Burada etiket son derece net (klinik direnç), docking skorundan bağımsız,
  ve karşılaştırma **aynı** ligand + **neredeyse aynı** protein — yani COX'u öldüren
  "temas sayısı farkı" confound'u minimum. Bu, FrustX için şimdiye kadar tasarlanmış
  en temiz test olabilir.
- **Veri:** WT ve mutant yapı (deneysel ya da tek noktalı Rosetta mutasyonu), aynı
  ligand pozu.
- **Genelleme:** `cox_selectivity.py`'nin ΔΔF çerçevesi, hedef ekseni WT/mutant.
- **Confound:** Mutant yapıyı modelle üretirsen, modelleme yapaylığı frustrasyonla
  karışır → deneysel yapı varsa o tercih edilmeli; yoksa aynı modelleme protokolü
  WT'ye de uygulanmalı (kendi kontrolü).

### 6.4 Konformasyon selektivitesi (DFG-in / DFG-out)
- **Soru:** Tip II inhibitörlerinin DFG-out cebindeki temasları, aynı ligand DFG-in
  yapısına yerleştirildiğinde daha mı frustre? Frustrasyon "bu ligand bu konformasyona
  ait mi" sorusunu ölçebilir mi?
- **Veri:** aynı kinazın DFG-in ve DFG-out yapıları (ör. ABL 2HYY/1IEP vs 3KFA tipi),
  tip I ve tip II ligand karışımı.
- **Genelleme:** hedef ekseni = konformasyon; `cox_pocket_analysis.py` doğrudan uyar.
- **Confound:** İki konformer farklı yapılardan geliyorsa çözünürlük/eksik loop farkı
  girer — GTPase çalışmasında tam bu confound bulguyu geri çektirdi. **Aynı hatayı
  tekrarlamamak için** en az iki bağımsız kinazda tekrarlanmalı (tümleyici
  zayıflıklarla, `gtpase_replication.py`'nin tasarım mantığı).

### 6.5 Panel-genişliği / promiskuite
- **Soru:** Bir inhibitörün panel genelindeki ortalama arayüz frustrasyonu, onun
  **seçiciliğini** (Gini/entropi) yorduyor mu? Yani promiskü bileşikler her yerde
  "orta derecede uyumlu", selektif olanlar tek hedefte çok uyumlu mu?
- **Veri:** ligand × kinaz matrisi + ölçülmüş Kd paneli.
- **Yeni metrik:** ligand başına frustrasyon profilinin yayılımı; sadece en iyi hedef
  değil, dağılımın şekli.
- **Confound:** Molekül büyüklüğü. Büyük ligand her yerde daha çok temas yapar. Ağır
  atom sayısı ve temas sayısı ikisi de regresyonla çıkarılmalı.

### 6.6 Temas-sınıfı kompozisyonu, toplam yerine
- **Soru:** Toplam frustrasyon değil, arayüzün **kompozisyonu**: kaç minimally, kaç
  highly, ve bunlar hangi *kimyasal* temas tipinde (H-bağı / hidrofobik / aromatik /
  halojen)?
- **Neden:** COX'ta `d_n_minimally` ham AUC'da en iyi betimleyiciydi (0.862) — sayım
  ortalamadan daha bilgili çıktı. Bunu temas tipine göre kırmak yeni bir eksen.
- **Veri:** mevcut `contacts.csv` + REF2015 terim ayrıştırması (`energies.py` bunu
  zaten yapabiliyor).
- **Confound:** Eşiklerin REF2015 için kalibre edilmemiş olması; sayımlar eşiğe
  duyarlı. Eşik duyarlılık taraması (0.5 / 0.78 / 1.0) raporlanmalı.

### 6.7 `specific` bileşeni doğru birimde
- **Soru:** Ligand arayüzünde `mean(specific) = 0` olduğu için (yapısal kimlik),
  temas-özgü bilgi sadece **yayılım** ve **protein tarafındaki** residülerde okunabilir.
  Bir cep residüsünün *kendi* temasları üzerinden specific artığı ne söylüyor?
- **Neden:** COX'ta `d_std_frustration_specific` ham 0.829 verdi — yani yayılımda
  gerçekten bilgi var, ortalamada olamaz.
- **Genelleme:** ayrıştırmayı ligand değil **residü** ekseninde topla.

### 6.8 Su-aracılı ve elektrostatik temaslar
- **Soru:** Kinaz cebinde korunmuş su köprüleri selektiviteyi taşıyor; frustrasyon bunu
  görüyor mu?
- **Durum:** `docs/method.md` kaydı net — su-aracılı karşılaştırma frustratometeR'a
  karşı **yapılamaz** (numeratörler farklı nicelikler) ve elektrostatik de
  karşılaştırılabilir değil. Yani bu ancak FrustX-içi bir soru olarak, açık su
  molekülleri poza dahil edilerek kurulabilir. **Yeni iş, ucuz değil.**

### 6.9 Baz çizgisi yarışması (yayın için zorunlu)
- **Soru:** Frustrasyon, ucuz baz çizgilerini yeniyor mu? Baz çizgileri: docking skoru,
  temas sayısı, gömülü yüzey alanı, `sum_native_energy` (Rosetta arayüz enerjisi),
  ligand ağır atom sayısı.
- **Neden:** COX'ta `sum_native_energy` ters yönde AUC 0.159 verdi (yani 0.841 ters
  okunuşla) — Rosetta'nın çıplak etkileşim enerjisi de sinyal taşıyordu. Bir
  frustrasyon iddiası bu listeyi geçmek zorunda.
- **Genelleme:** `cox_selectivity_stats.py`'nin kovaryans çerçevesi; her baz çizgisi
  ayrı ayrı ve birlikte çıkarılmalı.

---

## 7. Koda eklenmesi önerilenler (somut, sırayla)

1. **`scripts/check_ligand_identity.py`** — bir ligandın tüm hedef dosyalarının aynı
   canonical SMILES'i verdiğini doğrula; kopuk parça, formal yük ve H sayısı raporu.
   **Batch'ten önce koşan kapı.** COX'ta bu kontrol olmadığı için iki inhibitör
   yanlış karşılaştırıldı ve fark edilmedi.
2. **`scripts/kinase_align.py`** — KLIFS (veya HMM) cep numaralandırmasını her hedef
   PDB'ye eşle, çapa residülerinde doğrula, düşerse çık. `cox_pocket_analysis.py`
   içindeki `OFFSET`/`ANCHORS` mantığının aile-ölçeğine genellenmesi.
3. **`scripts/selectivity_collate.py`** — `cox_selectivity.py`'nin N-hedef genellemesi:
   çift farkı yerine ligand × hedef uzun tablo, ayar uyuşmazlığı reddi korunarak.
4. **`scripts/selectivity_stats.py`** — `cox_selectivity_stats.py`'nin sürekli hedefe
   (ΔpKd) genellemesi: AUC yanında Spearman ve kısmi Spearman, aynı permütasyon + BH
   disiplini, baz çizgisi merdiveni.
5. **`scripts/run_kinase_batch.sh`** — `run_cox_batch.sh`'nin N-hedef hali; hedef
   listesi ve ligand listesi ayrı dosyalardan, shard mantığı korunarak.
6. **`ligand_params.py` düzeltmelerini commit'le** ve `docs/method.md`'ye yaz —
   şu an çalışan bir düzeltme belgesiz duruyor; kompleks-PDB yolu kinazda gerekirse
   bu bilgi kaybolmamalı.
7. **Residü-ekseni ayrıştırma yardımcısı** (`additivity.py` üzerine): bir residünün
   kendi temasları üzerinden specific yayılımı — §6.7 için gereken nicelik.

---

## 8. Kırmızı çizgiler ve dürüstlük notları

- **Per-contact indeks doğrulanmadı** ve doğrulanacak bir referans yok (frustratometeR
  %95 toplanabilir, Chen per-contact değer yayımlamıyor). Kinaz çalışması bir
  *uygulama*; aynı zamanda doğrulama olarak sunulamaz.
- **Eşikler REF2015 için kalibre değil.** `minimally/highly` sayımları içeren her
  sonuç eşik duyarlılığı ile birlikte raporlanmalı.
- **Farklı `--protocol` sonuçları karşılaştırılamaz.** Aynı tabloda toplama denemesi
  script tarafından reddediliyor; bu davranış korunmalı.
- **Ham AUC yayımlanmaz.** COX'ta ham 0.87 vardı ve hiçbir şey değildi. Docking skoru
  ve temas sayısı çıkarılmadan hiçbir sayı rapor edilmemeli.
- **Ön-kayıt.** Cep pozisyon listesi, uç noktalar ve kontroller sonuçlara bakılmadan
  önce yazılmalı (pair-local ve pocket analizlerinde bu disiplin uygulandı; GTPase
  bulgusunu geri çektiren şey de tam bunun eksikliğiydi).
- **Tek çift / tek yapı sonucu tekrarlanmadan iddia edilmez.** GTPase dersi:
  p = 0.010, gürültü tabanının 4 katı, ve iki bağımsız çiftte tekrar etmedi.
- **Veri hijyeni.** `data/` ve `results/` git'te değil, DVC ile `gs://frustx`
  remote'unda (`data.dvc`, `results.dvc`). Yeni çıktı sonrası
  `dvc add results/ && dvc push`, ardından pointer commit'i. Scratchpad oturumlar
  arasında siliniyor — yeniden üretilmesi pahalı olan her şey `results/` altına.
