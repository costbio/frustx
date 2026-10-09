# Kinom veri seti — uçtan uca özet

Bu dosya, Tasarım A için kurduğumuz veri zincirinin ne olduğunu, nereden nasıl
çekildiğini, her adımda ne elde ettiğimizi ve şu anda elimizde tam olarak ne olduğunu
anlatır. Rakamlar `data/kinome/**/[a-z]*_funnel.json` ve `data/kinome/analysis/*.json`
dosyalarından alınmıştır; her biri üretildiği tarihi ve kullanılan eşikleri kendi içinde
taşır.

İngilizce karşılığı ve yöntem gerekçeleri `docs/method.md` içindeki
"The kinome dataset" bölümündedir. Bu dosya anlatı, o dosya spec.

**Amaç.** Aynı inhibitörün farklı kinazlarla ko-kristal yapıları üzerinde hesaplanan
frustrasyon indeksinin, ölçülmüş bağlanma afinitesini sıralayıp sıralamadığını test etmek
(**Tasarım A**). Bu dosyadaki işlerin hiçbiri frustrasyon hesaplamıyor; hepsi o testin
girdisini kurmak ve testin istatistiksel olarak mümkün olup olmadığını ölçmek içindir.

**Zincir.** Her adım bir sonrakinin girdisi:

```
KLIFS (yapılar)  →  BindingDB (kapsam + aktivite)  →  Davis 2011 (birincil Kd)
                                   ↓
                     Design A fizibilite  →  Genişletilmiş set
```

---

## 1. KLIFS yapı manifesti

**Script:** `scripts/klifs_manifest.py` · **Çıktı:** `data/kinome/klifs/`

### Kaynak ve yöntem

KLIFS REST API v2 (`https://klifs.net/api_v2`), stdlib `urllib` ile doğrudan çağrıldı.
`opencadd` kurulu değil ve bağımlılık zinciri (bravado/biopandas) Python 3.14'te güvenilir
değil; opencadd zaten aynı API'nin istemcisi olduğundan bağımlılık eklemeye gerek olmadı.
**Yapı dosyası indirilmedi** — bu adım yalnızca metadata üretir; indirme sonraki adımın işi.

### Tahmin etmeden okunan, filtreleri değiştiren dört gerçek

Her biri önce canlı API'den çekilip yazdırıldı, sonra filtre yazıldı:

1. "Ligand yok" değeri `""` veya `"-"` değil, **tamsayı 0**.
2. NMR girişlerinde `resolution = "0"`. Yani `resolution <= 2.5` tek başına **tüm NMR
   modellerini geçirirdi**. Bu satırlar model numarasını `alt` alanında taşıyor ("1".."20").
   `normalise()` sıfır çözünürlüğü NaN'a çevirir, böylece filtre reddeder.
3. `resolution` ve `quality_score` string olarak geliyor; ligand adları HTML-escape'li
   (`&apos;`).
4. KLIFS, RCSB'nin obsolete ettiği PDB ID'lerini hâlâ listeliyor.

Ayrıca: ortosterik ligand alanında **hiç iyon veya tampon yok**; tek inhibitör-olmayanlar
nükleotidler ve analogları. Dışlama listesi bu yüzden gerçekte görülenlerden kuruldu
(ATP, ADP, ANP, ACP, AGS, AMP, ADN, 3AM, AP2, M33, 112, NBS, 3GU + MG, MN).

### RCSB doğrulaması ve kritik sıra

KLIFS filtrelerinden geçen ~4.400 satırın **tamamı** RCSB GraphQL ile doğrulandı
(cache: `rcsb_cache.json`, yeniden koşuda ağa gitmez).

> **Sıra önemli:** RCSB kontrolü ve ona bağlı elemeler **tekilleştirmeden ÖNCE** yapılır.
> Aksi halde RCSB'nin reddettiği bir yapı (kinaz, ligand) çiftini kazanıp, geçerli bir
> alternatif varken çifti tamamen kaybettirirdi.

`rcsb_status` değerleri ve ne yaptıkları:

| durum | sayı | sonuç |
|---|---|---|
| `ok` | 4.396 | kalır |
| `obsolete_remapped` | 12 | **düşer** (aşağıda) |
| `ligand_missing` | 10 | düşer |
| `non_xray` | 2 | düşer |
| `obsolete_no_replacement` | 0 | düşer |

### Huni (çekim: 2026-10-04, KLIFS API 0.3.1)

| adım | yapı | PDB | kinaz | UniProt | ligand |
|---|---|---|---|---|---|
| ham (tüm türler) | 14.068 | 6.780 | 359 | 351 | 4.180 |
| `species == Human` | 13.325 | 6.375 | 318 | 311 | 4.014 |
| ortosterik ligand var | 11.286 | 5.541 | 289 | 286 | 4.014 |
| dışlama listesi | 10.051 | 4.970 | 265 | 262 | 4.001 |
| çözünürlük ≤ 2,5 Å | 7.404 | 3.586 | 222 | 219 | 2.986 |
| kalite skoru ≥ 6 | 7.180 | 3.478 | 218 | 215 | 2.894 |
| altloc `A` veya boş | 4.420 | 3.475 | 218 | 215 | 2.886 |
| RCSB: obsolete, yerine yok | 4.420 | 3.464 | 218 | 215 | 2.886 |
| RCSB: ligand girişte yok | 4.410 | 3.460 | 218 | 215 | 2.882 |
| RCSB: X-ray değil | 4.408 | 3.458 | 217 | 214 | 2.882 |
| RCSB: obsolete→remap | 4.396 | 3.458 | 217 | 214 | 2.882 |
| **tekilleştirme** | **3.196** | **3.190** | **217** | **214** | **2.882** |

Not: "obsolete→remap" adımında satır düşmez ama PDB sayısı azalır — eski ID'ler tabloda
zaten var olan yeni ID'lere katlanır. CDK7 bu adımlarda tamamen çıkar (tek yapıları 7B5O
ve 7B5Q cryo-EM).

### İki tasarım kararı

**Tekilleştirme anahtarı UniProt değil, KLIFS kinaz (domain) ID'si.** JAK1/2/3, TYK2,
RSK/MSK gibi proteinlerde tek UniProt altında iki kinaz domaini var (JH1 katalitik, JH2
psödokinaz; RSK'da N- ve C-terminal). UniProt anahtarıyla bu iki domainin yapıları tek
çifte birleşiyordu. Domain anahtarına geçince 3 yapı geri geldi: JAK2/2HB (4P7E),
JAK2-b/DQX (5UT4), JAK2-b/SKE (5USZ). Manifestte 222 satır `multi_domain_uniprot`.

**`obsolete_remapped` satırlar manifeste girmez.** KLIFS'in kalite/çözünürlük/eksik-rezidü
değerleri *eski deposite* aittir ve indirilecek dosyayı tarif etmez. Örnek: 5J7H yerine
geçen 6MX8 — KLIFS eski depositi 8 kalite ve 0 eksik rezidü ile skorlamış, 6MX8'in kendisi
6,8 ve 3 eksik rezidü. 12 satır düştü, **hiçbir (kinaz, ligand) çifti kaybolmadı**: KLIFS
yeni girişi kendi satırıyla zaten listeliyordu, o satır dedup'ı kazandı.

### Silmeyip işaretlediklerimiz

İndirme adımının bilmesi gereken, ama veriyi atmayı gerektirmeyen durumlar:

| bayrak | sayı | ne demek |
|---|---|---|
| `ligand_chain_differs` | 20 | ligand 3 karakterli zincirde (`AAA`, `BBB`) — PDB formatına sığmaz, mmCIF gerekir |
| `multi_copy_ligand` | 75 | aynı protein zincirinde ligandın birden fazla kopyası — cepteki seçilmeli |
| `altloc_ligand_conflict` | 8 | aynı PDB+zincirin farklı altloc'ları **farklı ligand** taşıyor (6HOP, CDK2 1H00/1H01/1H07/1H08, 1RDQ, 4KIO, 5LWN, 8AOA) |
| `multi_domain_uniprot` | 222 | UniProt'un birden fazla KLIFS kinaz domaini var |

### Çıktı dosyaları

- `klifs_manifest.csv` — 3.196 satır, 22 kolon. Her satır bir (kinaz domaini, ligand)
  kompleksi: `pdb_klifs` (KLIFS'in verdiği, küçük harf), `pdb` (geçerli RCSB ID), zincir,
  altloc, `ligand_code` (= HETATM adı), `allosteric_ligand_code`, çözünürlük, kalite,
  eksik rezidü/atom, DFG/αC durumu, `rcsb_status` ve yukarıdaki bayraklar.
- `klifs_raw.csv` — filtrelenmemiş ham tablo (tekrarlanabilirlik).
- `klifs_kinases.csv` — `kinase_information`, tüm türler (1.127 satır). İnsan kısmı 555
  kinaz / 542 UniProt. BindingDB adımının hedef listesi bu.
- `klifs_funnel.json` — her adımın sayıları, eşikler, çekim tarihi, API sürümü,
  `remap_collisions`.
- `rcsb_cache.json` — RCSB yanıtları (3,8 MB).

---

## 2. BindingDB aktiviteleri

**Script:** `scripts/bindingdb_activity.py` · **Çıktı:** `data/kinome/bindingdb/`

### Kaynak

URL tahmin edilmedi: indirme sayfası okunup oradaki bağlantı `SDFdownload.jsp` ara
sayfasına, oradan da gerçek dosyaya takip edildi. Dosya
**`BindingDB_All_202610_tsv.zip`** (567 MB), MD5 `ab7ca3dc152d395b773ff987d8e09f64`,
yayınlanan `.md5` dosyasıyla **doğrulandı**. Zip açılıp diske yazılmıyor; içindeki 9,0 GB
TSV akış halinde okunuyor (tek geçiş, ~2,5 dk). Diskte duran kopya MD5'i tutuyorsa
yeniden indirilmez; `.provenance.json` ilk indirme tarihini korur.

### Dosyanın yapısı (okundu, varsayılmadı)

- 3.243.660 satır, **640 alan**: 40 sabit kolon + hedef zincir başına 12 kolonluk 50 blok.
- Satırlar 640'a doldurulmuş. Yani çok zincirli hedef *daha uzun satır* değil, **daha
  fazla dolu blok** demek. Beklenen alan sayısıyla uyuşmayan satır `problems`'a yazılır.
- **Okunamayan satır: 0.** 226 satırda InChIKey yok (sayıldı, eşleşemez).
- Afiniteler string: `"12.5"`, `">10000"`, `"<0.03"`, bazen `"> 20000"`.
- Organizma kolonu aynı organizmayı üç farklı yazıyor: `Homo sapiens` (2,48 M satır),
  `Human` (0,21 M), `Homo sapiens (Human)`. Üçü de kabul edildi (`--organisms`).
- SwissProt primary-id hücresi boşlukla ayrılmış birden fazla accession tutabiliyor.

### Huni

| adım | satır | ligand | kinaz |
|---|---|---|---|
| tüm satırlar | 3.243.660 | 1.354.629 | 488 |
| insan organizma (3 etiket) | 2.695.170 | 1.155.171 | 474 |
| bir zincir KLIFS insan kinazı | 840.917 | 323.974 | 474 |
| tam olarak **bir** kinaz zinciri | 831.780 | 321.016 | 474 |
| Ki, Kd veya IC50 var | 806.171 | 310.936 | 474 |
| **ligand eşleşti: full InChIKey** | **32.083** | **1.297** | 443 |
| **ligand eşleşti: `bindingdb_no_stereo`** | **21.082** | **621** | 432 |
| *(yan dal)* yalnız skeleton | 1.071 | 68 | 381 |
| değerler (full + no_stereo, ölçüm başına) | 53.207 | 1.918 | 453 |
| kaynak tekrarı çıkarıldı | 50.747 | 1.918 | 453 |

~6.800 satır hedefinde **birden fazla** KLIFS kinazı adlandırıyor (ör. ABL1'i de listeleyen
bir "PI3K alpha" kaydı); değer tek kinaza atfedilemediği için ayrı bir huni adımında düştü.

### Yol boyunca çözülen dört problem

**1. BindingDB stereo katmanını sıklıkla düşürüyor.** Ruksolitinib, staurosporin,
aksitinib, lestaurtinib gibi bileşiklerin InChIKey'i `-UHFFFAOYSA-` ile bitiyor; RCSB'nin
anahtarları stereoyu taşıyor. 293 ligand bu yüzden hiç eşleşmiyordu. Üçüncü eşleşme
seviyesi tanımlandı:

- `full` — 27 karakterin tamamı aynı.
- **`bindingdb_no_stereo`** — ilk 14 karakter aynı, BindingDB anahtarının 2. bloğu
  `UHFFFAOYSA`, ligandın anahtarı stereo taşıyor, protonasyon aynı **ve** iskelet tek bir
  manifest ligand grubuna uyuyor.
- `skeleton` — kalan her şey, `match_note` nedeni söyler
  (`different_stereo_or_isotope`, `protonation_only`, `no_stereo_and_protonation`,
  `no_stereo_several_ligands`). Ayrı dosyada tutulur, ana tabloya karışmaz.

BindingDB'nin kendi HET-ID kolonuyla çapraz kontrolde uyuşmazlıklar **293 → 7**'ye düştü;
yani eksik stereo katmanı neredeyse tüm uyuşmazlığı açıklıyordu.

**2. HET tie-break.** Stereosu belirtilmemiş bir kayıt iki enantiyomere de uyuyorsa
(RXT/RG4 gibi), BindingDB'nin HET kolonu hangisi olduğunu söylüyor:
`match_note = "het_tiebreak"` (455 aktivite satırı, 18 ligand grubu). RXT böylece 0 → 388
aktivite satırına çıktı. HET boşsa veya aday grupların kodu değilse hiçbir şeye karar
vermez, satır skeleton'da kalır.

**3. Domain ataması — ölçüm başına, konstrükt aralığından.** Hedef adlarındaki kalıntı
aralıkları (`JAK2 [808-1132]`, `TYK2 [556-888]`, `(aa658-end)`, silme mutantları için
birden fazla aralık, `'YVMA'` insersiyonları, en-dash) ayrıştırılıyor. UniProt REST'ten
"Protein kinase" domain sınırları alınıp (cache), KLIFS'in 85 rezidülük pocket dizisi
UniProt dizisinde konumlandırılarak her KLIFS kinazı bir UniProt domainine eşlendi:
**26 kinazdan 25'i** (GCN2-b eşlenemedi — pocket'ı UniProt'un tek "Protein kinase" domaini
içine düşmüyor; tahmin edilmedi, raporlandı).

Sağlamalar doğru: JAK2 [808-1132] → JAK2 (JH1), [536-812] → JAK2-b (JH2),
TYK2 [556-888] → TYK2-b (JH2), [871-1187] → TYK2 (JH1).

Kural sırası: **konstrükt > yapı**. Aralık varsa ve bir domaini ≥%80 kapsıyorsa o domain
(`domain_source = "construct"`); aralık yoksa (tam protein) ligandın kristalinin bulunduğu
domain (`"structure"`); hiçbiri yoksa satır UniProt seviyesinde kalır (`"uniprot_level"`).
Her iki domaini kapsayan konstrükt `domain_ambiguous`.

Konstrükt kuralının yapı kuralından önce gelmesi 2 ölçümü başka domaine taşıdı (M4G, M57:
kristalleri JAK2-b'de ama o değerler JH1 konstrüktünde ölçülmüş) ve 14 ölçümü
`domain_ambiguous`'tan kurtardı.

**4. Kaynak tekrarları.** Aynı (uniprot, ligand, ölçüm, değer) + aynı PMID taşıyan satırlar
toplamadan önce tekilleştirildi: 2.546 satır (+ DOI üzerinden 1).

### Kalite kolonları

`inconsistent` (yayılım > 1 log birimi), `n` / `n_censored` (sansürlü değerler medyana
**asla** girmez), `multichain_target`, `multi_domain_uniprot`, `domain_source`,
`domain_ambiguous`, ve **`primary_eligible`**: domain ölçümün kendisinden biliniyorsa True
(`single_domain`, `construct`), kristalden çıkarıldıysa veya belirsizse False
(`structure`, `uniprot_level`, `*_ambiguous`).

### Çıktı ve kapsama

- `activities.csv` — 25.641 toplanmış satır (16.375 `full` + 9.266 `bindingdb_no_stereo`).
  Anahtar: (uniprot, kinaz domaini, domain_source, InChIKey, ölçüm, ligand grubu, seviye).
  `median_pX`, `n`, `std`, `n_censored` sansürsüz değerler üzerinden.
- `activities_skeleton.csv` — 601 satır, ayrı tutulan skeleton eşleşmeleri.
- `ligand_ids.csv` — 2.882 ligand kodu → grup, InChIKey, SMILES, eşleşme seviyesi,
  HET çapraz kontrol sayıları.
- `bindingdb_funnel.json`, `chemcomp_cache.json`, `uniprot_cache.json`.

Kapsama (3.194 manifest çifti üzerinden; "comparable" = aynı ligandın **≥2 farklı
UniProt**'ta yapısı *ve* her birine atanmış Kd/Ki):

| sürüm | Kd/Ki olan çift | yalnız IC50 | veri yok | comparable ligand |
|---|---|---|---|---|
| full | 413 | 844 | 1.908 | 29 |
| full, yalnız primary | 382 | 797 | 1.989 | 27 |
| full + no_stereo | 640 | 1.234 | 1.280 | 49 |
| **full + no_stereo, yalnız primary** | **588** | 1.153 | 1.417 | **45** |

Aynı proteinin iki domaini (JAK2/JAK2-b) seçicilik sayılmaz; `intra_protein_domain_pairs`
olarak ayrı listelenir — tek vaka: SKE.

---

## 3. Davis 2011 — birincil seçicilik kaynağı

**Script:** `scripts/davis_activity.py` · **Çıktı:** `data/kinome/davis/`

### Neden ayrı bir kaynak

BindingDB yüzlerce yayından, farklı assay'lerden derlenmiş; aynı çift için değerler
farklı laboratuvarlardan gelir. Davis et al. (*Nat Biotechnol* 29:1046, 2011,
PMID 22037378, DOI 10.1038/nbt.1990) **tek assay, tek ölçüm tipi**: KINOMEscan,
72 inhibitör × 442 kinaz konstrüktü, Kd. Seçicilik istatistikleri için bu **birincil**;
BindingDB kapsam ve duyarlılık analizi için kalır.

### Kaynak

Makalenin Nature sayfasındaki supplementary bağlantıları **caption'a göre** bulunup
indirildi (URL kalıbı tahmin edilmedi): Tablo 1 (kinaz listesi), Tablo 3 (72 bileşik),
**Tablo 4 (Kd matrisi)**. URL + MD5 + tarih `raw/provenance.json`'da.

> **İşlenmiş sürümler kullanılmadı.** TDC/DeepDTA gibi dağıtımlar "bağlanma yok"
> değerlerini sabit bir pKd'ye çeviriyor ve kinaz varyantlarını (mutant, fosfo,
> domain konstrüktü) kaybediyor. İkisi de bu çalışmanın tam olarak ihtiyaç duyduğu
> bilgi.

2011 `.xls` formatı için `xlrd` gerekti — venv'e kuruldu ve `pyproject.toml`'un `dev`
extra'sına eklendi (openpyxl bu formatı okumuyor). Import, okuma fonksiyonunun içinde:
ağsız testler xlrd'ye ihtiyaç duymuyor.

### Matris

442 × 72 = **31.824 hücre**, hepsi **metin** olarak saklanmış (0,016–9.900 nM).
Okunamayan hücre: 0. **22.400 hücre boş.**

Tablo 4'ün makale sayfasındaki açıklaması: *"Blank fields indicate combinations that were
tested, but for which binding was weak (Kd > 10 uM), or not detected in a 10 uM primary
screen."* Yani boş hücre = `censored=True`, ve **yalnızca alt sınır** olarak tutulur
(`censor_bound_pX` = pKd < 5). Sayısal değer gibi asla kullanılmaz, medyana girmez.

### Kinaz eşlemesi (DiscoverX adı → UniProt → KLIFS)

2011'in Entrez sembollerinin **23'ü güncelliğini yitirmiş** (FRAP1 = MTOR, ZAK, PCTK1,
CDC2L1…), bu yüzden eşleme **accession üzerinden** yapıldı (RefSeq protein → UniProt ID
mapping, sürüm numarası atılarak; cache: `idmapping_cache.json`).

**437/442 eşlendi:** 427 accession, 3 accession'ın kendisi zaten UniProt (WEE2 P0C1S8
gibi), 8 gen sembolü (yedek yol, accession hiçbir şey vermediğinde).

| durum | sayı | kimler |
|---|---|---|
| `ok` | 437 | — |
| `non_human` | 3 | PFCDPK1, PFPK5 (*P. falciparum*), PKNB (*M. tuberculosis*) |
| `no_uniprot` | 1 | CDC2L1 |
| `domain_unresolved` | 1 | GCN2(Kin.Dom.2,S808G) |

Varyant sınıfları: **wild_type 382**, `mutant` 54, `phospho_or_other` 1.
17 domain konstrüktü doğrudan ilgili KLIFS domainine atandı
(`JAK1(JH2domain-pseudokinase)` → JAK1-b; JH2 N-terminal olduğu için sıra 1, JH1 → 2).

**Açık karar:** `ABL1-nonphosphorylated` wild_type sayıldı
(`variant_note = "abl1_nonphos_as_wt"`) — Davis'te niteliksiz bir ABL1 satırı yok ve
fosforillenmemiş konstrükt ona en yakın olan. `ABL1-phosphorylated` `phospho_or_other`
kaldı. Bu karar comparable seti 17 → 19 ligandına çıkardı (ABL1 eklenen: 1N1, DB8, STI,
VX6; yeni gelen: AXI, NIL).

**İsimlendirme tuzağı:** Davis'in "RSK1"i (RPS6KA1) KLIFS'te **RSK3**, Davis'in "RSK3"ü
(RPS6KA2) KLIFS'te **RSK1**. Protein kimliği accession'dan geldiği için doğru; yalnızca
adlar farklı.

### Bileşik eşlemesi (ad → PubChem → InChIKey → manifest grubu)

Her isim ve eşanlamlısı PubChem'den çözüldü (cache: `pubchem_cache.json`).
**Hiçbir belirsizlik otomatik çözülmedi.** 5 bileşik kaldı; SMILES'ler nedeni gösterdi:
alias'lar aynı ana bileşiğin **tuz formlarına** çözülüyordu.

| bileşik | durum | karar |
|---|---|---|
| CHIR-258/TKI-258 | alias'lar farklı CID | **CID 135398510** (dovitinib, serbest baz) → grup **38O** |
| R406 | alias'lar farklı CID | **CID 11213558** (tamatinib, serbest baz) → grup **585** |
| PTK-787 | 151194 serbest baz / 151193 süksinat | manifest grubu yok → **dışarıda** |
| CI-1033 | 156414 / 156413 (2HCl) / 67292089 (HCl) | manifest grubu yok → **dışarıda** |
| BIBF-1120 (derivative) | türev, adlandırılan bileşik değil | **dışarıda** |

İlk ikisi `match_note = "salt_resolved"` ile işaretli; `SALT_RESOLUTIONS` sözlüğünde açıkça
yazılı ve seçilen CID'in PubChem'in döndürdüğü adaylardan biri olması zorunlu (değilse
hata verir, sessizce seçmez).

**37 bileşik** manifest ligand grubuna eşleşti (36 `full` + 1 `bindingdb_no_stereo`).
2'si yalnız skeleton seviyesinde (flavopiridol → F9Z, PHA-665752 → PFY).

Ayrıca Tablo 4 başlığı `INCB18424` yazarken Tablo 3 `INCB018424` yazıyor; kolonlar konuma
göre eşleştirildi ve yazım farkı raporlandı.

### BindingDB ile tutarlılık — ve döngüsellik tuzağı

BindingDB, Davis'i **kendi içinde taşıyor**: ChEMBL üzerinden alınmış 14.497 Kd satırı,
PMID 22037378. Bu PMID atıldıktan sonra bile kalan değerlerin **%74'ü** Davis'le birebir
aynıydı — başka atıflar altındaki kopyalar.

Script bu yüzden **değerlerinin ≥%80'i Davis'le özdeş olan referansları** dışlıyor
(`--copy-threshold`, varsayılan 0,8) ve PubMed'den başlık/yazar/yıl çekerek
**38 referansı** raporluyor. Başlıcaları:

| referans | özdeş/karşılaştırılan | kim |
|---|---|---|
| 18183025 | 1.054 / 1.289 | Karaman MW 2008, *Nat Biotechnol* |
| (referans yok) | 985 / 1.219 | — |
| 19654408 | 542 / 572 | Zarrinkar PP 2009, *Blood* |

Sonuç:

| karşılaştırma | çift | medyan Δ | medyan \|Δ\| | \|Δ\|>1 | Spearman ρ |
|---|---|---|---|---|---|
| Davis kopyası (sağlama — uyuşmalı) | 3.384 | 0,00 | 0,00 | %0,1 | **0,998** |
| yalnız Davis PMID çıkarılmış | 1.536 | 0,00 | 0,00 | %2,3 | 0,971 |
| **kopya referanslar da çıkarılmış** | **433** | **−0,16** | **0,30** | **%10,2** | **0,883** |

İlk satır kinaz ve ligand eşlemelerinin doğru olduğunu teyit eder. Üçüncü satır gerçek
bağımsız tutarlılıktır: ρ 0,88, değerlerin %10'u 1 log biriminden fazla ayrışıyor.

### Çıktı

- `activities.csv` — 31.464 satır. Kolonlar BindingDB `activities.csv` ile **uyumlu**
  (birleştirilebilsin diye) + `source="davis2011"`, `davis_kinase`, `davis_compound`,
  `variant_class`, `variant_note`, `kd_raw`, `kd_nM`, `censor_bound_pX`. Her satır **bir**
  ölçüm: `n` = 1 (veya sansürlüyse 0), `std` boş, hiçbir şey havuzlanmıyor — CDK4'ün iki
  siklin kompleksi de, mutant ile yabanıl tip de.
- `kinase_map.csv` (442 konstrükt), `compound_map.csv` (72 bileşik), `davis_funnel.json`,
  4 cache (`idmapping`, `pubchem`, `pubmed`, `uniprot`).

**Davis comparable set: 20 ligand** — STU 26 protein, 1N1 10, DB8 9, B49 6, MI1 5, STI 5,
B96 4, VX6 4, 8X7/BAX/R78/VGH 3, ve 2'li 9 ligand.

---

## 4. Design A fizibilite

**Script:** `scripts/design_a_feasibility.py` · **Çıktı:** `data/kinome/analysis/design_a_*`

Soru: veri bu testi taşıyor mu? Frustrasyon hesaplanmadı, model kurulmadı.

### Kapsam

Davis comparable seti — ligand başına, o ligandın **manifestte yapısı olan**, **wild-type**,
**sansürsüz** Kd'ye sahip kinazlar: **20 ligand, 96 kompleks, 68 protein**.

Sansürlü hücreler 10 µM sınırına **çekilmedi**, dışlandı — aksi halde her aralığı assay'in
hiç ölçmediği bir miktarda şişirirdi.

### Varsayımlar (hepsi JSON'da yazılı)

- Model: `frustrasyon_i = oran · pKd_i + N(0, 1)`, kompleks başına bağımsız.
- **Etki büyüklüğü `oran = slope / noise_sd`**, ikisi de afinite ölçeğinin log biriminde.
  Yalnızca oran önemli → **frustrasyon indeksinin kendi birimi hiç devreye girmiyor.**
- Test: Spearman ρ üzerinde iki yanlı **permütasyon testi**, α = 0,05. n ≤ 7 için
  **tam enümerasyon**, üstünde 5.000 örnek. x ekseninde gözlenen pKd vektörü kullanılır,
  yani gerçek aralık ve eşitlikler korunur.
- Yok sayılanlar: afinite ölçüm hatası (Davis Kd kesin kabul edildi), sansürlü hücreler,
  tek yapının kompleksi temsil etmesi, ligand içinde ortak frustrasyon sapması.

### Aralık

Medyan aralık **1,40 log** birimi (0,29 – 4,62). 1N1 en geniş (4,62), VGH en dar (0,29).

### Güç — ligand başına test çoğunlukla imkânsız

| ligand | n | min ulaşılabilir p | güç (oran 2,0) | güç (oran 1,0) |
|---|---|---|---|---|
| STU | 26 | 0,0002 | **1,00** | **0,99** |
| 1N1 | 10 | 0,0002 | **1,00** | **0,83** |
| DB8 | 9 | 0,0002 | **1,00** | **0,81** |
| B49 | 6 | 0,0028 | 0,58 | 0,28 |
| MI1 | 5 | 0,0167 | 0,36 | 0,12 |
| STI | 5 | 0,0167 | 0,15 | 0,06 |
| B96, VX6 | 4 | 0,0833 | 0 | 0 |
| R78, BAX, VGH | 3 | 0,3333 | 0 | 0 |
| 11 ligand | 2 | **1,0000** | 0 | 0 |

**11 ligandda n=2** — en küçük ulaşılabilir p 1,0, yani ilişki ne kadar güçlü olursa olsun
hiçbir sonuç anlamlı olamaz. n=4'te bile en küçük p 2/24 = 0,083 > 0,05. Tam enümerasyon
bunu tahmin değil, kesin sonuç olarak verir.

### Güç — havuzlanmış model

`frustrasyon ~ pKd + (1|ligand)` modelinin sabit etkisi: ligand **içinde** alınıp
merkezlenmiş sıralar üzerinde Pearson. Böylece yalnızca ligand içi sinyal sayılır, bir
ligandın genel frustrasyon seviyesi katkı yapamaz. Null: her ligandın içinde bağımsız
permütasyon.

| oran | 0,1 | 0,2 | 0,33 | 0,5 | **0,6** | 0,8 | 1,0 | 2,0 |
|---|---|---|---|---|---|---|---|---|
| havuzlanmış (96 kompleks) | 0,07 | 0,17 | 0,38 | 0,70 | **0,85** | 0,97 | 1,00 | 1,00 |
| yalnız ranking (78) | 0,08 | 0,17 | 0,38 | 0,69 | **0,83** | 0,97 | 1,00 | 1,00 |
| negatif kontrol (18) | 0,03 | 0,02 | 0,04 | 0,04 | 0,09 | — | 0,09 | **0,26** |

**Tespit edilebilir en küçük etki ≈ oran 0,6** (0,5 → 0,70; 0,6 → 0,85). Tez için cümle:
*havuzlanmış tasarım, frustrasyon indeksi kendi artık gürültüsüne göre log birim başına
en az ~0,6 birim hareket ediyorsa ilişkiyi saptar; bundan zayıfı bu veriyle görülemez.*
4.000 simülasyonda Monte-Carlo hatası ~±0,6 puan, dolayısıyla oranı 0,05'ten ince
çivilemek anlamsız.

### Roller

- `ranking` (13 ligand / 78 kompleks) — aralık ≥ 1 log.
- `negative_control` (**7 ligand / 18 kompleks**: VX6, BAX, VGH, GUI, NIL, LY4, 88Z) —
  aralık < 1 log. Setten **çıkarılmadı**: kinazları pratikte eşit potent olduğu için
  **ters hipotezi** test ederler (sağlam bir indeks burada sıralama *bulmamalı*).

> **Uyarı:** negatif kontrol setinin kendisi düşük güçlü (oran 2,0'da 0,26). Oradaki bir
> null sonuç "sıralama yok, öngörüldüğü gibi" ile "görmeye yetecek veri yok"u ayırt
> etmiyor. Tezde bu kadarıyla yazılmalı.

### Paralog çiftleri

**110 çift** (19 aynı KLIFS ailesi, 91 yalnız aynı grup). En yakın akraba çiftleri — iki
cep en az farkla ayrıldığı için frustrasyon indeksi açısından en zor ve en ilginç vaka.

| ligand | çift | ΔpKd |
|---|---|---|
| STI | ABL1 / ABL2 | 0,96 |
| VX6 | AurA / AurC | 0,21 |
| MI1 | JakA ailesi, 6 çift | 0,44 – 1,48 (JAK3 9,80 > JAK2 9,24 > JAK1 8,80 > TYK2 8,32) |
| B49 | HPK1 / PAK6 | 2,18 |
| DB8 | LOK / MST3 | 1,73 |
| 1N1 | BMX / BTK | **0,00** (tam eşitlik) |

### Kaldırılan: BindingDB çapraz kontrolü

Başlangıçta Davis aralıklarının BindingDB'de korunup korunmadığı kontrol ediliyordu. Ortak
96 kompleksin **%63'ünde** BindingDB değeri Davis'le birebir aynı çıktı — BindingDB'nin
ChEMBL üzerinden taşıdığı Davis kopyası. Karşılaştırma büyük ölçüde Davis'i kendisiyle
kıyasladığı için bölüm tamamen kaldırıldı; gerekçe script'in docstring'inde yazılı.

---

## 5. Genişletilmiş set

**Script:** `scripts/extended_set.py` · **Çıktı:** `data/kinome/analysis/extended_set*`

Soru: Davis'in ölçmediği BindingDB ligandları ne kadar katkı sağlıyor?

### Kurallar

- Bir ligand **tüm** değerlerini **tek** kaynaktan alır; Davis ölçmüşse Davis. İki kaynağı
  ligand içinde karıştırmak, assay/aggregation farkını tam da Tasarım A'nın yaptığı
  karşılaştırmanın içine koyardı.
- BindingDB tarafında **IC50 hiç kullanılmıyor**; ligand başına Kd *veya* Ki (daha çok
  proteini kapsayan, eşitlikte Kd). 37 ligandda ikisi de vardı; 5'i Ki seçti (3NG, 7KC,
  N5Q, QS0, YAM) — onlarda birer protein için var olan Kd değeri feda edildi.
- `inconsistent` değerler atıldı (45 satır).
- Sansürlü değerin medyana girmediği **assert** ile doğrulandı (iddia değil, kontrol).
- `single_measurement` = BindingDB'de **tek rapordan** toplanmış değer. Atılmıyor,
  işaretleniyor; her güç sayısı **iki kez** hesaplanıyor (dahil/hariç). Davis değerleri
  işaretlenmiyor: her biri tanımı gereği tek bir KINOMEscan Kd'si ve Davis birincil kaynak,
  dolayısıyla bayrak "genişletmenin ne kadar ince veriye yaslandığını" ölçüyor.

Kontroller boş döndü: iki kaynaktan beslenen ligand yok, iki ölçüm tipini karıştıran ligand
yok, sansürlü değer medyana girmemiş, bir komplekste birden fazla yapı yok.

### Darboğaz yapılar, veri değil

BindingDB tarafında 3.648 satır → **yapı filtresinden sonra 517** → ≥2 protein şartından
sonra **47**. Bağlanma verisi bol; eksik olan ko-kristal yapı.

### Set

| | ligand | kompleks | protein |
|---|---|---|---|
| Davis | 20 | 96 | 68 |
| BindingDB (eklenen) | 22 | 47 | — |
| **toplam** | **42** | **143** | **86** |

Eklenen 22 ligandın **20'si n=2** (biri 3, biri 4) — hiçbiri tek başına test edilemez.
Genişletme geniş ama sığ.

### Roller — ve `exploratory`

| rol | ligand | kompleks |
|---|---|---|
| `ranking` | 18 | **88** |
| `exploratory` | 3 | 7 |
| `negative_control` | 21 | 48 |

`exploratory` kuralı (genel, ligand adı hard-code edilmedi): aralığı ≥ 1 log olduğu için
ranking adayı olan, ama **en yüksek ve en düşük değeri de tek rapora dayanan** ligand
ranking'den çıkarılır. Gerekçe: o aralık ölçüm gürültüsü olabilir. Kural tam olarak
**8X7, 537, YAM**'ı yakaladı:

| ligand | kompleksler | durum |
|---|---|---|
| 537 | JNK3 7,66 / PDK1 5,90 | iki uç da tek rapor |
| YAM | FAK 8,03 / PYK2 6,82 | iki uç da tek rapor, üstelik Ki seçilmiş |
| 8X7 | PLK1 8,52 / Wee1 8,19 / MYT1 6,49 | iki uç tek rapor, ortadaki n=2 |

**Setten çıkmıyorlar** — frustx bu kompleksleri yine koşacak; yalnızca birincil sıralama
testinin kanıtı sayılmıyorlar. `needs_decision` bayrağı kuralın kimi yakaladığını kayıtta
tutar.

### Güç (single_measurement dahil)

| kapsam | n | 0,5 | **0,6** | 1,0 | 2,0 | %80 eşiği | min ulaşılabilir p |
|---|---|---|---|---|---|---|---|
| davis_only | 96 | 0,70 | **0,85** | 1,00 | 1,00 | 0,6 | 0,0002 |
| extended_all | 143 | 0,70 | **0,85** | 1,00 | 1,00 | 0,6 | 0,0002 |
| **extended_ranking** | **88** | 0,70 | **0,84** | 1,00 | 1,00 | **0,6** | 0,0002 |
| davis_ranking | 78 | 0,69 | 0,83 | 1,00 | 1,00 | 0,6 | 0,0004 |
| extended_negative_control | 48 | 0,08 | 0,09 | 0,18 | **0,53** | hiç | 0,0004 |
| bindingdb_ranking | 10 | 0 | 0 | 0 | 0 | **hiç** | **0,0636** |
| extended_exploratory | 7 | 0 | 0 | 0 | 0 | hiç | 0,0874 |

`single_measurement` hariç tutulduğunda büyük kapsamlar neredeyse değişmiyor
(extended_ranking 80 kompleks, 0,6'da 0,846), ama `bindingdb_ranking` **17 → 2 komplekse**
çöküyor: BindingDB'nin ranking katkısının neredeyse tamamı tek rapora dayanıyor.

### Üç temel bulgu

1. **Genişletme tespit edilebilir en küçük etkiyi düşürmüyor.** Davis tek başına da,
   genişletilmiş set de oran 0,6'da %80'i geçiyor (0,845 vs 0,849). 2 noktalı gruplar
   ligand içi bilgi neredeyse taşımıyor.
2. **Kazanç negatif kontrolde:** 18 → 48 kompleks, oran 2,0'da güç **0,26 → 0,53**. Hâlâ
   %80'in altında ama yaklaşık iki katı bilgi.
3. **BindingDB'nin kendi ranking katkısı kanıtlanabilir biçimde yetersiz:** 5 ligand ×
   2 kompleks → 2/32 = 0,0625 en küçük ulaşılabilir p (ölçülen 0,0636) **> 0,05**. Tek
   başına hiçbir etki büyüklüğünde anlamlı olamaz.

### Kaynak kovaryat olmalı mı? Hayır — tanımlanabilir değil

| | n | medyan pKd | ort. | std | min–max | ligand içi medyan aralık |
|---|---|---|---|---|---|---|
| Davis | 96 | 7,89 | 7,84 | 1,14 | 5,62 – 10,54 | **1,40** |
| BindingDB | 47 | 7,23 | 7,16 | 1,16 | 4,19 – 9,66 | **0,62** |

~0,65 log seviye farkı var, yayılımlar neredeyse aynı. Ama **hiçbir ligand iki kaynağı
karıştırmadığı için kaynak, ligand içinde yuvalı** — rastgele-kesişim modelinin ligand
terimiyle tam eşdoğrusal. Seviye farkı ligand kesişimi tarafından soğurulur ve ligand içi
eğimi yanlı hale **getiremez**; dolayısıyla `source`'u sabit etki olarak eklemek gereksiz.
Önemli olan ligand **içi** davranış farkı: BindingDB ligandlarının aralığı Davis'in yarısı
kadar ve provenance'ı daha ince. Bu, kaynak için **ayrı artık varyans** (ya da
ağırlıklandırma/katmanlama) gerektirir, kaynak kesişimi değil.

---

## 6. Bulunan ve bildirilen hatalar

Sessizce düzeltilen hiçbir şey yok; her biri raporlandı, karar kullanıcıya bırakıldı.

### Havuzlanmış testte istatistiksel hata (düzeltildi)

`pooled_power`, istatistiği null dağılımın **95. yüzdeliğiyle** karşılaştırıyordu;
permütasyon **p-değeri** hesaplanması gerekiyordu. Null sürekli olduğunda ikisi aynı sonucu
verir — büyük kapsamlar etkilenmedi (`pooled_ranking` üç ondalığa kadar birebir aynı,
`pooled_all` ≤0,001 fark) — ama **kaba null'da** ayrışırlar.

| kapsam | eski (yüzdelik) | yeni (p-değeri) |
|---|---|---|
| pooled_all (96), oran 2,0 | 1,000 | 1,000 |
| pooled_ranking (78), oran 1,0 | 0,996 | 0,996 |
| **negatif kontrol (18), oran 2,0** | **0,394** | **0,261** |
| 2 kompleksli kapsam, her oran | 1,000 *(saçma)* | **0** |

Uç vaka tanı koydurucu: tek 2-noktalı ligandın `|stat|`'ı her zaman tam 1'dir; yüzdelik de
1 olur, her çekim onu "geçer" ve güç 1,000 okunur — gerçek p-değeri ise 1, yani hiçbir şey
anlamlı olamaz. Düzeltme `tests/test_design_a_feasibility.py` (11 test) ile pinlendi: kaba
null'da iki yöntemin ayrıştığı, sürekli null'da örtüştüğü ve **her iki script'in** doğru
kuralı kullandığı (`@parametrize` ile `design_a` + `extended_set`) test ediliyor.
Aynı fonksiyonda boş kapsam çökmesi (`ValueError`) de bulundu ve guard eklendi.

**Bu düzeltmenin tezi etkileyen tek sayısı negatif kontrol gücüdür: 0,39 değil 0,26.**
Ranking kapsamları ve ≈0,6'lık tespit edilebilir en küçük etki değişmedi.

### Diğer bildirilenler

- 7 obsolete PDB ID (KLIFS hâlâ listeliyor), yerine geçenlerle birlikte raporlandı.
- 3 satırda `ligand_code` girişte yok (1P4F/DRG, 8AN8/N0U, 7OOV/6ID); 4'ünde benzer ama
  farklı kod (5N9N 8QK→KC5 gibi) — remap edilmedi, `ligand_missing` sayıldı.
- 1J6'nın JAK2/TYK2 konstrüktleri her iki domaini kapsıyor → `domain_ambiguous`.
- GCN2-b'nin pocket'ı UniProt domainine yerleştirilemedi.
- TYK2 JH2 serisi: BindingDB'de 9.016 satır / 2.978 ligand JH2 konstrüktlerinde ölçülmüş,
  ama yalnızca **3'ünün** manifestte yapısı var (ZRU, ZS3, ZSB).
- Tam protein ölçümleri, yapı kuralıyla JH2'ye atanıyor (ör. KZJ için tam boy TYK2 IC50'si
  → TYK2-b). Allosterik JH2 bağlayıcı gerçekten tam proteini inhibe eder, ama o değer bir
  enzim aktivitesi, JH2 bağlanma sabiti değil.

---

## 7. Şu anda elimizde ne var

### Kod

Hepsi aynı kalıpta: ağ katmanı ayrı, filtre/eşleme/istatistik mantığı **saf fonksiyonlar**,
testler ağsız, her script kendi `*_funnel.json`'ını provenance ile yazar.

| script | boyut | test |
|---|---|---|
| `scripts/klifs_manifest.py` | 27 KB | 36 |
| `scripts/bindingdb_activity.py` | 54 KB | 50 |
| `scripts/davis_activity.py` | 42 KB | 29 |
| `scripts/design_a_feasibility.py` | 24 KB | 11 (ortak) |
| `scripts/extended_set.py` | 29 KB | ↑ aynı dosyada `@parametrize` |

**193 test ağ erişimi olmadan geçiyor** (~25 s).

### Veri (`data/kinome/`, ~586 MB, git'te değil, DVC ile izlenir)

| klasör | boyut | içerik |
|---|---|---|
| `klifs/` | 9,3 MB | manifest (3.196 yapı), ham tablo, kinaz tablosu, funnel, RCSB cache |
| `bindingdb/` | 571 MB | 567 MB ham zip + provenance, 25.641 aktivite satırı, ligand kimlikleri, funnel, 2 cache |
| `davis/` | 5,8 MB | 3 orijinal `.xls` + provenance, 31.464 aktivite satırı, kinaz/bileşik haritaları, funnel, 4 cache |
| `analysis/` | 168 KB | Design A fizibilite (4 dosya), genişletilmiş set (4 dosya) |

### Analiz sonucu, tek paragrafta

Tasarım A **havuzlanmış** (mixed-model) analiz olarak yapılabilir ve frustrasyon indeksi
kendi artık gürültüsüne göre log birim başına ~0,6 birim hareket ediyorsa saptanır.
**Ligand başına** yalnızca STU, 1N1 ve DB8 testi tek başına taşıyabilir; 11 ligandda n=2
olduğu için hiçbir sonuç anlamlı olamaz. Birincil sıralama seti **18 ligand / 88 kompleks**,
negatif kontrol 21 / 48, keşifsel 3 / 7. BindingDB genişletmesi tespit edilebilir en küçük
etkiyi düşürmüyor; katkısı negatif kontrolü güçlendirmekte (0,26 → 0,53).

---

## 8. Açık kararlar

1. **8X7, 537, YAM** şimdilik `exploratory`. Birincil sıralama setine alınmamaları
   onaylanacak mı, yoksa tek-rapor uçları kabul edilip `ranking`'e mi dönecekler?
2. **Negatif kontrol gücü hâlâ düşük** (oran 2,0'da 0,53). Null sonucu kanıt sayabilmek
   için ya daha fazla dar-aralıklı ligand ya da eşdeğerlik testi (TOST) çerçevesi gerekir.
3. **TYK2 JH2 serisi** ayrı bir alt çalışma olabilir: 2.978 ligand ölçülmüş, 3'ünün yapısı
   var. Yapı tarafı genişletilirse allosterik bölge için bağımsız bir test seti çıkar.
4. **Tam protein ölçümlerinin domain ataması** (madde 6'nın son maddesi) gözden geçirilmeli
   mi — JH2 bağlayıcılar için tam boy IC50'leri JH2'ye atamak doğru mu?
5. **İndirme adımı henüz yazılmadı.** Manifest `pdb`/`chain`/`altloc`/`ligand_code`
   taşıyor; `ligand_chain_differs` (20) mmCIF, `multi_copy_ligand` (75) cep seçimi,
   `altloc_ligand_conflict` (8) altloc seçimi gerektirir.
