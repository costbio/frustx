# The method FrustX implements

Source: Chen M, Chen X, Schafer NP, Clementi C, Komives EA, Ferreiro DU, Wolynes PG.
**"Surveying biomolecular frustration at atomic resolution."**
*Nat Commun* 11:5944 (2020). PMID 33230150 / PMC7683549 / doi:10.1038/s41467-020-19560-9.

Quotations below are verbatim from the paper's Methods.

## 1. Frustration index (Eq. 1)

```
F°_ij = ( E⁰_ij − E^U_{i'j'} ) / σ( E^U_{i'j'} )
```

`E⁰_ij` is the contact energy of residues i,j in the native sequence; `E^U_{i'j'}` is the
same quantity over the decoy ensemble. A Z-score: the native contact energy measured
against the mean and spread of what that same geometric site would give for random
sequence.

## 2. Contact energy (Eq. 2) — the many-body term

```
E_ij = e_ij + ½ Σ_{k≠j} e_ik + ½ Σ_{l≠i} e_jl
```

Not the bare pair energy. `e_ij` is the direct interaction, plus **half of every other
interaction involving i**, plus **half of every one involving j**. Per the paper:

> "Since the fully atomistic force field is a many-body construct, the pairwise energy
> change assigned to forming a contact between residue i and j, Eij is defined by
> considering all the interaction energies that involve changing any of the two residues
> that are in contact."

> "In addition to the direct interactions between i and j (defined as eij), the addition
> of the background interactions accounts for the many-body effects elicited by local
> side-chain reconfigurational changes."

**This is the step the earlier `FrustX.py` prototype was missing** — it used the total
system potential energy, in which a single side-chain change is far below the noise.

## 3. Energy function

**Rosetta REF2015**, all-atom. *"we employed the REF2015 version of the rosetta energy
function, which has a set of well-tested weights for each energy term."*

**The repulsive Lennard-Jones term is removed** when computing `e_ij`:

> "Even after relaxation, the repulsive Lennard–Jones interactions used in computing eij
> give rise to very strong fluctuating clashes... This gives rise to an artificially large
> variance of the decoy energies. These repulsive clashes would make the algorithm too
> sensitive for detecting functionally relevant frustration... it is both easy and
> appropriate to simply remove the harsh rapidly varying repulsive force term."

Keeping `fa_rep` in gives a distinct quantity the authors name **packing frustration**,
which *"turns out to be good at diagnosing the health of a given high-resolution
structure (i.e., predicted structure)"* — i.e. useful for judging predicted models.

## 4. Decoy generation

> "the decoy ensemble of residue contacts is obtained by randomizing both the residue
> identities and locations of the residues in contact"

> "we randomly shuffle the protein sequence and then repack the resulting sequence onto
> the backbone that is provided without perturbing the backbone coordinates within each
> protein chain to make sure that only the side chains are re-packed. A short Monte-Carlo
> relaxation is then performed to better eliminate many of the possible side-chain clashes
> with the backbone fixed."

Native reference energies: *"obtained in a similar fashion by omitting the shuffling
step."*

Key consequences for implementation:

- **Shuffle, not mutate.** A random permutation of the native sequence automatically
  satisfies *"sequence space is randomly sampled according to the native amino acid
  frequency distribution"* — composition is preserved for free.
- **Backbone stays fixed throughout.** Side chains only.
- **1000 decoys per contact.** One shuffle+repack yields an energy for *every* contact at
  once, so this is ~1000 shuffle-repack cycles for the whole protein, not
  1000 × n_contacts.

## 5. Contact definition

> "Protein contacts are defined by the CαCα distances between residues, and a cutoff of
> 10 Å is used."

Implemented in `frustx/contacts.py`.

## Open questions to settle empirically (Phase 6)

1. **Sign convention.** Eq. 1 as written makes a minimally frustrated contact *negative*
   (native energy below the decoy mean). `frustratometeR` reports minimally frustrated as
   `F > 0.78`. One of the two flips a sign; resolve by running both on the same protein
   rather than by guessing.
2. **Sequence separation.** The Methods state a distance cutoff and nothing else, so we
   default to keeping every pair (`min_seq_sep=1`). Other frustration tools exclude
   near-sequence neighbours, whose contact is dictated by covalent geometry rather than by
   packing. On ubiquitin the very first contact found is MET1–GLN2 at 3.8 Å, which shows
   the question is not academic. Settle against reference output.
3. **Repack radius during decoy generation.** "Repack the resulting sequence onto the
   backbone" reads as a global repack; whether restricting to a shell around each contact
   changes the answer is worth measuring, since it dominates runtime.

## Deliberately out of scope for now

- Protein–ligand / binding-pocket frustration (the paper's EGFR and COX sections).
- OpenMM as an alternative energy backend. It cannot reproduce these numbers — different
  force field, different decoy semantics. It is a separate experiment, not a prerequisite.
- Packaging, Docker, web server.

---

# Implementation notes: extracting `e_ij` from Rosetta

Verified empirically on 1UBQ (ubiquitin, 76 residues) with PyRosetta
2026.32 / REF2015. Probe scripts were run in scratch, not committed; the numbers below
are reproducible from a bare `pose_from_pdb` + `get_score_function()`.

## The naive energy graph is NOT the whole energy

Eq. 2 assumes the energy decomposes into pairwise `e_ij`. Reading only the `EnergyGraph`
edges plus one-body terms **silently loses 49.26 REU** on ubiquitin:

| term | weighted energy off the graph |
|---|---|
| `hbond_lr_bb` | −23.13 REU |
| `hbond_sr_bb` | −18.83 REU |
| `rama_prepro` | −7.30 REU |

Losing the backbone–backbone hydrogen bonds would be fatal: they are exactly the
interactions that stabilise secondary structure, i.e. the thing frustration analysis is
supposed to be measuring.

## Fix 1 — put bb–bb hbonds on the graph

```python
opts = sf.energy_method_options()
opts.hbond_options().decompose_bb_hb_into_pair_energies(True)
sf.set_energy_method_options(opts)
```

This is a score-function option, **not** a command-line flag (`-score:decompose_bb_hb...`
does not exist and raises at init). Control: the weighted total is 32.6778 REU with the
option on *and* off — it redistributes where energy is stored, it does not change the
physics.

## Fix 2 — read `rama_prepro` from its long-range container

`rama_prepro` lives in a `PolymerBondedEnergyContainer`
(`LongRangeEnergyType.ramaprepro_lr`), reachable via
`pose.energies().long_range_container(...)`. On ubiquitin it yields 75 pair records, all
strictly (i, i+1), summing to the pose total exactly (−16.2164, difference 0.0000).

Iterate with `it.pre_increment()` — the PyRosetta binding has no `.next()`.

**`rama_prepro` cannot be ignored as sequence-independent.** With the backbone held fixed,
single mutations at V5 of ubiquitin move it by:

| mutation | Δ rama_prepro |
|---|---|
| V5A | +2.89 |
| V5W | +2.58 |
| V5G | +10.35 |
| V5P | +12.51 |

So it does not cancel in the Z-score under sequence shuffling. It is a genuine (i, i+1)
two-body term and Eq. 2 can carry it.

## Result: the decomposition is exact

`one-body + EnergyGraph edges + long-range containers` reproduces the REF2015 total to
**0.00e+00 REU**. Eq. 2 is exactly implementable with no unaccounted energy.

Sweep long-range containers generically rather than hard-coding `ramaprepro_lr`:
`dslf_fa13` appears the same way for disulfide-bonded proteins, and ubiquitin has no
disulfides so it would not have shown up in this test.

## Caveat on raw input structures

A crystal structure scores **+32.7 REU** (unfavourable) before any relaxation, because
`fa_rep` sees clashes that crystallographic refinement tolerates. Never score a bare input
PDB and treat the number as meaningful — this is why the paper specifies a short
Monte-Carlo relaxation, and it previews the `fa_rep` problem solved by deleting the term.

## Validation of the extraction on a real structure

1UBQ, REF2015 with `fa_rep` removed as the paper specifies:

- **Completeness holds on real input**: one-body + pairwise reproduces the REF2015 total
  to 6e-13 REU.
- **The 10 Å Cα cutoff is well chosen**: −323.27 REU of pair energy falls inside the
  contact set, only −10.32 REU outside it. The cutoff captures ~97% of pairwise energy.
- **Magnitudes are sane**: mean e_ij = −0.72 REU over contacts, range −6.26 to +1.24.
- **The strongest contacts are the right ones.** The five most favourable are
  ILE44–HIS68, VAL5–ILE13, MET1–VAL17, ILE3–LEU15, ARG42–VAL70 — the ubiquitin
  hydrophobic core, and Ile44/His68/Val70 are the canonical Ile44 patch that nearly every
  ubiquitin receptor binds. An independent check that the extraction is physically
  meaningful, not merely self-consistent.

### Open question

98 of 550 contacts (18%) have exactly zero e_ij: their Cα atoms are within 10 Å but no
atom pair falls inside Rosetta's interaction cutoff. Whether these should enter the decoy
statistics as genuine zeros or be excluded is not addressed by the paper. They will have
zero variance across decoys where the residues stay far apart, which risks a divide-by-zero
in Eq. 1 — this needs an explicit decision when the frustration index is implemented.

---

# Eq. 2 simplifies: the direct contact term cancels

Expanding Eq. 2 with `R_i = Σ_k e_ik` (residue i's total interaction energy, diagonal
zero):

```
Σ_{k≠j} e_ik = R_i − e_ij
Σ_{l≠i} e_jl = R_j − e_ij

E_ij = e_ij + ½(R_i − e_ij) + ½(R_j − e_ij) = ½ (R_i + R_j)
```

**The `e_ij` term cancels exactly.** Verified numerically against a literal double-loop
transcription of Eq. 2 on 1UBQ: max difference 7.1e-15.

The cancellation is robust to the ambiguity in the summation ranges. Whether `k` runs
over all residues except `j`, or over all except both `i` and `j`, the result is the same,
because `e_ii = 0`.

## What this means

`E_ij` depends only on the two residues' *total* interaction energies — not on how
strongly they touch each other. The paper's contact-level frustration index is therefore
a **pair average of a residue-level quantity**. The contact set enters only by selecting
which (i, j) pairs get reported, not by weighting the direct interaction between them.

This follows from the paper's own stated intent — *"considering all the interaction
energies that involve changing any of the two residues that are in contact"* — so it
appears deliberate rather than a transcription error. It is worth confirming against
`frustratometeR`, whose AWSEM configurational index does not share this property.

Practical consequences:

- **Fast.** Row sums, not a double loop over pairs.
- **The zero-`e_ij` worry is void.** The 98 of 550 ubiquitin contacts with `e_ij = 0`
  raised a divide-by-zero risk in Eq. 1. Since `E_ij` never uses `e_ij`, they get finite
  energies and finite decoy variance like any other pair. The earlier open question is
  closed.

---

# Sign convention: FrustX negates the paper's Eq. 1

The paper writes `F = (E0 − ⟨E_U⟩)/σ`, which makes a **minimally** frustrated contact
come out **negative** — the native energy sits below the decoy mean. Measured on
ubiquitin, the literal formula gives a range of −4.05 to +0.55, with a mean of −0.84:
nearly everything negative, as expected for a well-folded protein.

Every other tool in the field reports the opposite sign. `frustratometeR` classifies
minimally frustrated as `F > 0.78` and highly frustrated as `F < −1`.

**FrustX negates**, so output matches the field:

| FrustX `frustration_index` | meaning |
|---|---|
| high positive | minimally frustrated — native much better than random |
| near zero | neutral |
| negative | highly frustrated — native no better than random |

`tests/test_frustration.py` guards this: a designed native must beat shuffled decoys and
therefore score positive. Anyone "correcting" the sign back to the paper's literal form
breaks that test.

## Classification thresholds are borrowed; the fractions differ, the cut point survives

`MINIMALLY_FRUSTRATED = 0.78` and `HIGHLY_FRUSTRATED = −1.0` are taken from
frustratometeR so output is comparable. **They were calibrated on the AWSEM
coarse-grained energy function, not on atomistic REF2015.** Measured on 1UBQ
(`frustx_min500`, 458 contacts with a defined index; the 17 σ = 0 contacts of the 475 are
excluded — see the cutoff-edge artifact below):

| index distribution | mean | sd | p5 | p25 | med | p75 | p95 | min | max |
|---|---|---|---|---|---|---|---|---|---|
| FrustX (REF2015) | +0.204 | 0.875 | −0.58 | −0.23 | −0.01 | +0.39 | +1.81 | −3.30 | +5.97 |
| fR configurational | +0.408 | 1.065 | −1.31 | −0.30 | +0.38 | +0.98 | +2.39 | −2.75 | +2.89 |
| fR mutational | +0.484 | 1.056 | −1.16 | −0.48 | +0.57 | +1.34 | +2.04 | −1.87 | +2.43 |

The standard deviations are similar, but the *shapes* are not. FrustX's interquartile
range is 0.62 against 1.28 and 1.82 for the two reference modes, while its extremes run
further in both directions. FrustX's index is sharply peaked near zero with long tails;
the AWSEM indices are broad and comparatively light-tailed. A cut point that lands in the
shoulder of one distribution lands in the far tail of the other.

The class fractions the borrowed cut points produce:

| | minimally (≥ +0.78) | neutral | highly (≤ −1.0) |
|---|---|---|---|
| FrustX (REF2015) | 17.5% | 80.3% | 2.2% |
| fR configurational | 33.4% | 58.9% | 7.7% |
| fR mutational | 42.4% | 48.8% | 8.7% |

FrustX assigns ~2x fewer contacts to the minimally frustrated class and ~4x fewer to the
highly frustrated class, leaving four contacts in five neutral. Read on its own this looks
like a miscalibration; the test two subsections below shows it is not, at least on the
minimally frustrated side. Recorded here because it is what motivated the investigation.

### Percentile matching is the obvious fix and it is wrong

Forcing FrustX to call the same *fraction* of contacts into each class as the reference
requires these cut points:

| target | minimally ≥ | highly ≤ |
|---|---|---|
| fR configurational fractions (33.4% / 7.7%) | +0.160 | −0.475 |
| fR mutational fractions (42.4% / 8.7%) | +0.060 | −0.452 |

A threshold of +0.06 would label any contact marginally better than its decoy mean
"minimally frustrated". That is not a calibration, it is a rank cut wearing a Z-score's
clothes. The rule fails because the reference fractions are not ground truth:
configurational's fraction is a *raw energy* cut in disguise (its index is the energy
rescaled by two constants — see below), and mutational's is a cut on a quantity that is
95% residue-additive. Neither is a per-contact frustration frequency worth reproducing.

### Testing the cut point on its own terms

FrustX's index is a genuine per-contact Z-score, so a cut at Z = 0.78 has a self-contained
meaning — *fewer than ~22% of random substitutions at this position pair do this well* —
that needs no AWSEM calibration. But that is a normal-distribution statement, and the decoy
energies had never been checked for normality: production keeps only running sums.
`scripts/dump_decoy_samples.py` retains the full per-decoy tensor so the shape can be
tested; `scripts/test_decoy_normality.py` runs the test. Below: 1UBQ, 200 `min` decoys,
`min_seq_sep = 2`, 475 contacts (456 with σ > 0), matching `frustx_min500/run.json`.

**The decoy energies are emphatically not normal.** Median excess kurtosis +10.3
(p95 +183.9), median skewness −1.04, and a D'Agostino–Pearson K² omnibus test rejects
normality for **98.0%** of contacts at p < 0.05. This is not subtle: the repack occasionally
finds a far better or far worse arrangement than typical, and those outliers dominate the
higher moments.

The K² implementation is hand-written (scipy is not a dependency) and was checked against
distributions with known answers before use: 5.6% rejection on normal input at α = 0.05,
100% on exponential (skew +1.88) and on uniform (excess kurtosis −1.18, true −1.2).

**Despite that, Z still tracks the empirical decoy tail on the minimally frustrated side.**
For each contact the empirical tail is the fraction of decoys with energy at or below the
native — the honest answer to "how many random substitutions do this well":

| Z band | n | empirical tail p10 | median | p90 | normal predicts |
|---|---|---|---|---|---|
| 0.4 – 0.6 | 24 | 13.1% | 24.5% | 34.1% | 30.9% |
| 0.7 – 0.9 | 13 | 10.2% | 15.5% | 20.4% | 21.2% |
| 1.0 – 1.2 | 11 | 6.5% | 13.5% | 18.5% | 13.6% |
| 1.4 – 1.6 | 11 | 5.0% | 6.5% | 8.5% | 6.7% |

At Z ≥ 1.0 the median empirical tail sits within 0.2 points of the normal prediction. The
heavy tails inflate the higher moments without displacing the median much, which is why a
formal normality test can fail overwhelmingly while the practical Z-to-tail mapping holds.
The p10–p90 spread is real — contacts sharing a Z differ by roughly ±8 points in actual
tail — so Z is a noisy classifier, but not a meaningless one.

**So the earlier "measurably wrong" verdict is withdrawn for the minimally frustrated
threshold.** At Z = 0.78 the empirical tail is ~15%: fewer than one random substitution in
six does as well as the native. That is a coherent definition of "minimally frustrated" on
its own terms. The 17.5%-vs-33% fraction gap against AWSEM is therefore *not* evidence of a
misplaced cut point — it says FrustX and frustratometeR disagree about how many contacts of
ubiquitin are strongly determined, which is a substantive difference between an atomistic
and a coarse-grained energy function, not a calibration error.

**The highly frustrated threshold remains unresolved.** Only 10 contacts reach Z ≤ −1.0, too
few to calibrate against. Worse, inverting the map gives no clean ordering on that side: an
empirical tail of 80% implies Z ≈ −0.42 while a tail of 90% implies Z ≈ −0.29, i.e. the
worse contacts map to the *higher* Z. With n = 15 and n = 20 and IQRs spanning ~0.5 that
inversion may be noise, but nothing here supports −1.0 either.

**Decision: both constants stay as they are.** `0.78` is now supported rather than merely
borrowed. `−1.0` is retained for want of evidence to move it, and should be treated as
provisional in any result that depends on the highly-frustrated class.

### What this rests on

One protein, 200 decoys, and Z bands holding 11–13 contacts each. The direction of the
result is clear but the numbers in the table are not precise. Repeating it on two or three
more structures is the obvious next step, and is cheap: ~13 min per protein at this size.

# Known artifact: glycine positions read as minimally frustrated

Whole-sequence shuffling drops bulky residues onto positions where only glycine fits
sterically. Decoy energies there balloon, and the native glycine looks superb by
comparison.

On ubiquitin the C-terminal LRGG tail lands among the *most* minimally frustrated
contacts:

| contact | F (conventional sign) | E0 | ⟨E_decoy⟩ |
|---|---|---|---|
| GLY75–GLY76 | +3.66 | +1.0 | +8.5 |
| ARG74–GLY76 | +3.45 | −0.4 | +4.4 |
| LEU73–GLY76 | +2.96 | −1.5 | +3.3 |

That tail is flexible, solvent-exposed, and the functional conjugation site. It should
read as *frustrated*. What the index is actually reporting is "glycine fits here and
bulky residues do not" — a composition-and-sterics effect, not folding frustration.

**CORRECTED after validation.** This was originally recorded as "inherent to the paper's
shuffling protocol rather than to this implementation". The frustratometeR comparison
disproves that. On the same structure, over 344 shared contacts:

| | GLY contacts (median index) | non-GLY |
|---|---|---|
| FrustX | **+0.39** | +0.55 |
| frustratometeR | **−0.41** | +0.40 |

frustratometeR classifies glycine contacts as *frustrated*; we classify them as
*minimally frustrated*. The signs are opposite, not merely different in magnitude
(PHE45–GLY47: ours +0.57, theirs −0.50).

So this is a defect on our side, not a property of the published protocol. It is very
likely a symptom of the Eq. 2 degeneracy documented below rather than an independent bug:
an index reducible to per-residue burial will read a small, well-fitting glycine as
optimal regardless of what it actually interacts with.

---

# Validation against frustratometeR (1UBQ)

Setup: identical input PDB (frustratometeR's own prepared `1ubq_A.pdb`, 76 residues),
FrustX at 1000 decoys with `--protocol min --min-seq-sep 2` to match their contact set.
frustratometeR at `Mode="configurational"`, defaults.

Note `SeqDist=12` does **not** mean minimum |i−j| ≥ 12, as its name suggests. Their
output contains contacts at |i−j| = 2 (74 of them). `SeqDist` selects the AWSEM
`Welltype` classification — `short` / `long` / `water-mediated` — via the two bundled
LAMMPS binaries. Their effective minimum sequence separation is 2.

## Result: weak agreement

```
FrustX 475 contacts, frustratometeR 389, shared 344
Spearman +0.159   Pearson +0.133   class agreement 56.7%

                 min      median    max      sd
FrustX          -0.18     +0.53    +2.84    0.43
frustratometeR  -2.75     +0.37    +2.89    1.07
```

FrustX finds **zero** highly frustrated contacts in ubiquitin (minimum −0.18, threshold
−1.0). frustratometeR finds 30. A frustration index that never reports frustration is
not functioning.

## Diagnosis: the Eq. 2 cancellation is the cause

Predicting each contact index from the mean of its two residues' indices:

| | ρ |
|---|---|
| **FrustX** | **+0.865** |
| frustratometeR | +0.464 |

Our contact index is 87% reducible to a residue-level quantity; theirs is not. This is
the algebraic cancellation showing up in data: `E_ij = ½(R_i + R_j)` contains no term
describing the specific i–j interaction, so contact-level structure cannot survive.

Corroborating evidence:

- Our per-residue values correlate **better** with their *single-residue* index
  (ρ = +0.348) than our contact values do with their *configurational* index
  (ρ = +0.159). We built a single-residue measure while intending a contact measure.
- By AWSEM well type: ρ = +0.32 (`long`), +0.27 (`short`), but **−0.05
  (`water-mediated`)** — 46% of contacts. Water-mediated contacts are exactly those whose
  energy depends on the specific pair rather than on burial.

## Caveat

AWSEM is not ground truth, and the paper's premise is that atomistic and coarse-grained
frustration differ. Weak correlation alone would not condemn the implementation. But
"no frustrated contacts at all" and "87% reducible to residue identity" are internal
red flags independent of whether frustratometeR is right.

## Experiment: does the Eq. 2 background explain the disagreement?

Two sweeps, 1000 decoys each, same contacts, `protocol="min"`.

**Sweep 1** — `E_ij = ½(R_i + R_j) + α·e_ij` for α ∈ {0, +1, −½}, the three defensible
readings of Eq. 2's summation ranges. All three were indistinguishable
(ρ = 0.147–0.170, zero frustrated contacts, pair-specificity 0.86). The reason is
magnitude: `e_ij` ≈ −0.7 REU against `½(R_i + R_j)` ≈ −10 REU, so every α is a small
perturbation on a dominant term. **The cancellation was real but never the binding
constraint.**

**Sweep 2** — `E_ij = e_ij + w·½(R_i + R_j)`, sweeping the background weight. `w = 1` is
Eq. 2 as written; `w = 0` is the direct pair energy alone, which is structurally what
AWSEM/frustratometeR uses.

```
variant       rho     min   median    max     sd   n<-1   GLYmed   pair-spec
frustratometeR --   -2.75    0.37    2.89   1.07     30    -0.41     +0.464
w=0.0      +0.128   -3.15    0.04    6.17   0.92     11    -0.29     +0.438
w=0.05     +0.176   -1.62    0.55    5.87   0.68      2    +0.25     +0.637
w=0.1      +0.172   -0.80    0.56    5.41   0.59      0    +0.35     +0.716
w=0.25     +0.160   -0.39    0.56    4.17   0.50      0    +0.41     +0.802
w=0.5      +0.157   -0.16    0.56    3.11   0.46      0    +0.43     +0.839
w=1.0      +0.159   -0.11    0.55    2.82   0.44      0    +0.43     +0.855
```

### Two independent conclusions

**1. The background term is what makes the index degenerate.** At `w = 0` every pathology
disappears and the statistical character matches frustratometeR closely:

| | FrustX `w=0` | frustratometeR |
|---|---|---|
| pair-specificity | +0.438 | +0.464 |
| spread (sd) | 0.92 | 1.07 |
| minimum | −3.15 | −2.75 |
| contacts below −1 | 11 | 30 |
| glycine median | **−0.29** | **−0.41** |

The glycine artifact **reverses sign and disappears** at `w = 0`. It was a symptom of the
background term, exactly as suspected — confirming the correction recorded above.

**2. None of this explains the disagreement with frustratometeR.** ρ stays at 0.13–0.18
across the entire sweep, while pair-specificity moves from 0.44 to 0.86. The correlation
is *insensitive to the whole Eq. 2 question*. Whatever drives the disagreement is not the
contact-energy definition.

So: dropping the background gives an index with the right shape, the right spread, real
frustrated contacts and correct glycine behaviour — but it still ranks individual contacts
differently from AWSEM. Right distribution, different assignment.

### What this does not settle

ρ ≈ 0.15 at n = 344 is statistically significant (p ≈ 0.005) but weak. Remaining
candidates, in the order worth testing:

1. **Decoy locality.** We shuffle the entire sequence, so a decoy energy for contact
   (i,j) depends on every other position too. AWSEM's decoys change at most two residue
   identities. The atomistic paper does specify whole-sequence shuffling, but the two are
   not statistically equivalent.

   > **CORRECTION (this sentence was wrong as first written).** It previously said the
   > configurational decoy "randomises the identities and geometry *of the contacting
   > pair*, holding the rest native". Neither half is right. Read from
   > `results/reference/awsem_source/fix_backbone.cpp:5256-5271`, the configurational
   > decoy does not perturb the pair's geometry — it **replaces** it, drawing `r_ij` from
   > a randomly chosen *different* pair elsewhere in the protein that happens to be in
   > contact, and then drawing `rho_i` and `rho_j` from a **second, independent** random
   > pair (so the distance and the two burial densities do not even come from the same
   > place). What is held native is the rest of the *sequence*, not the geometry. The
   > accurate contrast is in "What the three modes actually randomise" below.
2. **Relaxation.** These runs used `protocol="min"`. The default `"relax"` may change the
   decoy spread substantially, and σ is Eq. 1's denominator.
3. **Genuine method difference.** REF2015 and AWSEM are different force fields, and the
   paper's premise is that atomistic resolution reveals what coarse-graining cannot. Low
   correlation is not by itself proof of a bug.

> **All three were tested, and the conclusion drawn from those tests was wrong.** See the
> next section: the comparison itself was not measuring what it appeared to measure,
> because frustratometeR's configurational index is not a per-contact Z-score.

---

# The reference implementation: frustratometeR's configurational index is a rescaled energy, by design

Everything in the two sections above compares our index against a quantity we had
misidentified. This section records what frustratometeR actually computes, read from its
source, and what that does to the comparison. It supersedes the water-mediated conclusion
previously recorded here.

## The decoy distribution is computed once per protein, not once per contact

From the AWSEM source that builds the LAMMPS binary frustratometeR ships
(`Scripts/lmp_serial_12_Linux`), with the authors' own comment:

```c
// the configurational decoy statistics only need to be computed once because they are
// the same for every contact
if (!strcmp(tert_frust_mode, "configurational")==0
    || (strcmp(tert_frust_mode, "configurational")==0 && !already_computed_configurational_decoys)) {
  compute_decoy_ixns(i_resno, j_resno, rij, rho_i, rho_j);
}
```

`fix_backbone.cpp:5095-5101`; the flag is set at `:5341`. Provenance is not inferred — the
shipped binary contains the symbol `already_computed_configurational_decoys`.

This is directly visible in the reference output. All 389 contacts of 1UBQ carry an
identical decoy mean and standard deviation:

```
$ awk 'NR>1{print $10,$11}' 1ubq_A.pdb_configurational | sort | uniq -c
    389 -1.517 0.495
```

Therefore `FrstIndex = (−1.517 − E_native)/0.495`, an affine function of the native
energy. Measured: **Spearman(FrstIndex, NativeEnergy) = −0.999998**, the residual being
three-decimal rounding in the output file.

**Their configurational index carries no per-contact decoy information.** It is the raw
AWSEM contact energy, sign-flipped and rescaled by two constants.

Two further properties of their decoy construction, for the record:

- **No decoy structure is ever built.** `compute_decoy_ixns` (`fix_backbone.cpp:5249-5344`)
  is a 1000-iteration loop of `rand()` draws into the *native* distance, density and
  identity pools, followed by analytic lookups into the frozen `gamma.dat` /
  `burial_gamma.dat` tables. No packer, no minimiser, no steric term, no coordinate write.
  Confirmed independently against the shipped binary's disassembly.
- **The draws are unseeded.** `rand` is imported; `srand` is not called anywhere. Their
  reference numbers are bit-identical on every run, so all sampling noise in every
  comparison in this document has been on the FrustX side.

## Why this is correct, not an optimisation bug

The compute-once guard is not a shortcut that happens to be safe. It is forced by the
definition of the configurational decoy. From the body of `compute_decoy_ixns`
(`fix_backbone.cpp:5255-5277`):

```c
if (strcmp(tert_frust_mode, "configurational")==0) {
  // choose random rij, rho_i, rho_j
  rand_i_resno = get_random_residue_index();
  rand_j_resno = get_random_residue_index();
  rij = get_residue_distance(rand_i_resno, rand_j_resno);
  while (rij > tert_frust_cutoff || rand_i_resno == rand_j_resno) { ... }
  // get new pair of random residues for burial term
  rand_i_resno = get_random_residue_index();
  rand_j_resno = get_random_residue_index();
  rho_i = get_residue_density(rand_i_resno);
  rho_j = get_residue_density(rand_j_resno);
}
else {
  // if in mutational mode, use configurational parameters passed into the function
  rij = rij_orig;  rho_i = rho_i_orig;  rho_j = rho_j_orig;
}
```

Note what configurational does with its arguments: the function is *handed* this contact's
native distance and densities (`rij_orig`, `rho_i_orig`, `rho_j_orig`) and discards them,
redrawing all three from protein-wide pools. Nothing about contact (i, j) survives into the
decoy. The distribution therefore cannot depend on the contact, and computing it once per
protein is the only self-consistent implementation.

Mutational takes the other branch: geometry pinned to the native contact, only the residue
identities randomised. That is why mutational has 389 distinct (mean, sd) pairs and
configurational has one.

The two modes ask different questions, and both are legitimate:

| Mode | Question | Yardstick |
|---|---|---|
| configurational | Is this contact better than a randomly assembled contact *anywhere in this protein*? | protein-wide, identical for every contact |
| mutational | Is this contact better than *other amino acids at this same geometry*? | contact-specific |
| FrustX (Eq. 1) | Is this contact better than other amino acids at this same geometry, *with side chains repacked and scored at all-atom resolution*? | contact-specific |

FrustX's decoy is a mutational decoy, not a configurational one. Every comparison in the
two superseded sections above was therefore cross-mode as well as cross-resolution, which
is a second reason the numbers were never going to line up.

## Consequence: the headline correlation was never index-vs-index

ρ = +0.141 compared *our* genuine per-contact Z-score against *their* raw energy. Putting
both sides on the same footing — replacing our per-contact σ with one global (mean, sd),
as theirs effectively is:

```
rho(our index,             their index) = +0.1407   p = 0.009
rho(our index, GLOBALISED, their index) = −0.0303
rho(our E0,                their E0   ) = −0.0303
```

**−0.030 is the commensurable number.** The +0.141 is an artifact of comparing objects of
different kinds.

One qualification, because the aggregate figure misleads in the other direction too:
within a well type, globalising barely changes anything (`long` +0.346 → +0.384, `short`
+0.314 → +0.249). The aggregate collapse is a mixing effect across well types, not the
denominator doing all the work.

## Consequence: the numerators are different quantities, and this kills the water-mediated story

frustratometeR's per-contact energy is `water_ij + burial_i + burial_j`
(`fix_backbone.cpp:5459-5500`). Burial is a **one-body** quantity: it depends only on
(residue identity, local density) and is therefore identical across every contact a given
residue makes. FrustX's `E_ij` at `w = 0` is a bare two-body pair energy with all one-body
terms excluded by construction (`frustx/energies.py:105-109`).

Fitting `E ~ c + a_i + a_j` — the exact functional form of `burial_i + burial_j`:

| | additive R² |
|---|---|
| frustratometeR E₀ | **0.560** |
| FrustX e_ij (`w = 0`) | **0.133** |

Over half their variance is one-body. Residualising that component out of **both** sides:

| AWSEM well type | n | ρ(E₀, E₀) raw | ρ one-body removed |
|---|---|---|---|
| `long` | 71 | +0.382 | **−0.053** |
| `short` | 115 | +0.249 | **+0.070** |
| `water-mediated` | 158 | −0.053 | −0.064 |
| all | 344 | −0.030 | −0.178 |

**The direct-contact agreement was carried entirely by a shared one-body burial term.**
Once it is removed, the two-body contact physics agrees at approximately zero in every
well type, water-mediated included.

This refutes the explanation previously recorded in this section. The well-type split is a
real and reproducible observation, but "the disagreement is concentrated in water-mediated
contacts" is false: the disagreement is everywhere, and direct contacts merely *looked*
better because both models encode the same burial information. The previous subsection
here was titled "Why this is the expected result, not a defect" — it assumed its
conclusion and has been deleted.

## Also not comparable: electrostatics

`huckel_flag` initialises to `0` (`fix_backbone.cpp:122`) and is set only on an exact
match against `"[DebyeHuckel]"` (`:467`). The shipped parameter file's section header is
`[DebyeHuckel]-`, with a trailing dash — AWSEM's convention for a disabled section — so
the match never fires. **The reference energy contains no electrostatic term at all**,
while FrustX's `e_ij` includes `fa_elec`.

## Status of the earlier candidate list

- **Candidate 1, decoy locality — not ruled out.** The pair-local experiment moved ρ
  negligibly (+0.265 vs +0.273) but nearly doubled the separation between the contacts
  frustratometeR calls frustrated and those it calls minimally frustrated
  (`scripts/local_decoy.py`). Given that their configurational σ is a single global
  constant, ρ was never the statistic that could settle this. Reopened.
- **Candidate 2, relaxation — still ruled out.** The matched `min`/`relax` runs below
  remain valid; they are a FrustX-internal comparison and do not depend on what the
  reference computes.
- **Candidate 3, genuine method difference — untested.** It cannot be assessed until the
  numerator mismatch is removed, because no frustratometeR mode has a bare-pair numerator
  to compare against.

## What is established, and what is not

Established, from source and from data already in `results/validation/`: their
configurational index is affine in their native energy; their decoys are analytic and
structureless; the commensurable correlation is −0.030; the direct-contact agreement was
a one-body artifact; their reference energy omits electrostatics.

**Not** established: whether FrustX is correct. Nothing here is evidence either way. The
comparison that was doing the validating has been withdrawn, not replaced. The next
informative tests are frustratometeR's `mutational` mode — the only per-contact mode with
a genuinely varying σ — and, independently, reproducing a specific published figure from
Chen et al. (2020).

---


---

# E3/E4: the other frustratometeR modes, and what they reveal

Run after the section above, with predictions registered in advance. **The E3 prediction
was wrong**, and the way it was wrong is the most informative result so far.

Reproduce with:

```bash
Rscript -e "library(frustratometeR); calculate_frustration(
  PdbFile='/workspaces/frustx/results/validation/1ubq.pdb', Chain='A',
  Mode='mutational', Graphics=FALSE, Visualization=FALSE,
  ResultsDir='/workspaces/frustx/results/validation/frustra_mutational/')"
```

`ResultsDir` must be **absolute**: `calculate_frustration` does `setwd(JobDir)` at
`calculate_frustration.R:64` and then keeps using the caller's relative path, which no
longer resolves. Fails with "cannot open the connection" on a relative path.

## E3: mutational mode has a real per-contact sigma, and correlates twice as well

Unlike configurational, mutational does not hit the compute-once short-circuit. All 389
contacts get distinct decoy statistics (389 unique `(DecoyEnergy, SDEnergy)` pairs, sigma
from 0.500 to 8.203), and `rho(FrstIndex, NativeEnergy) = -0.88` rather than -1.0.

| target | n | rho | p |
|---|---|---|---|
| configurational | 344 | +0.141 | 0.009 |
| **mutational** | 344 | **+0.322** | **1e-9** |

Per well type, and this is the part that matters:

| well type | n | vs configurational | vs mutational |
|---|---|---|---|
| `short` (direct) | 115 | +0.314 | +0.397 |
| `long` (protein-mediated) | 71 | +0.346 | +0.346 |
| `water-mediated` | 158 | **-0.030** (p 0.71) | **+0.223** (p 0.005) |

**The water-mediated non-correlation was an artifact of the constant sigma, not physics.**
Against a target with a real per-contact denominator it is positive and significant. The
prediction registered before this run was "rho stays below +0.20"; the actual +0.322 is
outside that, so candidate 1 (decoy locality / decoy construction) is not merely reopened,
it is implicated.

### But the improvement is not coming from their sigma

| decomposition | rho |
|---|---|
| our index vs their index (full Z vs full Z) | +0.322 |
| our index vs their index **globalised** (their sigma removed) | +0.314 |
| our index **globalised** vs their index (our sigma removed) | +0.176 |
| our E0 vs their E0 (numerators only) | +0.222 |

Removing *their* per-contact sigma costs almost nothing (+0.322 -> +0.314). Removing
*ours* costs half of it. So mutational's advantage over configurational is in the
**numerator** -- their mutational E0 sums every (i,k) and (j,k) contact of both partners --
not in the fact that its sigma varies.

## E4: residue level against singleresidue mode

Prediction registered: rho >= +0.40. Confirmed.

FrustX `mean_frustration` per residue vs frustratometeR `singleresidue` `FrstIndex`:
**rho = +0.412, p = 0.0002, n = 76.** Note the singleresidue output has no `FrstState`
column, so classification cannot be compared -- only rank.

### Follow-up: `mean_frustration` was the wrong FrustX quantity

`mean_frustration` is the mean of a residue's *contact* indices (`frustx/output.py:67`)
— a summary of pair frustration, not a residue Z-score. Their `singleresidue` index is a
genuine per-residue Z-score. The comparison was therefore not like-for-like.

With the full decoy tensor available (`results/validation/decoy_samples/1ubq_min200.npz`)
the matching quantity can be built directly:

```
R_i = sum_j E_ij                                  residue i's total contact energy
Z_i = ( <R_i>_decoys - R_i_native ) / sd(R_i)     same sign convention as the contact index
```

This needs the retained tensor, not the production running sums: contact energies within a
residue are correlated, so `sd(R_i)` cannot be recovered by adding per-contact variances.

| FrustX quantity | vs their `singleresidue` | Pearson |
|---|---|---|
| A. `mean_frustration` (mean of contact indices) | ρ = **+0.412**, p = 2.2e-4 | +0.376 |
| B. true residue Z-score (from the tensor) | ρ = **+0.478**, p = 1.3e-5 | +0.494 |

A vs B correlate at ρ = +0.862 — related but not interchangeable, which is why the choice
mattered. **ρ = +0.478 is the best agreement with frustratometeR we have obtained**, at any
level, in any mode.

`scripts/compare_singleresidue.py` produces both, plus
`results/validation/plots_mutational/singleresidue_scatter.png`.

Caveat that does not go away: our decoys shuffle the *whole* sequence, while their
singleresidue decoys mutate only residue i and leave every other position native. Ours
perturbs residue i's environment at the same time as its identity, so it is the noisier
estimator of the two.

### Why the residue level is the one comparison that is not confounded

The additive-R² result below shows frustratometeR's *contact* indices are largely a sum of
two residue terms, which is what makes contact-level agreement uninterpretable — we cannot
tell shared pair physics from shared burial. **At residue level that objection disappears**,
because both sides are explicitly one-body quantities. There is no pair-specific component
being masked. ρ = +0.478 between an all-atom and a coarse-grained model, on the same
structure, is a real and interpretable agreement.

It is also, for the same reason, not evidence that FrustX's *per-contact* resolution works.
That still needs Chen et al.

## The result that reframes everything: how one-body is each index?

Fitting each quantity to `X ~ c + a_i + a_j`, the exact functional form of a sum of two
residue-level terms. High R^2 means the "contact" index is really a residue property.

| quantity | additive R² |
|---|---|
| **FrustX e_ij (numerator, w=0)** | **0.133** |
| **FrustX index (w=0)** | **0.234** |
| frustratometeR configurational E0 | 0.560 |
| frustratometeR configurational index | 0.561 |
| frustratometeR mutational E0 | 0.994 |
| frustratometeR mutational index | 0.954 |

**FrustX at w = 0 is by a wide margin the most pair-specific of the three indices.**
frustratometeR's mutational index is 95% reducible to a residue-additive function -- it is
a residue-level measure reported per contact. Its numerator is essentially `R_i + R_j`,
which is the `w = 1` form this project rejected precisely because it made the FrustX index
degenerate.

So the higher correlation with mutational is largely agreement on **residue-level burial
structure**, not on contact physics. That is consistent with E4: the residue-level
comparison (+0.412) is stronger than any per-contact comparison.

### What this means for validation strategy

Neither frustratometeR mode supplies a strongly pair-specific target on 1UBQ:
configurational is 56% one-body, mutational 95%. Per-contact rank agreement against either
is therefore bounded by how much contact-specific information the reference contains, which
is not much. **frustratometeR cannot validate the one property FrustX was built to
provide.** This is a limit of the comparison, not a defect in either tool, and it means:

- Residue-level agreement (rho +0.41 vs singleresidue) is the defensible cross-check, and
  should be reported as the headline instead of any per-contact number.
- Per-contact validation requires a target with genuine pair specificity -- i.e.
  reproducing a specific figure from Chen et al. (2020). That is now the only remaining
  test, and it is no longer optional.
- The earlier `w` sweep should be read in this light. It rejected `w = 1` for producing a
  degenerate, residue-reducible index. frustratometeR's mutational mode *is* that
  degenerate index, and it is the published reference. That is worth understanding before
  concluding the sweep settled the question.


---

# Rechecking the w sweep against a working target

The original `w` sweep (above) concluded that the correlation with frustratometeR is
"insensitive to the whole Eq. 2 question" -- rho stayed at 0.13-0.18 from w = 0 to w = 1.
That conclusion was drawn against `configurational` mode, whose index is affine in its own
native energy. It had to be rechecked against `mutational`, which has a genuine
per-contact sigma.

## Method: one ensemble, every w, in closed form

`scripts/sweep_background.py`. The contact energy is linear in w:

    E_ij(w) = e_ij + w * B_ij,     B_ij = 1/2 (R_i + R_j)

so over a decoy ensemble the first two moments follow analytically:

    mean_E(w) = <e> + w <B>
    var_E(w)  = var(e) + 2 w cov(e, B) + w^2 var(B)

Accumulating the five running sums `Se, See, Sb, Sbb, Seb` gives the index at any w from a
**single** decoy pass. This replaces one 500-decoy job per w with one job total, and -- more
importantly than the compute saving -- every w shares the *identical* decoys, so
differences between w values are not confounded by decoy sampling noise the way
independent runs are.

Validated before use: a 20-decoy probe reproduced the committed 500-decoy `w = 0` index at
rho = +0.893, consistent with sampling noise at that ensemble size.

Run: 500 decoys, `protocol="min"`, `min_seq_sep=2`, on the frustratometeR-prepared
`1ubq_A.pdb`. Output `results/validation/sweep_w/sweep_min500.csv`.

## Result: the original conclusion was an artifact, but w = 1 is still not vindicated

| w | rho vs configurational | rho vs mutational | additive R² | contacts < −1 |
|---|---|---|---|---|
| 0.0 | +0.136 | +0.321 | 0.193 | 10 |
| 0.05 | +0.177 | +0.458 | 0.408 | 1 |
| 0.1 | +0.176 | +0.501 | 0.526 | 0 |
| 0.25 | +0.163 | +0.526 | 0.672 | 0 |
| 0.5 | +0.156 | +0.536 | 0.734 | 0 |
| 1.0 | +0.149 | **+0.537** | 0.754 | 0 |

Against configurational, rho is flat -- reproducing the original sweep and explaining why
it concluded w does not matter. Against mutational, rho rises monotonically and by 67%,
from +0.321 to +0.537 (p 5e-27). **So w does matter, and the earlier "insensitive"
conclusion was an artifact of scoring against a constant-sigma target.**

But `additive R²` rises in lockstep, 0.193 -> 0.754. Raising w makes the FrustX index more
residue-additive, and frustratometeR's mutational index is 95% residue-additive. The
suspicion is therefore that the agreement is bought purely by becoming equally degenerate.

## Removing the one-body component settles it

Residualising `X ~ c + a_i + a_j` out of **both** sides before correlating:

| w | raw rho | one-body removed |
|---|---|---|
| 0.0 | +0.321 | −0.116 (p 0.03) |
| 0.05 | +0.458 | −0.070 (p 0.19) |
| 0.1 | +0.501 | −0.056 (p 0.30) |
| 0.25 | +0.526 | −0.015 (p 0.79) |
| 0.5 | +0.536 | +0.026 (p 0.64) |
| 1.0 | +0.537 | +0.045 (p 0.41) |

**Zero contact-specific agreement at every w.** The entire w-dependence -- all 0.22 of the
rise -- is the one-body component. The only nominally significant entry is w = 0, and it is
*negative*.

## What this settles

- The earlier sweep's conclusion is withdrawn: rho is *not* insensitive to w. It looked
  that way only because the target had no per-contact information.
- The corrected sweep nonetheless does **not** argue for w = 1. Higher w buys agreement
  with frustratometeR only by making the index a residue-level quantity, which is the
  property it was built not to be, and at w >= 0.1 it reports **zero** frustrated contacts
  in ubiquitin.
- `DEFAULT_BACKGROUND_WEIGHT = 0.0` therefore stands, but the justification has changed.
  It is not "w does not affect agreement" (false). It is: **frustratometeR cannot
  adjudicate w at all**, because agreement with it at any w is entirely one-body, so w must
  be chosen on internal grounds -- pair specificity (0.193 vs 0.754 additive R²) and the
  ability to report frustrated contacts at all (10 vs 0).
- This is the third distinct conclusion in this document that was produced by comparing
  against `configurational` mode and did not survive contact with a working target. Any
  future claim resting on that comparison should be treated as unsupported until rechecked.

# Same equations, different force field: separating force field from protocol

Every FrustX-vs-frustratometeR comparison so far changed two things at once — the energy
function (REF2015 all-atom vs AWSEM coarse-grained) and the protocol (contact definition,
decoy construction, what enters the numerator). Reimplementing AWSEM's energy and running
*it* through FrustX's own machinery fills the missing cell of a 2x2 and separates them:

| | FrustX protocol | frustratometeR protocol |
|---|---|---|
| **REF2015** | `frustx_min500` | infeasible |
| **AWSEM** | `frustx_awsem` (new) | `frustra_mutational` |

`frustx/awsem.py` transcribes `compute_water_energy` (`fix_backbone.cpp:5444`),
`compute_burial_energy` (`:5478`), the gamma table layout (`:624-700`) and the CB-based
distance (`:5558`). `scripts/awsem_frustration.py` runs it through FrustX's contact set,
FrustX's whole-sequence shuffle decoys (literally `frustx.decoys.shuffle_sequence`, same
seeds) and Eq. 1 with a per-contact σ. The pair energy used is the AWSEM *water* term
alone, which is the right parallel to REF2015's pair energy under `w = 0`.

## The energy function is verified exact; the density function is not

Their `configurational` native energy is exactly `water(i,j) + burial_i + burial_j`
(`fix_backbone.cpp:5210`), so our transcription can be checked contact by contact against
their own output. On 1UBQ's 389 contacts, using their reported densities and our distances:

**max |difference| = 1.1e-3, mean 3.0e-4, r = 0.99999977** — i.e. agreement to their
output's three-decimal rounding. The energy function, the gamma tables, the residue-type
ordering and the CB distances are all confirmed correct.

**Our transcription of the density (`compute_ro`) does not reproduce theirs** and is not
used. Their reported ρ runs 0.000–5.225 (mean 1.106); ours is systematically higher on
helical residues, and no combination of well radii or sequence-separation exclusion we
swept reproduces their values. Since ρ depends only on geometry, and geometry is fixed
across the native and every decoy, `scripts/awsem_frustration.py` reads their reported
densities instead. That is both safe and strictly more faithful than using a version we
cannot verify — but it means the module cannot yet be run on a structure frustratometeR
has not already processed. **Open item.**

## Result: the force field is the larger difference, and the only one carrying signal

344 contacts shared by all three, 500 decoys:

| comparison | what differs | ρ |
|---|---|---|
| FrustX-REF2015 vs FrustX-AWSEM | force field only | **+0.175** (p = 0.001) |
| FrustX-AWSEM vs frustratometeR | protocol only | **+0.270** (p = 4e-7) |
| FrustX-REF2015 vs frustratometeR | both | +0.322 (p = 1e-9) |

Holding the protocol *completely* fixed — same contacts, same decoys, same equation — and
swapping only the energy function gives ρ = +0.175. The two force fields substantially
disagree about which contacts are frustrated. Note also that neither single difference
reaches the +0.322 obtained when both differ, so these do not compose additively and the
+0.322 should not be read as "two small problems stacking up".

## The 95% one-body result is a protocol artifact, not an AWSEM property

| index | additive R² |
|---|---|
| FrustX REF2015 | 0.234 |
| **FrustX AWSEM** | **0.385** |
| frustratometeR mutational | 0.954 |

This is the most useful thing the experiment produced. AWSEM's energy run through FrustX's
protocol is only 38.5% residue-additive — far from the 95.4% of frustratometeR's mutational
index. **The near-total one-body character of their index is caused by their numerator
summing over every contact of i and every contact of j (`fix_backbone.cpp:5214-5240`), not
by anything in the AWSEM energy function.** AWSEM is a pair-specific energy; their
mutational protocol is what discards the pair specificity.

## With the one-body component removed

Stripping the additive fit from all three indices and re-correlating the residuals — the
control that killed the earlier water-mediated conclusion:

| welltype | n | force field | protocol | both |
|---|---|---|---|---|
| ALL | 344 | **+0.122** | −0.575 | −0.138 |
| short (direct, r < 6.5 Å) | 115 | +0.188 | −0.410 | −0.221 |
| long (protein-mediated) | 71 | +0.190 | −0.802 | −0.035 |
| water-mediated | 158 | +0.039 | −0.522 | −0.104 |

Two things follow.

**The force-field comparison is the only one that survives the control.** REF2015 and AWSEM
retain a small positive pair-level agreement (+0.122 overall, ~+0.19 on direct and
protein-mediated contacts) after all one-body content is removed. It is weak, but it is
real and it is positive.

**The protocol comparison inverts.** Same energy function, different protocol, and the
residuals anti-correlate at −0.575. Their mutational index has only 4.6% non-additive
variance to begin with, and what remains does not merely fail to track genuine pair
frustration — it runs against it. Caveat: this is a correlation computed on the small
residual of a mostly-additive quantity, so its magnitude should not be over-read; the sign
and its consistency across all three well types are the durable part.

**Water-mediated contacts, properly controlled.** With protocol held fixed and one-body
content removed, the two force fields agree at +0.039 on water-mediated contacts against
~+0.19 on direct and protein-mediated ones. The original intuition — that AWSEM's explicit
desolvation well and REF2015's implicit `fa_sol` are modelling different physics — is
supported here, in the controlled form the earlier withdrawn version lacked.

## What this means for FrustX

The disagreement with frustratometeR is now decomposed and largely explained. FrustX's index
is the least one-body of the three (R² 0.234), the force-field difference is real and
measured, and the residual disagreement with frustratometeR is traceable to a protocol that
discards pair specificity. **None of this validates FrustX's per-contact resolution** — a
weak positive agreement with a coarse-grained model is consistent with FrustX being right,
being wrong, or both being wrong. That still needs Chen et al.

# Chen et al. publishes no per-contact values: that validation route is closed

Every previous section ends with "this still needs Chen et al." Having now read the full
text (Europe PMC `PMC7683549`), that route does not exist.

## What the paper actually publishes

Figure captions, read directly:

| Figure | Content | Quantitative? |
|---|---|---|
| 1 | Funnel schematic | no |
| 2 | 5 allosteric conformer pairs: structures **+ a per-residue profile at right** | **partly** |
| 3 | 12 enzymes, cartoons with green/red contact sticks | no |
| 4 | 7 protein complexes, all-atom vs coarse-grained renderings | no |
| 5 | EGFR–inhibitor renderings; **5e** affinity correlation | 5e only — **drug** |
| 6 | COX–inhibitor renderings; **6d** box plot | 6d only — **drug** |

**No figure, drug or non-drug, reports a frustration value for an identified contact.**
Figs. 3 and 4 are galleries: green and red lines on a cartoon, no axis, no scale, no
statistic. The only quantitative panels in the paper are 5e and 6d, and both are ligand
work, which is out of scope here.

The main-text scalars in the non-drug half are 14.2% (atomistic) vs 26.0% (AWSEM)
minimally frustrated interface contacts, pooled over Fig. 4's seven complexes. These are
real quantitative non-drug numbers and the section below works through what can be done
with them; do not read the per-contact conclusion here as covering them.

**Consequence: "validate FrustX per-contact against Chen et al." is not achievable.** This
is a closed question, not an outstanding task. Nothing downstream should be written as if
the check is merely pending.

## Category ratios: what the paper reports, and what of it we can test

Aggregate class fractions are a different question from per-contact values, and here the
paper is not silent. It publishes exactly three numbers, all in the Fig. 4 / Supplementary
Fig. 8 interface analysis:

| quantity | value |
|---|---|
| interface contacts minimally frustrated, **atomistic** | **14.2%** |
| interface contacts minimally frustrated, **AWSEM** | **26.0%** |
| water-mediated interactions minimally frustrated, AWSEM | ~24% |

plus two comparative claims carrying no number:

> "short-range and long-range contacts that are not water-mediated do display **similar
> fractions** of minimally frustrating interactions in both frustratometers"

> "the atomistic frustratometer shows only a **much lower fraction** of minimally frustrated
> water-mediated interactions and a **higher level of high frustration** in the
> water-mediated contacts than does the AWSEM"

### 14.2% / 26.0% cannot be reproduced — the systems are unnamed

These are pooled over Fig. 4's seven complexes. The Fig. 4 caption gives **no PDB IDs**, and
the only complex named anywhere in the main text is **2PCC** (a "wet" interface, panel f or
g). The remaining six are identified only inside the figure image or Supplementary Fig. 8.
Without them there is no denominator to match, and a single complex cannot stand in for a
pooled fraction over seven.

### The two unnumbered claims are testable, and we hold the data

`scripts/category_ratios.py`, on 1UBQ, over the **344 contacts all three calculations share**
(joined on contact identity so the welltype label and the denominator are identical
throughout). Welltype comes from frustratometeR, the only source that assigns it.

| welltype | n | calculation | minimally % | neutral % | highly % |
|---|---|---|---|---|---|
| short | 115 | frustratometeR / AWSEM | 34.8 | 56.5 | 8.7 |
| | | FrustX / REF2015 | 29.6 | 67.0 | 3.5 |
| | | FrustX / AWSEM | 24.3 | 63.5 | 12.2 |
| long | 71 | frustratometeR / AWSEM | **67.6** | 31.0 | 1.4 |
| | | FrustX / REF2015 | **23.9** | 73.2 | 2.8 |
| | | FrustX / AWSEM | 25.4 | 73.2 | 1.4 |
| water-mediated | 158 | frustratometeR / AWSEM | 33.5 | 55.1 | 11.4 |
| | | FrustX / REF2015 | 17.7 | 79.7 | 2.5 |
| | | FrustX / AWSEM | 18.4 | 71.5 | 10.1 |

**Claim 1 — "similar fractions for non-water-mediated" — half holds.** Short-range: 34.8%
vs 29.6%, similar. Long-range: **67.6% vs 23.9%**, a 44-point gap. On this monomer the claim
fails badly for exactly one of the two welltypes it asserts.

**Claim 2 — first half reproduces, second half is contradicted.** Atomistic water-mediated
minimal frustration *is* much lower (17.7% vs 33.5%) — the paper's central observation about
Rosetta lacking explicit waters comes out of an independent implementation. But the
"higher level of high frustration" does not: we get **2.5% against AWSEM's 11.4%**, the
opposite direction.

**That contradiction is confounded and should not be reported as a refutation.** The `−1.0`
cut point is AWSEM-calibrated and is already on record here as unsupported for REF2015 (only
10 contacts in 1UBQ reach it, and the empirical tail is non-monotonic). Every "highly %"
column is therefore dominated by a threshold we know does not transfer, whereas `+0.78` is
supported and the "minimally %" columns are informative. Note FrustX/AWSEM recovers 10.1%
against frustratometeR's 11.4% on the same contacts, which places the discrepancy in the
force field, not in our protocol or arithmetic.

### The paper inherits the same thresholds we did

Searched directly: `0.78`, "threshold", and every phrasing of a cut point appear **nowhere**
in the paper. The classification is introduced only as "as discussed in Ferreiro et al. 4,
the individual contacts can be roughly classified as being either minimally frustrated,
highly frustrated, or neutrally frustrated" — i.e. the AWSEM-calibrated values, adopted for
an atomistic index without recalibration, exactly as FrustX does. This is inference from a
citation rather than a stated method, but it means 14.2% was very likely computed with the
same `+0.78` we use, so the number is comparable *in construction* even though it is not
reproducible for want of the systems.

### Side finding: the paper does work at 5 Å from the Cα

Recorded because it bears on the vicinity choice above. The Fig. 2 caption never defines
"vicinity", but the Results say of a different analysis: *"The regions with mobile residues
are enriched in highly frustrated interactions **up to 5 Å from the Cα** of the mobile
residues"* (Supplementary Fig. 3). That is not the Fig. 2 quantity and does not define it,
but it is direct evidence that the paper's own spatial analyses use a 5 Å radius measured
from Cα — which is what `XAdens` does. It strengthens, without settling, the choice made in
`scripts/vicinity_profile.py`.

## The deeper reason it was never coherent

Eq. 2 collapses to `E_ij = ½(R_i + R_j)` — the direct term cancels exactly (verified to
7.1e-15 above). That is *precisely* the additive `a_i + a_j` form, the same construction as
frustratometeR's mutational numerator (`fix_backbone.cpp:5214-5240`). The paper's own
contact index is therefore built to be a residue property.

Measured on 1UBQ from the retained decoy tensor, varying only `w`:

| w | additive R² | what it is |
|---|---|---|
| 0.0 | **0.204** | FrustX default, `e_ij` only |
| 0.5 | 0.741 | intermediate |
| 1.0 | **0.757** | **Chen et al. Eq. 2 as written** |

The per-contact σ puts back a little pair specificity, but the paper's published index is
~76% one-body. **FrustX's per-contact specificity is a deliberate departure from the paper,
not a reproduction of it.** A paper whose own index is three-quarters residue-additive
cannot validate a property it does not have. The `w = 0` choice must stand or fall on
internal grounds, as already recorded — this closes the door on external adjudication of it
from the paper as well as from frustratometeR.

## The literal Eq. 2 experiment: it is the paper's Methods, not our code

The open question was whether FrustX reporting **zero** highly frustrated contacts at
`w = 1` — while the paper's entire Results rest on highly frustrated clusters existing —
came from our arithmetic, our `e_ij` extraction, or the paper's Methods. `scripts/eq2_literal.py`
answers it from the retained decoy tensor (1UBQ, 200 decoys, `min` protocol, 475 contacts).
No new Rosetta run: every `w` shares identical decoys, so no difference below is decoy noise.

Note the parameterisation. `frustration.contact_energy_matrix` computes `e_ij + w·½(R_i+R_j)`,
which keeps `e_ij` on top and so overshoots the literal Eq. 2 by exactly `e_ij` at `w = 1`.
The script uses `E_ij(w) = (1-w)·e_ij + w·½(R_i+R_j)` instead, so that `w = 1` **is** Eq. 2
verbatim rather than approximately. This changes no conclusion (the two differ by ~0.7 REU
against a ~10 REU background) but it means the endpoint is the thing it claims to be.

**Not our arithmetic.** Eq. 2 written out term by term with the `k≠j` / `l≠i` exclusions
spelled out, versus `½(R_i+R_j)`: max abs diff **7.1e-15**. The collapse is exact.

**Not our `e_ij` extraction.** On contacts, mean `|e_ij|` = **0.663 REU** against mean
`|½(R_i+R_j)|` = **10.03 REU** — the background is **15.1×** the direct term, reproducing the
~14× recorded in `config.py`. (Use the *mean*: 42% of contacts sit near the 10 Å cutoff with
`|e_ij| < 0.05 REU`, so the median describes the cutoff edge and inflates the ratio to 68×.)

**It is the Methods.** The sweep:

| w | min F | median F | max F | med numerator | med σ | #high | #min | numerator > 0 |
|---|---|---|---|---|---|---|---|---|
| 0.00 | −2.97 | −0.03 | 6.54 | −0.00 | 0.21 | **10** | 78 | 45.5% |
| 0.10 | −0.73 | 0.56 | 4.66 | 0.75 | 1.55 | 0 | 132 | 96.0% |
| 0.25 | −0.36 | 0.57 | 3.40 | 1.87 | 3.67 | 0 | 112 | 98.1% |
| 0.50 | −0.18 | 0.55 | 2.94 | 3.62 | 7.33 | 0 | 104 | 99.2% |
| 1.00 | −0.18 | 0.54 | 2.94 | 7.17 | 14.54 | 0 | 94 | **99.6%** |

The frustrated tail is not compressed away by an inflated σ — numerator and σ grow together,
leaving the median index almost constant (0.56 → 0.54). It dies because **the numerator turns
uniformly positive**: 45.5% of contacts at `w = 0`, **99.6%** at `w = 1`.

The mechanism is a one-body tautology. At `w = 1` the contact energy depends on `i` and `j`
only through their own totals `R`. Shuffling the sequence of a real protein degrades it
globally — native `Σ R_i` = **−661 REU** against **−106 REU** for the decoy mean, and **72 of
76** residues have `R_i` better than the decoy mean. Every contact inherits that 555 REU gap
through `½(R_i+R_j)`, whether or not it is locally frustrated. So Eq. 2 as written measures
*"is this a well-optimised sequence"*, which is true almost everywhere by construction, and
it cannot report a frustrated contact for reasons that have nothing to do with the structure.

The collapse is immediate, not gradual: **`w = 0.1` already gives zero frustrated contacts**,
because the background is 15× the direct term the moment it is admitted at all.

**Conclusion.** Eq. 2 as literally published cannot reproduce the paper's own qualitative
result. This is not a defect we introduced, and it is not fixable by tuning `w` — any `w > 0`
destroys the frustrated tail. It independently confirms `DEFAULT_BACKGROUND_WEIGHT = 0.0` on
grounds stronger than the earlier additive-R² argument: at `w = 1` the index is not merely
75% one-body, it is *incapable of the paper's central claim*. Either the paper's implementation
differs from its Eq. 2, or its `e_ij` is not a pairwise decomposition of the kind REF2015 gives.
We cannot distinguish these — no code accompanies the paper.

## The quantitative non-drug target that is actually runnable

Fig. 2's right column: "a quantification of the minimally frustrated interactions (green)
or highly frustrated interactions (red) **in the vicinity of each residue**" for the pairs
1XTQ/1XTS, 1OIV/1OIW, 1KAO/2RAP, 1HH4/1MH1, 1H4X/1H4Y, with local Qi in black.

This is a per-*residue* spatial profile, one level coarser than per-contact. It can test
whether FrustX puts frustration in the right places; it cannot test any individual
contact's Z-score.

Two limits, both from the text itself:

- **"Vicinity" is never defined *by the paper*.** The word appears exactly once, in this
  caption. "Local Qi" likewise. There is no radius, in the caption, the Results, or the
  Methods. **This limit is now largely lifted** — see the next section: frustratometeR's
  `XAdens` computes exactly the quantity the caption describes, and we reproduce its
  output bit-for-bit, so we can adopt a *verified* reference definition instead of
  inventing one. What remains unproven is that Chen et al. used that same definition.
- **The counts depend on the classification thresholds**, which the paper does not state
  either, and which we have already shown are only half-supported for REF2015 (`+0.78`
  supported, `−1.0` not).

Smallest candidate: **1XTQ / 1XTS** (human Rheb·GDP / Rheb·GTP), 169 and 171 residues,
single chain, no numbering breaks.

## "Vicinity" recovered: XAdens, verified bit-for-bit

`frustratometeR:::XAdens` (dumped from the installed package; not exported, and not
documented in the paper either) generates the `*_5adens` files, and computes precisely the
Fig. 2 caption's quantity. The rule:

- each **contact** is given a position — the **midpoint of its two interacting atoms**
- for each residue `i`, count contacts whose midpoint lies within `radius` of `i`'s **CA**
- strictly `< radius`, default **5 Å** (the R code uses `<`, not `<=`)
- split at the usual cut points: highly `≤ −1`, minimally `≥ 0.78`

**This is not "the contacts involving residue `i`".** It is a spatial density of contact
midpoints near `i`, so a residue accumulates contacts it takes no part in. That is exactly
what "in the vicinity of" buys over "of", and reading it the other way would change the
profile shape — the only thing Fig. 2 lets us compare.

Verified, not assumed. `scripts/vicinity_profile.py` reproduces frustratometeR's own 1UBQ
`_mutational_5adens` file **exactly**: 76/76 residues matching on `Total`, `nHighlyFrst`,
`nNeutrallyFrst` and `nMinimallyFrst`, max abs diff **0**.

The coordinate convention was settled by the same test rather than by preference:

| midpoint atoms | residues matching on all four counts | max abs diff in `Total` |
|---|---|---|
| **CB** (CA for Gly) | **76 / 76** | **0** |
| CA | 10 / 76 | 9 |

CB wins decisively, which is what AWSEM's contact definition implies. It stays a parameter
(`atom=`) all the same: this pins down *frustratometeR's* definition, not Chen et al.'s,
which remains unstated. FrustX defines contacts by CA–CA distance, so the two conventions
are genuinely different choices and the difference is not negligible.

## Blocker: heteroatoms crash the contact map

Verified live on 1XTQ. Rosetta loads 171 residues — 169 protein plus `MG` and `pdb_GDP`,
both *recognised* rather than dropped by `-ignore_unrecognized_res` — and
`ca_coords_from_pose` raises `RuntimeError: ResidueType MG does not have an atom CA`.

Every Fig. 2 candidate is a nucleotide-binding protein and carries the same problem. A
protein-only, single-chain preparation step is a prerequisite for any of this work, and
`run.json`'s `n_residues` should be asserted against the expected count so silent drops
cannot pass unnoticed.

# Superseded: relaxation protocol and the well-type split

Retained because the measurements are sound and the `min`/`relax` comparison is still
load-bearing. The *interpretation* offered here is superseded by the section above.

## Result: Rheb GDP vs GTP, the first reproduction attempt at Fig. 2's level

> **SUPERSEDED — see "The Rheb result does not replicate" below.** The biological
> claim in this section did not survive replication across two further GTPase conformer
> pairs. It is kept verbatim rather than edited, because the way it failed is the point.

1XTQ (Rheb·GDP, inactive) and 1XTS (Rheb·GTP, active), protein-only, REF2015, `w = 0`,
500 `min` decoys each, XAdens 5 Å vicinity on CB midpoints. Figure:
`results/fig2/rheb_profile.png`; per-residue table `results/fig2/rheb_profile_comparison.csv`.

| | contacts | σ=0 dropped | highly | minimally | median F |
|---|---|---|---|---|---|
| 1XTQ (GDP) | 1306 | 42 | 31 | 181 | −0.06 |
| 1XTS (GTP) | 1305 | 32 | 34 | 185 | −0.06 |

Global profiles are highly similar (`Total` r = 0.964, `nMinimallyFrst` r = 0.809), as they
must be for two conformers of one protein — so the question is entirely about *where* they
differ.

### The change concentrates on the functional elements

Taking the P-loop (12–20), switch I (32–41) and switch II (63–79) from small-GTPase biology,
**a priori**, not chosen after looking:

| region | n | mean Δ highly | mean Δ minimally |
|---|---|---|---|
| P-loop | 9 | +0.56 | −0.22 |
| switch I | 10 | +0.30 | −1.20 |
| switch II | 17 | +0.65 | −1.29 |
| **elsewhere** | 133 | **−0.11** | **+0.30** |

All three functional elements move the same way on activation — *more* highly frustrated,
*less* minimally frustrated — while the rest of the protein moves the opposite way.

### Controls

**Decoy noise floor, measured not assumed.** Splitting each structure's own 500 decoys into
halves and profiling each gives a within-structure difference of pure sampling origin. The
measured floor compares two 250-decoy estimates while the real comparison is two 500-decoy
estimates, so it overstates the relevant noise by √2 and is corrected:

| | between-structure signal | corrected noise floor | ratio |
|---|---|---|---|
| Δ highly | 0.550 | 0.152 | **3.6×** |
| Δ minimally | 1.408 | 0.341 | **4.1×** |

> This control is sound but measures the **wrong noise scale** for a between-structure claim
> — see "The methodological lesson" below. It bounds decoy sampling noise, not the variation
> between two crystals of one protein.

**Spatial null.** Region means were tested against randomly placed *contiguous blocks* of the
same lengths, not permuted residue labels — the profile is spatially autocorrelated, and
label permutation would destroy that and understate the null. Individually only switch II
reaches nominal significance (Δ minimally, p = 0.033), which does not survive correction for
6 tests. The **joint** a-priori contrast (all 36 functional residues vs the other 133,
three random blocks of matched size, 20 000 draws) does:

| | contrast | null sd | p | Bonferroni (2 metrics) |
|---|---|---|---|---|
| Δ highly | +0.63 | 0.31 | 0.044 | 0.088 |
| Δ minimally | **−1.30** | 0.52 | **0.010** | **0.020** |

### It is not a restatement of geometry, but it cannot be separated from it either

CA RMSD after superposition is 0.86 Å overall, and localises exactly where expected:
switch I **2.18 Å** (max 4.51), switch II 1.35 Å, elsewhere 0.59 Å, P-loop only 0.33 Å.

Per-residue structural deviation explains almost none of the frustration change globally —
Spearman ρ = +0.07 (Δ highly) and +0.02 (Δ minimally), regression R² = 0.036 and 0.059. So
the profile is **not** re-reporting RMSD in different units.

But adjusting for deviation drops the joint contrast to p = 0.149 / 0.123. Read this
carefully: deviation is collinear with region membership (both concentrate on the switches),
so the adjustment partly removes the thing being tested — it over-controls rather than
refutes. The defensible statement is the weaker pair: **the functional elements do show a
frustration shift on activation (p = 0.010), and that shift is not explained by local
structural change globally — but "frustration changes *beyond* the conformational change" is
not established by this data.**

### Limits that remain

- **Absolute counts are not comparable to the published figure.** Chen et al. state neither
  their thresholds nor their radius; ours are frustratometeR's, and `−1.0` is on record here
  as unsupported for REF2015. Only shape is comparable.
- **The two crystal structures differ in quality**: 1XTQ at 2.00 Å (R 0.219), 1XTS at 2.80 Å
  (R 0.231). Some of the difference may be crystallographic rather than conformational. A
  same-structure control cannot rule this out; only a second conformer pair can.
- **"Local Qi"**, the black trace in the published figure, is not reproduced — the paper
  never defines it.
- This validates *spatial localisation at residue resolution*. It says nothing about any
  individual contact's Z-score, which remains unvalidatable from this paper.

## The Rheb result does not replicate

The section above stands as written — it is what one conformer pair showed, and I am not
editing it away. But it named its own remedy ("only a second conformer pair can" separate
crystallographic quality from conformation), and that remedy has now been run. It comes back
negative.

`scripts/gtpase_replication.py`. Three GDP/GTP pairs from the paper's own Fig. 2 set, chosen
because their weaknesses are **complementary** — no single confounder is shared by all three,
so an effect surviving all three would not be explained by any one of them:

| pair | protein | strength | weakness |
|---|---|---|---|
| 1XTQ / 1XTS | Rheb | — | 2.00 vs 2.80 Å |
| 1KAO / 2RAP | Rap2A | sequences identical, 0 mismatches | 1.70 vs 2.60 Å |
| 1OIV / 1OIW | Rab11A | 1.98 vs 2.05 Å, resolution-matched | 1OIW carries Q70L, *inside* switch II |

Residue 70 is additionally excluded from the Rab11A test: a mutated residue has different
frustration by construction, so leaving it in would manufacture a signal in switch II.

**Regions are derived, not eyeballed** (`scripts/gtpase_replication.py:61`). The three
proteins are numbered differently, and hardcoding a range per structure would let me fit the
boundaries to the answer. Instead the two sequence motifs are located and the canonical Ras
offsets applied uniformly to all three:

```python
p = re.search(r"G.{4}GK[ST]", txt)                                    # Walker A / G1
d = next(m.start() for m in re.finditer(r"D..G", txt) if m.start() > p.end())   # G3
return {"P-loop":   (nums[p.start()],    nums[p.end() - 1]),          # Ras 10-17
        "switch I": (nums[p.start()+20], nums[p.start()+28]),         # Ras 30-38
        "switch II":(nums[d+3],          nums[d+19])}                 # Ras 60-76
```

Sanity check on the rule rather than on the answer: applied to Rheb it returns switch I 33–41
and switch II 63–79, against the 32–41 / 63–79 used before this script existed. It reproduces
the earlier hand choice instead of redefining it.

### The numbers

Joint a-priori contrast, all functional residues vs the rest, three random contiguous blocks
of matched size, 20 000 draws — the same test as the Rheb section:

| pair | d highly | p | d minimally | p |
|---|---|---|---|---|
| Rheb | +0.66 | 0.038 | **−1.32** | **0.010** |
| Rap2A | **−0.56** | 0.038 | −0.07 | 0.919 |
| Rab11A (excl. 70) | +0.00 | 1.000 | +0.22 | 0.675 |
| **pooled** | **+0.04** | **0.858** | **−0.39** | **0.186** |

Signs are inconsistent on both metrics. Rap2A does not merely fail to replicate — it
**reverses** on `d highly`, and at nominally the same p as Rheb, which is the clearest
possible demonstration that p = 0.038 on one pair carries no weight here. Rab11A, the one
resolution-matched pair, is flat: +0.00 and +0.22.

### The negative is not a broken run

Checked before believing it. All six calculations are comparable in every global quantity:

| | contacts | σ=0 dropped | highly | minimally | median F | mean vicinity |
|---|---|---|---|---|---|---|
| 1XTQ | 1306 | 42 | 31 | 181 | −0.056 | 32.4 |
| 1XTS | 1305 | 32 | 34 | 185 | −0.061 | 32.1 |
| 1KAO | 1248 | 27 | 36 | 185 | −0.045 | 31.2 |
| 2RAP | 1281 | 28 | 26 | 187 | −0.048 | 32.2 |
| 1OIV | 1283 | 27 | 36 | 169 | −0.050 | 32.2 |
| 1OIW | 1264 | 19 | 35 | 172 | −0.045 | 32.2 |

No run is an outlier in contact count, dropped-σ fraction, category counts, median index or
vicinity density. The nulls are nulls, not failures.

And the resolution confounder does not tidy this up either: if resolution mismatch drove the
Rheb effect, effect size should track the gap. It does not. **Rap2A has the largest gap
(0.90 Å) and the smallest effect on `d minimally`**, while Rheb's 0.80 Å gap gives the
largest. No single confounder explains the pattern; the honest reading is that there is no
effect to explain.

### What is withdrawn, and what survives

**Withdrawn:** "the functional elements of small GTPases shift toward frustration on
activation." Three pairs, pooled p = 0.858 / 0.186, inconsistent signs. It is not supported.

**Survives:** the Rheb *observation* — in this one protein, on these two crystals, the
functional elements do move that way. What the replication removes is any licence to read
that as a property of GTPase activation rather than of one crystal pair.

Also unaffected: everything the Rheb section established about the *machinery* — that the
per-residue profile is reproducible, that the two conformers' global profiles agree (r =
0.964), that XAdens is the right vicinity rule. The pipeline works. It is the biological
claim that does not.

### The methodological lesson, which is the real result

The Rheb section reported the effect as "3.6–4.1× the measured noise floor," and that floor
was measured honestly — split each structure's own 500 decoys in half, profile each half,
√2-correct. But it measures **decoy sampling noise only**: how much the answer moves when you
redraw the decoys for a *fixed* structure.

That is the wrong noise scale for the claim being made. The claim compares two structures, so
the relevant variability is **between-structure** — everything that differs between two
crystals of one protein that is not the conformational change of interest: resolution,
refinement protocol, crystal contacts, occupancy, the modeller's choices. Decoy noise is a
strict subset of it, and evidently a small one.

The three pairs are the first actual estimate of that scale, and it is large enough to
swallow the Rheb effect whole. **Rule going forward: a split-half decoy control licenses no
statement about two different structures.** Any between-structure claim needs replication
across independent structure pairs, and one pair is not evidence regardless of its p-value.

## Experiment: relaxation protocol (candidate 2)

Two matched 500-decoy runs on the frustratometeR-prepared `1ubq_A.pdb`, `w = 0`,
`--min-seq-sep 2`, differing only in `--protocol`:

| | ρ vs frustratometeR | sd | median | GLY median |
|---|---|---|---|---|
| `min` | +0.141 | 0.94 | +0.04 | −0.31 |
| `relax` | +0.146 | 0.93 | +0.01 | −0.44 |

The two runs correlate with **each other** at ρ = +0.951 (n = 457 defined contacts).
Side-chain relaxation of the decoys does not materially change the index.

**Candidate 2 is ruled out**, and this conclusion stands: it is a FrustX-internal
comparison that does not depend on what frustratometeR computes. Side-chain relaxation of
the decoys does not materially change the index.

(The original text here also claimed decoy locality was "already eliminated" on the basis
of ρ +0.265 vs +0.273. That claim is withdrawn — see the candidate-list status above.)

## Where the disagreement actually lives

Splitting the shared contacts by AWSEM's own well type is decisive, and reproduces
independently in both runs:

| AWSEM well type | n | ρ (`min`) | ρ (`relax`) | p |
|---|---|---|---|---|
| direct — short + long | 186 | **+0.309** | **+0.312** | < 0.0001 |
| water-mediated | 158 | −0.030 | −0.014 | 0.71 / 0.86 |
| all | 344 | +0.141 | +0.146 | ≈ 0.008 |

The methods agree significantly on contacts where **both model direct atomic interaction**,
and not at all — statistically indistinguishable from zero — on contacts where AWSEM
invokes its water-mediated well. Water-mediated contacts are 46 % of the shared set, so
averaging them in is what drags the overall ρ from 0.31 down to 0.14.

Restricted to direct contacts, our index also separates *their* classes in the right
direction: their `highly` contacts sit at our +0.16, their `minimally` at our +0.86
(separation +0.70; +0.67 for `relax`).

### Interpretation withdrawn

Two subsections stood here: "Why this is the expected result, not a defect", which argued
that AWSEM's explicit water-mediated well and REF2015's implicit `fa_sol` are not
approximating the same quantity, and an "honest statement" that treated the direct-contact
ρ ≈ 0.31 as evidence the implementation is not fundamentally wrong.

Both are withdrawn. The one-body residualisation above shows the direct-contact agreement
was carried by a shared burial term rather than by contact physics, so ρ ≈ 0.31 was not
evidence of anything about the pair interaction. The solvation argument may still be true,
but nothing measured here tests it — the well-type split is fully explained by the
one-body confound, and a real solvation difference would be invisible underneath it.

The tables above remain accurate as measurements.
## Side finding: σ = 0 contacts at the cutoff edge

17 of 475 contacts have `decoy_std == 0` and `native_energy == 0` — no REF2015 term fires
for the pair in the native structure or in any of 500 decoys. All lie at Cα–Cα 8.5–9.9 Å
(defined pairs: median 7.5 Å), i.e. the outer edge of the 10 Å cutoff, where a Cα-based
contact criterion admits pairs with no atomistic interaction. Eq. 1 is 0/0 there; they are
emitted with `frustration_class = "undefined"` rather than a fabricated number.

None of the 17 appear in frustratometeR's contact set, so they never entered any
comparison above. Worth noting as an intrinsic mismatch between a Cα contact definition
and an all-atom energy — not a bug, but a reason the raw contact counts differ (475 vs 389).

## Readout scope: the largest identified driver of the frustratometeR disagreement

Every FrustX-vs-frustratometeR comparison in this document up to here compared our bare
`e_ij` against their neighbourhood sum. That was a **scope mismatch on top of a force-field
difference**, and nobody had isolated the two.

frustratometeR's mutational decoy energy is not a pair energy. Reading it out of the AWSEM
source that builds its shipped LAMMPS binary (`fix_backbone.cpp:5214-5243`, vendored at
`results/reference/awsem_source/`):

    E_fR(i,j) = water(i,j) + burial_i + burial_j + SUM_k water(i,k) + SUM_k water(j,k)

FrustX at `w = 0` measures `e_ij` alone. So the two indices were never scoped to the same
thing. `frustx/frustration.py:apply_readout` now makes that axis explicit and testable:

```python
def apply_readout(E, mask, readout=DEFAULT_READOUT):
    if readout == "pair":
        return E
    row = (E * mask).sum(axis=1)
    return row[:, None] + row[None, :] - (E * mask)
```

`E_ij` is subtracted once because it appears in *both* row sums. The transform must be
applied per-structure **before** Eq. 1, never after: it is linear in E, but Eq. 1 is not,
and σ of a sum is not the sum of σ. Unit tests on hand-checkable toy matrices are in
`tests/test_readout.py` (a lone contact's neighbourhood must equal its own pair energy:
counting `E_ij` twice gives 10, dropping it gives 0, only counting once gives 5).

### Result: seven structures, decoy ensemble and energy function held fixed

`scripts/readout_scope.py` -> `results/readout_scope.csv`. Nothing but the readout changes;
no new decoys, no PyRosetta, pure post-processing of tensors already on disk.

| scope | ρ vs fR | additive R² | **residualised ρ** | AUC |
|---|---|---|---|---|
| pair (current default) | +0.3008 | 0.2629 | **−0.0650** | 0.674 |
| neighbourhood (fR-matched) | +0.5210 | 0.9525 | **+0.2302** | 0.843 |

Means over 1UBQ + the six GTPases (n = 344–970 shared contacts each). The effect is not a
one-protein artifact: **ρ rises in all 7, and the residualised ρ flips sign in all 7**,
from a −0.011…−0.111 band to a +0.165…+0.299 band.

The residualised column is the one that matters. Regressing `c + a_i + a_j` out of both
sides removes everything a residue carries to all of its contacts — burial, exposure, local
packing — so what survives is contact-specific by construction. **This is the first
configuration in this document's record with positive contact-specific agreement with
frustratometeR.** It is the test that killed the `w` sweep (+0.32 → +0.54 raw, ~0 after
residualisation), killed the AWSEM-force-field comparison, and killed the pair-local decoy
claim. Scope is the first knob to survive it.

### What this does not license

**Do not make it the default, and do not read the magnitude as agreement.** Additive R²
goes 0.26 → 0.95: the neighbourhood readout is 95% reducible to a residue-additive
function, and frustratometeR's mutational index is itself ~95% additive. So most of the ρ
gain is the two indices becoming *jointly degenerate* — the same failure mode as `w = 1`,
reached by a different route. `+0.2302` is the residual of a 95%-additive quantity; the
**sign and its consistency across 7 structures are the durable findings, not the value**.

This cuts against the project's own purpose. FrustX exists to provide per-contact
specificity; a readout that is 95% one-body does not provide it, however well it correlates.
The honest statement is: *scope mismatch accounts for more of the FrustX–frustratometeR
disagreement than any force-field or protocol difference measured so far, and once scope is
matched a real contact-specific signal appears that was previously absent.* Whether that
signal is worth having at 95% additivity is a separate question this does not settle.

It also **retires the class-separation statistic** as evidence. Separation moves from
AUC 0.674 to 0.843 without touching a single decoy — so the "nearly doubled separation" that
was the sole support for the pair-local decoy scheme (`scripts/local_decoy.py`) cannot
support a conclusion about decoy locality. That claim needs re-running with rank-based
endpoints and a matched shell-restricted native reference, or dropping.

`DEFAULT_READOUT = "pair"` stays. The flag is `--readout {pair,neighbourhood}`, diagnostic.

## What the three modes actually randomise, read from the source

Recorded because this document twice described frustratometeR's null from its papers'
prose and got it wrong, and because the FrustX-vs-fR comparison cannot be interpreted
without it. Everything here is read from
`results/reference/awsem_source/fix_backbone.cpp` (the AWSEM source that builds the
LAMMPS binary frustratometeR ships), not from a paper.

| | `r_ij` | `rho_i`, `rho_j` | identities | rest of sequence | readout scope |
|---|---|---|---|---|---|
| **configurational** | **replaced** — drawn from a random *other* in-contact pair (`:5257-5266`) | **replaced** — from a *second, independent* random pair (`:5268-5271`) | random | native | pair only (`:5210-5212`) |
| **mutational** | native (`:5275`) | native (`:5276-5277`) | random | native | pair **+ all (i,k), (j,k)** (`:5215-5243`, `:5297-5310`) |
| **singleresidue** | n/a | native | random, residue i only | native | i's whole neighbourhood |

Two consequences that were not previously written down.

**Configurational does not hold geometry native.** It replaces `r_ij` wholesale with the
distance of some unrelated contacting pair, and then draws the two burial densities from a
*different* random pair again — so in a single configurational decoy the distance and the
two densities need not come from the same place in the protein, or from each other's. This
is a much more violent null than "perturb the pair", and it is why the configurational
index behaves as a rescaled energy rather than a per-contact Z-score (recorded earlier).

**Identities are drawn with replacement; FrustX permutes.** Both modes pick an identity as
`get_residue_type(get_random_residue_index())`, and

```c
int FixBackbone::get_random_residue_index()
{
  int index;
  index = rand() % n;      // fix_backbone.cpp:5550-5555
  return index;
}
```

Two independent draws. So frustratometeR samples i.i.d. from the native composition, and
the two positions of a contact are **independent**. FrustX shuffles
(`frustx/decoys.py:59-67`), which is sampling **without** replacement: the composition is
reproduced exactly rather than in expectation, and positions are weakly negatively
dependent (the indicator correlation is exactly −1/(L−1), a 0.6% effect at L ≈ 170).

The dependence is negligible and is **not** worth engineering around. What matters is that
the two tools' nulls agree only in their marginal, and the docstring at
`frustx/decoys.py:21-23` — which justifies shuffling by appeal to the paper's "native amino
acid frequency distribution" — should not be read as claiming the reference implementation
does the same thing. It does not.

## Contact definition: CB fixes the contact set, not the benchmark

A Cα–Cα cutoff is the paper's definition ("a cutoff of 10 Å is used"), but it admits pairs
whose **side chains point away from each other**, for which no REF2015 term fires at all.
Across the six GTPase runs, **1824 of 7687 contacts (23.7%) have decoy σ ≤ 0.02** — Eq. 1
is 0/0 or nearly so. These are not merely quiet contacts, they are less reliable: split-half
on 1XTQ gives **0.897 for them against 0.988 for the rest** (Spearman-Brown at N=500).

### Which criterion separates the dead pairs

Tested three, at **matched contact counts** so the comparison is not just "keep fewer":

| criterion | ~4000 kept | ~5100 kept |
|---|---|---|
| Cα–Cα | 3.1% dead | 9.8% dead |
| min heavy-atom distance | 1.8% dead | 8.2% dead |
| **Cβ–Cβ** (Gly → CA) | **0.4% dead** | **3.0% dead** |

Cβ wins, and beats minimum heavy-atom distance — which is initially counterintuitive, since
heavy-atom distance is the more directly physical quantity. The reason is that **Cβ encodes
side-chain direction**: two residues can have close backbones while their side chains point
apart, which minimum-atom distance sees as a contact and the energy function does not.

At the paper's cutoff scale: Cα ≤ 10.0 keeps 7687 (23.7% dead), Cβ ≤ 9.5 (frustratometeR's)
keeps 5713 (6.0%), Cβ ≤ 9.0 keeps 5103 (3.0%).

### It does not improve agreement with frustratometeR

`scripts/contact_definition.py` → `results/contact_definition.csv`, means over 7 structures:

| definition | contacts | dead | shared with fR | ρ | residualised ρ | AUC |
|---|---|---|---|---|---|---|
| Cα ≤ 10.0 | 1166 | 24.8% | 865 (74%) | +0.3008 | −0.0650 | 0.674 |
| Cβ ≤ 9.5 | 958 | 7.7% | 958 (**100%**) | +0.2870 | −0.0516 | 0.659 |
| Cβ ≤ 9.0 | 812 | 3.2% | 812 (**100%**) | +0.3084 | −0.0649 | 0.669 |

**ρ is flat.** So is the residualised ρ, and AUC slightly falls. The dead contacts were never
dragging the benchmark down, because the merge with frustratometeR already excluded them —
they are largely contacts fR does not have. The reliability deficit was real but invisible to
every comparison in this document.

What Cβ does fix is the **contact-set mismatch**: under Cα ≤ 10 only 74% of FrustX's contacts
exist in frustratometeR's set at all; under Cβ essentially **100%** do. The long-standing
475-vs-389 discrepancy on 1UBQ is a contact-definition artifact, now closed.

### Decision

`DEFAULT_CONTACT_ATOM = "CA"` **stays**, and `--contact-atom CB` is exposed. The case for Cβ
is internal-quality only — a quarter of reported contacts carry an index that is 0/0 or near
it — and the paper specifies Cα. Changing the default would also break comparability with
every run measured so far. This is recorded so the default can be revisited deliberately
rather than drifting; the honest summary is that **Cβ makes FrustX's own output cleaner and
its contact set interoperable, and buys nothing on the benchmark.**

## Rao-Blackwellisation: correct, and not worth wiring in

Eq. 1's numerator needs E[e_ij] over the decoy law. The plain estimator is the sample mean,
which carries Monte-Carlo error in *which identity pairs happened to be drawn*. That law is
known exactly — FrustX shuffles the native sequence, so

    q(a,b) = n_a n_b / (L(L-1))        a != b
    q(a,a) = n_a (n_a - 1) / (L(L-1))

Re-weighting by the exact `q` instead of the empirical frequency is post-stratification:
**same estimand**, and normally lower variance, at zero new Rosetta compute.

**Prerequisite, verified rather than assumed.** The whole analysis needs decoy sequences,
which the `.npz` files do not store. They are recoverable because `make_decoy` seeds a
stdlib `random.Random(k)` whose only consumer is one shuffle (`frustx/decoys.py:130-131`,
`scripts/dump_decoy_samples.py:50`). Checked against PyRosetta directly: `pose.sequence()`
is identical to the one-letter form of `npz["resname"]` on both 1XTQ (169) and 1UBQ (76),
the shuffle is deterministic per seed, and composition is preserved. **So any
identity-resolved analysis of runs already on disk is free** — no re-dump, no PyRosetta.
Worth knowing before anyone budgets a re-run for it.

The weights themselves are tested against brute-force enumeration of every permutation
(`tests/test_pair_weights.py`), not merely against themselves.

### Result

`scripts/rao_blackwell.py` → `results/rao_blackwell.csv`. 360 contacts over 1UBQ, 1XTQ and
1OIV; 25 replicates of n = 250 decoys; endpoint is the SD of the index across replicates at
fixed N, so lower is a strictly better estimator.

| estimator | median SD ratio vs plain | bias vs full-500 |
|---|---|---|
| plain sample mean | 1.000 | −0.0005 |
| post-stratified on the **joint** (a_i,a_j) cell | **1.010** | +0.0031 |
| post-stratified on the **additive** marginals | **0.917** | +0.0022 |

All three are unbiased to within ±0.003.

**Joint post-stratification is slightly worse than doing nothing**, exactly as predicted
before running it. A contact sees a median of 217 of 400 cells with ~1 observation each; a
"cell mean" computed from one sample *is* that sample, and re-weighting by q/p̂ adds variance
faster than conditioning removes it. Sparse strata defeat the method.

**The additive form works**: conditioning on the two identities separately gives 20+20
levels (~25 observations each) instead of 400 cells, for an 8.3% SD reduction — equivalent
to about **19% more decoys, free**.

### Why it is not being wired in

Because 19% more decoys buys nothing anyone needs. The attenuation analysis already showed
decoy sampling noise is not the limiting factor by three orders of magnitude: reliability at
N=500 is 0.987, the ceiling it imposes on any external correlation is 0.99, and the observed
correlation is 0.32. Improving reliability from 0.987 to roughly 0.989 is immaterial to every
question currently open.

So this goes in as a **measurement, not a feature**. Adding a second estimator path to
`frustx/frustration.py` would mean a second thing to keep correct, a second thing to explain
in `run.json`, and a divergence between runs — to move a number that no conclusion depends
on. `scripts/rao_blackwell.py` records the result and the machinery if the situation ever
changes (a low-N regime, or a decoy scheme whose per-contact budget is genuinely scarce —
which pair-local decoys would be).

Note also a limitation of the additive form: it re-weights only the **numerator**, keeping
the plain sample SD as Eq. 1's denominator, and it assumes identities act additively — which
is not obviously safe given that the joint pair cell governs ~70% of within-contact σ. It is
unbiased here empirically; that is not the same as being safe in general.

## The packer-noise control: decoy locality has something to recover

This gates the pair-local decoy idea, so it was run before spending compute on the idea
itself. About 28% of within-contact decoy energy variance is not explained by the identity
of the contacting pair. Two candidate explanations, with opposite consequences:

- **context** — the other ~94% of the sequence being shuffled too. Freezing the environment
  would recover it, and a pair-local scheme is worth building.
- **packer stochasticity** — Rosetta is initialised without `-constant_seed`
  (`frustx/energies.py:37`), so building the *same* sequence twice gives different rotamers
  and different energies. If this dominates, no locality scheme recovers anything, because
  the variance is not information about context at all.

`scripts/packer_noise.py` measures it directly: rebuild ONE fixed decoy sequence 12 times
(4 sequences × 12 rebuilds = 48 poses, ~2 min) so sequence and context are both exactly
constant, and compare that spread to the total across different decoy sequences.

**Verified the measurement is not vacuous first.** If the packer were deterministic the
whole thing would read zero for trivial reasons. It is not: 5 rebuilds of one sequence give
total energies of 112.2, 110.7, 109.6, 120.7, 124.0 REU, max per-contact |ΔE| = 9.5 REU.

### Result: packer noise is 5.4% of the variance

| percentile of packer-var / total-var | |
|---|---|
| p10 | 0.00000 |
| p25 | 0.00001 |
| p50 | 0.00046 |
| p75 | 0.01115 |
| p90 | 0.11459 |
| p95 | 0.21831 |
| **mean / pooled** | **0.0730 / 0.0537** |

410 contacts, 1UBQ, `protocol="min"`. Only 46 of 410 contacts have packer noise above 10%
of their total variance, 14 above 50%.

**The gate opens.** The non-pair variance is overwhelmingly context, not noise, so freezing
the environment is recovering real information rather than suppressing Monte-Carlo scatter.
This is the first positive result for the decoy-locality proposal that does not rest on the
retired class-separation statistic.

It does not, on its own, say the pair-local index is *better* — only that the variance it
removes is not noise. The 14 contacts where packer noise exceeds half the variance are worth
watching: for those, a locality scheme is shrinking σ against a floor it cannot beat.

## Measured cost of a pair-local pose, and what it rules out

`scripts/bench_local_decoy.py`, 12 contacts spanning the shell-size range per structure:

| | shell (residues) | s/pose median | scaling |
|---|---|---|---|
| 1UBQ (n=76) | 8–39 (median 24) | **0.248** | flat — 0.240 s fixed, no shell dependence |
| 1XTQ (n=169) | 11–53 (median 31) | **0.469** | 0.0215 s per shell residue |

**This is 2.5× the estimate the budget was built on** (0.186 s/pose for 1XTQ, extrapolated
from whole-sequence cost by shell fraction). The extrapolation missed a fixed per-pose cost —
cloning the pose and building the task factory — which is why 1UBQ shows no shell dependence
at all: at that size the fixed cost is everything.

Revised budget, at the measured cost:

| experiment | poses | core-h | wall on 2 cores |
|---|---|---|---|
| 1UBQ balanced 224 × 200 | 44.8k | 3.1 | ~1.7 h |
| 1XTQ balanced 224 × 200 | 44.8k | 5.8 | ~3.2 h |
| 1XTQ full 1306 × 200 | 261k | 34.0 | ~19 h — **not affordable** |

The overnight experiment survives at 3.2 h rather than 1.3 h. Full-protein pair-local is out.

## Checkpointing: why the last long run lost everything

`scripts/dump_decoy_samples.py` preallocated the whole tensor and called `savez_compressed`
only after the final decoy, so a job killed at 499/500 lost 100% of the work — which is what
happened to a fig2 run. The comment at `:49` claimed the run was "resumable"; nothing
implemented it.

It now writes `<out>.partial.npz` every 25 decoys and resumes from the first unfinished one,
via write-to-temp-then-`os.replace` so a kill *during* a checkpoint write cannot truncate a
good checkpoint. The native reference is stored in the checkpoint rather than recomputed,
because its repack draws from the unseeded global RNG and recomputing would give a different
E0 for the second half of the run.

Tested by killing a 60-decoy run at 50 and resuming: resumed at 50, completed, no gaps in
the tensor, partial cleaned up.

**Resumption reproduces the sequence ensemble exactly, not the energies bit-for-bit.**
Decoy k's shuffle depends only on `seed=k`, but the packer is unseeded, so decoy k built
after a resume has the same sequence and a slightly different packing. A resumed run is
statistically equivalent to an uninterrupted one, not identical — fine for a sampled
ensemble, but it means a resumed run cannot reproduce an earlier run's exact numbers.

## Tier 2 result: decoy locality does not work, and the old claim was a scale artifact

The last surviving item of the decoy-redesign proposal. `scripts/pair_local_run.py` on
1XTQ: 112 frustratometeR-'highly' + 112 fR-'minimally' contacts, **every reachable identity
pair enumerated** (360 cells — 19 distinct amino acids, minus one singleton diagonal), decoy
mean and σ computed exactly under the shuffle weights rather than sampled. 105 min on 8
cores, 80,640 poses. Endpoints pre-registered in commit `8d6481b` before any index existed.

| scheme | ρ vs fR | additive R² | **residualised ρ** | AUC |
|---|---|---|---|---|
| whole-sequence (production) | +0.2205 | 0.6135 | −0.0108 | 0.6386 |
| pair-local | +0.2144 | **0.7231** | −0.0114 | 0.6307 |

*(These whole-sequence numbers are lower than the +0.2958 / 0.311 reported for full 1XTQ
because this is a balanced subset of 224 extreme contacts, not all 959. The comparison is
within one contact set, which is what matters.)*

**It fails on every endpoint.** ρ down, AUC down, residualised ρ unchanged at essentially
zero — and additive R² **rises** 0.61 → 0.72, i.e. the pair-local index is *more*
residue-additive than the one it replaces. That is the opposite of the intended direction:
FrustX exists to supply contact specificity, and this trades some away.

### The mechanism did not fire

The whole rationale was that freezing the environment removes context variance from σ. The
packer-noise control said 28% of within-contact variance is context rather than packer
noise, which predicts σ falling by roughly 15%. Measured:

- median σ ratio (pair-local / whole-sequence) = **0.968** — a 3% reduction, not 15%
- σ is smaller for only **127 of 224** contacts (57%, barely above a coin flip)
- the two indices correlate at **ρ = +0.906** — freezing the context barely moves the index

So context variance is real (the packer-noise control was right) but is not recoverable by
this construction. Note that `local_decoy`'s shell still repacks — it only forbids identity
changes outside the pair — so some of the context variation survives by design.

### The old "nearly doubled separation" claim, resolved

`docs/method.md` recorded that pair-local decoys "nearly double the separation" between fR's
frustrated and minimally-frustrated contacts, and that was the sole evidence for this whole
line. It **reproduces, and it is an artifact**:

| | whole-sequence | pair-local |
|---|---|---|
| median F gap (a **scale** statistic) | 0.224 | **0.289** ↑ |
| AUC (a **rank** statistic) | 0.6386 | 0.6307 ↓ |

The gap widens by 29% while the ranking gets *worse*. This is exactly the diagnosis recorded
before the run — that a shell-local scheme shrinks σ and inflates |F| mechanically, and that
any scale-based separation statistic would rise without the ordering improving. **The
original claim is withdrawn**; it measured σ shrinkage, not discrimination.

### Two things that did work

**Enumeration replaces sampling entirely.** The pair-local energy is essentially a
deterministic function of (i, j, a_i, a_j) — 11 of 12 tested combinations bit-identical
across processes and seeds. So there is no decoy *sample* at all: all 360 reachable cells
are evaluated and Eq. 1's mean and σ are exact. This retires the question that started this
whole line of work. "How many decoys per contact?" has an answer, and it is *all of them*,
at ~2× the cost of 200 sampled ones.

**The matched native reference was required but nearly free.** Median |E0_local − E0_whole|
is 0.0028 REU. It had to be built — comparing an unprepared native against shell-repacked
decoys is the error `decoys.native_reference` exists to prevent — but it changes almost
nothing numerically here.

### Verdict on the decoy-redesign proposal

All four items are now closed, three negative:

| proposal | outcome |
|---|---|
| decoy counts too low | dead — reliability 0.987, more decoys buy +0.002 |
| minimum decoys per contact | vacuous under shuffling; **answered by enumeration** under locality |
| strategic amino-acid selection | inadmissible (positivity violation); identity governs ~70% of σ |
| localise the decoy | **fails** — ρ, AUC down; additive R² up; σ reduction 3% not 15% |

Decoy construction is not where the FrustX–frustratometeR disagreement lives. Readout scope
is the only axis measured so far that moves contact-specific agreement at all.

## Parallel decoy generation, and the two bugs the review caught

The package had no parallelism: `compute_frustration` and `dump_decoy_samples.py` both ran
a serial `for`. Decoys are independent, so they now fork a worker pool. `n_jobs` defaults
to 1, leaving every archived run and existing caller untouched.

**Measured on 1UBQ, 96 decoys, `protocol="min"`** (4 physical cores + SMT):

| n_jobs | wall (s) | speedup | efficiency |
|---|---|---|---|
| 1 | 60.9 | 1.00 | — |
| 2 | 31.4 | 1.94 | 0.97 |
| 4 | 18.8 | 3.24 | 0.81 |
| 8 | 13.9 | 4.40 | 0.55 |

The sublinearity at 8 is **SMT, not Amdahl**: the serial part (the native reference) is
~1% of runtime, which would allow 7.5×. Per-pose cost rises 0.628 s → 0.665 s at j=4 →
1.07 s at j=8. Four workers is the efficient point; eight buys a further 36% for double
the memory.

### The hazard, and why the obvious implementation is wrong

Rosetta's packer draws from a global RNG and forked workers inherit it, producing
correlated packings and a quietly understated σ — no error, no warning. The trap:
`frustx.energies.init_rosetta()` **cannot** be the worker initializer, because its
`_INITIALISED` module guard is inherited as `True` across the fork, making the call a
silent no-op. Workers call `pyrosetta.init` directly with `-constant_seed -jran <base + w>`.

`fork` is mandatory rather than incidental: `ScoreFunction` is not picklable, so workers
must inherit the pose and both score functions.

### Two bugs found by adversarial review, both real

**1. Parallel runs were not independent samples.** `jran_base` defaulted to a fixed `1`, so
every invocation seeded worker *w* identically and **two runs of the same parallel command
returned a bit-identical ensemble** — maxdiff exactly 0.0, against 0.53 for two serial
runs. Anyone re-running with `-j 8` to grow an ensemble, or merging two runs, would add
**zero information while the apparent standard error fell as though they had**. It also
contradicted this implementation's own docstring. Fixed: `jran_base=None` draws fresh
entropy per invocation, and the resolved value is returned on the result and written to
`run.json`, so a run is reproducible *on demand* rather than *by accident*.

**2. The test suite did not guard the hazard it was written for.** Deleting
`initializer=_worker_init` from the Pool call left all 73 tests passing. The bit-identity
tests set `packing_seed`, and `_seed_packer` overrides the worker's stream as the first
statement of every decoy, so they are blind by construction.

Two attempted fixes *also* failed against a sabotaged copy, and why is worth recording:

- comparing outputs across `jran_base` values — the two calls run sequentially in one
  parent whose RNG has advanced between them, so they differ either way;
- comparing them with the parent RNG pinned — without the initializer the workers do start
  from identical state, but **which worker draws which decoy varies between runs**, so the
  outputs still differ.

No assertion about output values can separate "workers have distinct streams" from "the
scheduler dealt the decoys differently". The test must observe the wiring: a spy on
`_worker_init` reporting each worker's actual seed through a file (a module global would
not survive the fork). That version passes on the repo and fails on the sabotaged copy.

### Why the checks are exact rather than statistical

Packer noise is only 5.4% of decoy variance, so even total RNG correlation across 8 workers
shifts σ by ~2.7%. Measured with a paired design (same 200 decoy sequences per arm, only
the packing RNG differing): `median(σ_parallel / σ_serial) = 1.00000`, and a **maximally
broken** run — workers verified to produce byte-identical packings, sd exactly 0.0 — was
**still indistinguishable** in σ. A statistical check here has no power and must never be
cited as evidence the RNG is correctly seeded.

The exact check: with `packing_seed` set, `decoy_mean`, `decoy_std`, `index` and
`native_energy` are bit-identical between `n_jobs=1` and 2/3/8 — maxabsdiff exactly 0.0,
verified on 1UBQ as well as the test helix. Two things make that possible: `pool.imap`
rather than `imap_unordered`, so the parent accumulates in serial order (float addition is
not associative), and pinning the **native reference** as well as the decoys — omitting the
latter left mean and σ bit-identical while `index` still wandered, since
`index = (mean − E0)/σ`.

Also fixed from the review: a worker dying *without raising* (Rosetta hard-exit, segfault,
OOM kill) hung the parent forever — `Pool` has no broken-worker detection — now bounded by
`decoy_timeout`; unbounded parent-side buffering in `imap` (2.0 GB worst case at n=500,
N=1000, the exact figure the running-sum accumulation exists to avoid) — now windowed to
`4 × n_jobs`; and `n_jobs=-1` silently running serial, which now raises.

`scripts/packer_noise.py` stays serial deliberately, and says so at the loop: its result
depends on the packer being unseeded, and parallelising it by copying this pattern would
make it measure exactly zero.

### The same lesson, learned again on the regression guard itself

`test_parallel_reruns_are_independent_samples` originally asserted that two unpinned
parallel runs produce *different* `decoy_mean` arrays. That looked like the strong form of
the test. It was the weak form, and flaky in both directions:

- **False alarms.** On the 14-residue test helix the packer has few enough rotamer choices
  that it lands on the same answer from either RNG stream. The assertion failed about
  **1 run in 6** in isolation — not an ordering effect, and not caught earlier because a
  single green run says nothing about a 17% failure rate.
- **False passes, which is worse.** Re-run against a sabotaged copy with the original bug
  restored (`jran_base = 1`), the two runs' decoy means still differed
  (−0.60007 vs −0.60957). The assertion the guard rested on could have passed on the exact
  build it exists to reject.

Replaced with two exact assertions in opposite directions: two unpinned runs must **draw**
different `jran_base`, and a **pinned** `jran_base` + `packing_seed` must reproduce
bit-identically. Together they say the seed is what controls the ensemble, which is what
the bug broke. Verified 8/8 green on the real repo and failing with a legible message
(`assert 1 != 1`) on the sabotage.

This is the third time in this document that a statistic over decoy energies has been shown
to have no power to detect an RNG fault. The rule is now explicit: **never assert on energy
values to test seeding; assert on the seed.**

## Reporting both parts: one-body and contact-specific

This document has repeatedly used "additive R²" as a criticism — of `w = 1`, of the
neighbourhood readout, of frustratometeR's mutational mode. That framing was too narrow,
and it is corrected here rather than left standing.

Write the index as

    F_ij  =  [ c + a_i + a_j ]  +  [ residual ]
             one-body / additive     contact-specific

**Both parts are real quantities.** The first is everything a residue carries to all of
its contacts — burial, exposure, local packing — and burial genuinely is most of what
determines whether a contact is well-optimised. Ferreiro's single-residue frustration is
*deliberately* a residue-level index. A 95%-additive index is not measuring nothing.

What a high additive fraction does mean is narrower: **such an index cannot support a
claim about a particular contact**, because nearly all its variation is predictable from
the two residues alone. That is the only thing the earlier sections were entitled to
conclude, and it is why the one-body-residualised ρ was the control that mattered.

So FrustX reports both, per contact, rather than offering them as alternatives:

| column | question it answers |
|---|---|
| `frustration_index` | the index itself |
| `frustration_index_onebody` | is this residue in a frustrated environment? |
| `frustration_index_specific` | is THIS PAIR frustrated, beyond what its residues predict? |

plus `onebody_coefficient` per residue in `residues.csv`, and `additive_r2` in `run.json`.
`frustration_class` stays on the raw index: the 0.78 / −1.0 thresholds are inherited from
AWSEM and calibrated against that quantity, not against a residual.

### Identifiability, which is the one piece of maths that bites

The design matrix for `c + a_i + a_j` is **rank deficient**: adding *t* to `c` while
subtracting *t*/2 from every `a_i` leaves every fitted value unchanged (verified — rank 6
of 7 columns on a 6-residue toy, two coefficient vectors differing by 3.7 giving identical
fits). Therefore:

- `fitted` and `residual` are **unique**, so both per-contact columns are well defined;
- the per-residue coefficients are **not**. `lstsq`'s minimum-norm solution is
  deterministic and reproducible, but the absolute level is arbitrary. Compare residues
  *within* a run, read differences, never compare coefficients across runs.

A fit with fewer usable contacts than parameters is exact by construction and meaningless,
so it reports NaN rather than R² = 1.0.

### Sanity check: the specific part finds chemistry

On 1UBQ (40 decoys, `min`, w = 0), the three contacts with the largest
`frustration_index_specific` are:

| contact | index | one-body | specific |
|---|---|---|---|
| LYS27–ASP52 | +6.22 | +0.15 | **+6.08** |
| ASP39–ARG74 | +5.19 | +0.84 | **+4.35** |
| GLU16–LYS29 | +4.44 | +0.44 | **+4.01** |

All three are **salt bridges**, and LYS27–ASP52 is ubiquitin's well-known buried one. This
is what a contact-specific index should do: a charge pair is minimally frustrated *because
these two residues are together*, not because either is individually buried — and the
decomposition puts almost all of it in the specific column (+6.08 of +6.22) rather than the
one-body one. Not a validation of the index's absolute scale, but a direct check that the
residual is chemistry rather than noise.

---

## The explainer page (`docs/explainer/`)

A standalone page that explains the method and shows it working on 1UBQ. Built in two
halves so neither ends up committed in the other's place:

| File | Tracked in | What |
|---|---|---|
| `docs/explainer/template.html` | git | Markup, CSS, JS. **No numbers.** |
| `docs/explainer/mathml.py` | git | LaTeX → MathML, at build time |
| `docs/explainer/build.py` | git | Assembles the two |
| `scripts/artefact_payload.py` | git | Recomputes the numbers from `results/` |
| `results/artefact/payload.json` | DVC | The numbers |
| `results/artefact/explainer.html` | DVC | The built page |

```bash
.venv/bin/python scripts/artefact_payload.py   # results/ -> payload.json
.venv/bin/python docs/explainer/build.py       # template + payload -> explainer.html
```

### Why the equations are MathML and not text

The first version set the equations as preformatted monospace with HTML `<sub>` tags and
hand-counted spaces. That cannot work: a `<sub>` glyph is narrower than a monospace cell,
so a fraction bar drawn as a row of em-dashes drifts out of register with its own
numerator, and the drift changes with the reader's font.

The fix is `latex2mathml` at **build** time. MathML is native in current browsers, so the
shipped page has no runtime dependency — which matters because the publishing target
allows exactly one external host (Google Fonts) and would not load KaTeX or MathJax from
a CDN. `\[ … \]` and `\( … \)` in the template are converted by `docs/explainer/mathml.py`;
`build.py` asserts the conversion count so a mangled delimiter fails the build instead of
shipping raw LaTeX.

Two things the converter needed help with, both verified by screenshotting headless
Chromium rather than assumed:

- **`\underbrace` does not stretch.** latex2mathml emits a literal U+23DF and leaves the
  stretching to the font's OpenType MATH table. With no math font installed the glyph
  stays a ~19 px stub under a 200 px group. `mathml.py` strips it and the rule is drawn
  with a CSS `border-bottom`, which is exactly as wide as its box by construction.
- **`\langle` alone becomes an `<mi>`**, i.e. a variable, and gets variable spacing.
  `\left\langle … \right\rangle` makes it a real fence.

### Notation fixed while writing it

- The residual in the additive decomposition is written **`s_ij`**, not `r_ij`. `r_ij` is
  already the Cβ–Cβ *distance* in the frustratometeR mode table on the same page.
- `w` (the background weight) is introduced **with Eq. 2**, which is printed in the form
  FrustX implements — `E_ij = e_ij + (w/2)Σ e_ik + (w/2)Σ e_jl` — rather than the paper's
  literal `w = 1` form with `w` appearing later out of nowhere.
- Symbols inside uppercased table headers and label lanes need `text-transform:none`.
  Without it `ρ` renders as a capital Rho, which reads as a Latin `P`, and `w` as `W`.

## What the EGFR atomfrust branch already solved

Recon of `osercinoglu/egfr_analysis_pipeline_withRosetta`, branch
`atomfrust-implementation`, HEAD `efe04f1` (13 commits in the `feat(atomfrust)` series;
the `costbio` upstream has only `main`). Surveyed with five parallel readers. **All line
numbers below are as of `efe04f1`** and will drift; they are recorded because "somewhere
in the contact code" is not a usable pointer.

This is written down because that branch has already paid for the expensive part — not
the code, but the knowledge of *where a heteroatom breaks a protein-only pipeline*. Each
failure below is one we would otherwise have to rediscover by running it.

Scale, for calibration: `atomfrust/` is 60 modules / 22,347 lines, `tests/` 12,679 lines,
844 tests, 11 CLI subcommands. We are not porting that. We are porting five or six
decisions out of it.

### The five mechanics

**1. The contact rule has to branch on residue kind.** `graph.py:330-340`:

```python
selected = np.where(both_protein, d[definition] <= cutoff_A, d_heavy_min <= ligand_cutoff_A)
```

Protein–protein keeps Cα–Cα ≤ 10 Å (`settings.py:149-153`); anything involving a ligand
switches to minimum heavy-atom distance ≤ `ligand_cutoff_A`, default **6.0 Å**
(`settings.py:154`), under a superset ceiling `heavy_cutoff_A` = 8.0 Å
(`settings.py:116-123`).

The trap this avoids is worth stating plainly, because it is silent and it is not
ligand-specific: **a ligand has no Cα, so `d[definition]` is `NaN`, and `NaN <= 10.0` is
`False`.** Not an error, not a warning — every protein–ligand pair simply is not in the
contact set, and the run completes and reports a clean protein-only answer. The same
mechanism eats a protein residue with a missing Cα.

**2. One graph node per Rosetta residue, no per-fragment decomposition** (`pose.py:231-303`).
Node identity is `(chain, resseq, icode)`. A ligand is one node, however many atoms it
has. This is the choice that keeps Eq. 1 and Eq. 2 unchanged — see the degeneracy note
below for the price it charges.

**3. `.params` and the CCD flag.** Pose loading is `generate_nonstandard_residue_set` +
`pose_from_file` (`pose.py:166-187`) with `-in:file:load_PDB_components false`
(`pose.py:44-50`). With that flag left at its default, they loaded 5GMP *with no params
file at all* and Rosetta quietly typed the atoms from its bundled Chemical Component
Dictionary — while the run manifest recorded that curated params had been used. A
provenance lie, not a crash.

**4. Pair energy is kind-agnostic, and `fa_rep` is stored rather than subtracted.**
`energy.py:101-106` computes `e_direct = edge.dot(weights)` for every pair regardless of
kind, and keeps `edge[fa_rep] * fa_rep_weight` in a **separate column**;
`effective_energy` (`:136-148`) does the subtraction at analysis time. This is strictly
better than what FrustX does now — we subtract at measurement time, so changing our mind
about the repulsive term means re-running the decoys.

**5. Eq. 1 and Eq. 2 need no ligand branches at all.** The many-body background applies to
ligand contacts with the same ½ weight (`energy.py:176-184`), and the Z-score
(`analyze/zscore.py:92-99`) and the 0.78 / −1.0 class boundaries
(`analyze/classify.py:86-124`) contain zero kind-dependent code. If the contact set and
the pair energies are right, the index does not know a ligand is present.

### Covalent ligands: bookkeeping only

Anchors are read from the mmCIF `_struct_conn` records (`covalent.py:87`), the anchor
residue is frozen (`pose.py:263-269`), and the anchor edge is forced into the graph
(`covalent.py:262-303`). But `apply_covalent_constraints` (`covalent.py:306`) applies **no
Rosetta constraint whatsoever**, despite the name. The geometry is held only by freezing.
5GMP's own native structure fails their clash gate at 1.81 Å, which is what a covalent
bond looks like to a gate that assumes non-bonded contact.

### The four decoy axes, and which one works

| Axis | What it randomises | Status |
|---|---|---|
| `null` | nothing | control |
| `identity` | protein side-chain identity, ligand fixed | the only one wired to the CLI (`_SUPPORTED_AXES = {"identity"}`, `cli/generate_decoys.py:103`) |
| `pose` | ligand placement, by perturbation | not docking — smina/gnina absent |
| `chemotype` | swaps in a different ligand, MCS-aligned from DUD-E/DEKOIS/MUV | **fails its own positive control**: AUROC 0.333, 0.000 residualised |

The chemotype failure is documented honestly on their side —
`tests/test_decoys_chemotype.py:556` deliberately does not assert that the control passes.
An AUROC of 0.333 is not noise around 0.5; it is anti-correlated, which usually means a
sign or a ranking direction is inverted somewhere. Worth knowing before we build on that
axis.

**The 0.78 Å question is only partly answered there.** `assert_backbone_identical`
(`decoys/identity.py:96`) `continue`s on non-protein residues, and *no test asserts that
ligand coordinates are invariant across identity decoys*. The two mutation routes
disagree: the default `mutation='sequential'` pushes a whole-pose `RestrictToRepacking()`
(`identity.py:358`) and sets `movemap.set_chi(True)` globally (`identity.py:529`), so
ligand torsions do minimise; only the `packer_task` route emits `PreventRepackingRLT`
(`identity.py:474-478`). The drift is bounded in practice only because no `PDB_ROTAMERS`
line is ever written into the params.

### The convergent result, which is the most useful thing here

Their Stage A independently rediscovered **the Eq. 2 degeneracy this document already
records**: with the many-body background on, `e_ij` cancels out of the Z-score, R² =
1.000000 across 38–51 structures.

It is *sharper* with a ligand node than without one. Every protein–ligand contact of a
given ligand shares the same `0.5·B_ligand` term, because they all share the ligand as one
endpoint — so the background contributes a single constant across the entire binding site
rather than a per-pair quantity. Two independent implementations reaching the same
degeneracy from different directions is the strongest support yet for keeping FrustX's
`--background-weight 0` default.

### Two things not to copy

- **The hydrogen and formal-charge handling in the crystal-params path.** Stripping
  hydrogens before typing makes lapatinib's secondary amine type as `Nhis` — an acceptor,
  where chemically it is a donor. And the CCD per-atom formal-charge column is read by
  nobody, so a charged inhibitor is parametrised neutral. Both are silent, and both change
  the electrostatics of exactly the contacts we would be trying to measure.
- **`chem/protonation.py`** — 636 lines, fully tested, called by nothing.

### Status of the science, stated bluntly

There is **no evidence yet that a ligand-aware frustration index tracks binding affinity.**
The prototype's own descriptors are null at n=61 (r = +0.038, p = 0.77). An encouraging
n=19 correlation that once appeared in their `CLAUDE.md` did not survive extension to
n=61 and has been withdrawn. Their `project_status/PROJECT_STATUS.md` predates this and is
stale and misleading; do not read it as current.

So the reason to port from that branch is **mechanical, not evidential**: it tells us how
to make a heteroatom survive the pipeline, not that doing so will show us anything.

### Port list, by cost

**Tier 1 — small, no new dependencies, and none of it needs a ligand present:**

1. Branch the contact rule on residue kind, and make a `NaN` distance loud rather than
   silently `False`. This is a correctness fix for missing-Cα protein residues today,
   independent of ligands.
2. Store `e_direct` and `e_fa_rep` as two columns; subtract at analysis time.
3. Reseed **all three** RNGs per decoy inside the worker (`identity.py:321-326`):
   `random`, `np.random`, and `pyrosetta.rosetta.numeric.random.rg()`. We currently reseed
   the first two. The third is the one that actually drives the packer.
4. Pass `-in:file:load_PDB_components false` at init, and wrap the bare `RuntimeError`
   Rosetta raises on a missing residue type into something that names the residue.

**Tier 2 — medium:** stage-partitioned settings keyed by a `regeneration_key` sha256, so a
settings change invalidates exactly the stages it affects; and superset-once geometry with
contact definitions as columns, so CA/CB/heavy-min are one distance pass rather than three.

**Tier 3 — only if we commit to ligands properly:** vendoring `molfile_to_params.py` plus
`rosetta_py/`. Be warned it is Python-2-era — `open(f, 'rU')` raises on 3.12, and their
`environment.yml` pins 3.10 for this reason.

### Tier-1 ports: what was actually implemented, and what turned out to be unnecessary

Working through the tier-1 list above. Two of the four landed as changes, one was already
solved here, and one had to be done differently than the recon suggested.

**1. Contact rule / NaN.** Done, but narrower than billed. FrustX never *produces* a NaN
coordinate today: `load_ca` drops CA-less residues (deliberately, and tested at
`tests/test_contacts.py:104`) and the pose path raises. So this was a latent hazard, not a
live wrong answer, and the fix is a guard in `contact_pairs` rather than a rewrite of the
contact rule. `np.nan <= cutoff` is `False` with no warning, so a NaN placeholder would
have removed a residue from the contact set while the run reported a clean result.
`isfinite`, not `isnan`: an `inf` coordinate compares *True* and would invent contacts
with everything. The pose-side error now names *which* residue rather than leaving
Rosetta's `ResidueType HOH does not have an atom CA`.

**2. RNG reseeding — already done here, and done better.** The recon recommended copying
atomfrust's three-way reseed. Checking first: `_seed_packer` (`frustx/frustration.py:253`)
already reseeds `pyrosetta.rosetta.numeric.random.rg()` per decoy in both the serial
(`:423`) and parallel (`:295`) paths, `_worker_init` gives each worker its own `-jran`
stream, and `decoys.py:139` uses an explicit `random.Random(seed)` *instance* rather than
reseeding the global module — which is the stronger pattern, since nothing else in the
process can perturb it. `np.random` is not used in the decoy path at all. **No change
made.** Recorded because "port the reseeding" would otherwise look like an outstanding
task forever.

**3. `-in:file:load_PDB_components false`.** Done, in `DEFAULT_INIT_FLAGS`
(`frustx/energies.py:47`), and it propagates to workers automatically because
`_worker_init` reuses that constant rather than a copy.

The flag matters more than it sounds. Reproduced on FrustX's own defaults: a bare `BNZ`
HETATM, with **no `.params` file anywhere in this repo**, loads as a real fifth residue —
Rosetta falls back to its bundled Chemical Component Dictionary and builds it with atom
types and charges nobody chose. `-ignore_unrecognized_res` does not ignore it. With the
flag, it is dropped, which is the honest outcome when we have no parameters for it.
`tests/test_init_flags.py` asserts both halves in subprocesses, since Rosetta's init is
once-per-process and the point is to compare two settings.

**4. Splitting `e_direct` from `e_fa_rep` — done, but NOT the way atomfrust does it.**

atomfrust scores once with `fa_rep` present and subtracts at extraction. Measured here,
that route perturbs the surviving terms by ~1 ULP (2.2e-16 relative). That is normally
irrelevant, and here it is not: this project deliberately holds serial and parallel runs
to **bit-identical** agreement rather than ~1e-12 (see the ordered-`imap` note), and every
existing number would have shifted in the last bit.

So instead: a second, `fa_rep`-only score function (`make_fa_rep_score_function`,
`frustx/energies.py`) and a second scoring pass. Cost measured at **3.2 ms against a
~700 ms decoy, ~0.5%**, which is why it is unconditional rather than behind a flag. The
default index is bit-identical to before the change, verified by running the pre-change
code from a git worktree and comparing arrays.

**The covariance is the part that is easy to get wrong.** Two columns alone do not let
anyone recombine, because

    Var(E + wR) = Var(E) + w² Var(R) + 2w Cov(E, R)

so `FrustrationResult` carries `fa_rep_cov` as well, accumulated as a running `sum(E·R)`
with the same 1/N convention as the variance. `index_at_fa_rep(weight)` then reconstructs
the index exactly. The sign is the caller's, because it depends on what `sf_measure` did:
`+1` adds the term back to a default run, `-1` removes it from a `--packing-frustration`
one.

Validated against a *real second run* measured with `fa_rep`, not against the same
arithmetic that produced it: **max relative difference 4e-14** over the 126 finite entries of the index
matrix, in both directions. (126 is entries, not contacts -- the matrix is
symmetric and the test peptide has 55 contacts.) The `contact_table` gets the raw parts as columns and deliberately *not* a
ready-made recombined index, which would invite adding two sigmas downstream.

One trap worth recording for whoever writes the next test here: the first version of this
check used a poly-alanine peptide and showed a 2e-8 discrepancy. That was not an error in
the recombination — poly-A decoys come out near-identical, `decoy_std` collapses to ~1e-9,
and every index is a 0/0. Multiplying the difference back by σ gave 4.4e-16, i.e. machine
precision. **Test the energy pipeline on a heterogeneous sequence**; a degenerate ensemble
will manufacture alarming-looking relative errors out of nothing.

**What adversarial review caught, after the above was written and passing.** Recorded
because two of the three were silent-wrongness bugs of exactly the kind this document
exists to accumulate.

- **`index_at_fa_rep(weight=1.0)` had the wrong default half the time.** `+1` is correct
  only when `sf_measure` removed `fa_rep`. Called on a `--packing-frustration` result it
  double-counts — and the output does not look broken: measured on the test peptide it
  correlates with the true index at **r = 0.99**, same signs, same shape, twice the
  scale. `FrustrationResult` now records `fa_rep_in_measure`, the default flips the run
  to its complement rather than assuming a direction, and an explicit weight of the
  wrong sign raises.
- **The `fa_rep` weight was read from `DEFAULT_WEIGHTS` while `sf_measure` is
  caller-supplied.** `compute_frustration(sf_measure=make_score_function(weights="score12"))`
  is a legal call, and score12's `fa_rep` weight is 0.44 against ref2015's 0.55 — a 25%
  error in the column with nothing raising. `compute_frustration` now takes `weights` and
  cross-checks it against `sf_measure` wherever `fa_rep` survived there to be compared.
- **`scripts/dump_decoy_samples.py` was left behind.** It drives `_WORKER` and
  `_worker_decoy` directly, so the new 3-tuple and the new `sf_fa_rep` key broke its
  `n_jobs > 1` path. It fails loudly, but hours into a checkpointed run. Fixed. The
  general lesson: `_WORKER` is a private dict with an external consumer, which is a
  coupling worth remembering before changing that payload again.

A test-quality note from the same pass, worth generalising: `assert abs(cov).max() > 1e-9`
was **vacuous as written** — it would have passed unchanged had `index_at_fa_rep` ignored
the covariance entirely, since it only asserts that a stored array is nonzero, not that
anything reads it. It now recomputes the index with the cross term dropped and requires
that version to *fail*. Asserting that a quantity exists is not asserting that it is used.

### Tier-2, part 1: regeneration keys, and the resume bug they were written for

The tier-2 item was "stage-partitioned settings keyed by a `regeneration_key` sha256".
Mapping the repo first turned up a live corruption path rather than a hypothetical one,
which decided the shape of the work.

**The bug.** `scripts/dump_decoy_samples.py` resumed a checkpoint after validating its
SHAPE and nothing else — decoy count and residue count, two integers. So resuming a `min`
checkpoint with `relax` on the command line passed the check and mixed two ensembles into
one tensor. Worse: `native` is loaded from the checkpoint and deliberately *not*
recomputed (there is a good reason — recomputing it draws from an unseeded global RNG and
would silently give a different E0 for the second half of the run), so the reference
stayed pinned to the first protocol while every post-resume decoy used the second. The
reference and the ensemble end up prepared differently, which is exactly the failure
`frustx/decoys.py:16-19` exists to prevent, reintroduced at the resume boundary.

**Stages are partitioned because the settings genuinely differ per stage.** From tracing
`compute_frustration`, not from the CLI's flag list — the two disagree:

| Stage | Reads |
|---|---|
| `structure` | structure content, init flags |
| `contacts` | + contact_atom, cutoff, min_seq_sep |
| `ensemble` | structure, init flags, weights, protocol, repeats, seed, n_decoys, packing_seed |
| `measurement` | all of `ensemble`, + background_weight, readout, packing_frustration |

`ensemble` is the expensive stage (~700 ms per decoy) and reads a **strictly smaller** set
than `measurement`. So changing `--readout` must not invalidate decoy structures, and the
test asserts that equality as hard as it asserts the inequalities — a key that
over-invalidates is not safe, it is just useless, because it discards expensive work.

**One dependency is conditional and a static table would get it wrong.** At the default
`readout="pair"`, `apply_readout` returns `E` untouched, so the contact map never reaches
the energies and `cutoff` / `min_seq_sep` / `contact_atom` are *not* measurement inputs.
At `readout="neighbourhood"` they are. `regeneration_key` resolves that at call time.

**Details that are load-bearing:**

- **A missing setting raises rather than defaulting.** This is the one way the module
  could actively cause the bug it prevents: a key computed over a silently absent field
  compares EQUAL to one computed with the field present at its default.
- **The structure is keyed by content, not path.** Two runs against different edits of the
  same filename are otherwise indistinguishable, and a path proves nothing about what was
  in the file.
- **A keyless checkpoint is refused, not trusted.** Files written before the guard existed
  carry no key, and "no key" is indistinguishable from "written under different settings".
- **A mismatch exits rather than starting fresh.** Silently overwriting hours of decoys
  because a flag was mistyped is the more expensive of the two mistakes. Verified: exit
  code 1, checkpoint preserved on disk, no output written.
- **The settings travel with the key** in the `.npz`, so the refusal can print
  `protocol: 'min' -> 'relax'`. A comparison can only say yes or no; that is not enough to
  decide whether to delete the file or fix the command line.

Verified end to end, not only in unit tests: a real run interrupted after 4 of 8 decoys
resumes correctly under the same protocol, and refuses under a changed one with the
diagnosis above.

**Not built:** a general caching layer. Nothing here decides what to recompute — it only
answers "were these produced under the same settings as those?". FrustX has exactly one
artefact today that outlives a run, and that is the checkpoint.

### Tier-2, part 2: superset geometry, and two bugs found on the way

**What was NOT done, and why.** The recon's framing was "CA/CB/heavy-min in one distance
pass rather than three". On inspection that framing is wrong for the shared-work reason it
implies: CA, CB and heavy-min are *different* computations over *different* coordinate
sets, so there is no common subexpression to hoist. The real waste is repeated selection —
`scripts/contact_definition.py` sweeps cutoffs 10.0 / 9.5 / 9.0 over seven structures and
calls `contact_pairs` each time, paying for 21 full distance passes where 7 would do.
Nothing about a cutoff changes the distances.

So `ContactGeometry` computes distances once per *definition* and makes selection cheap,
rather than pretending three definitions share arithmetic they do not.

**`contact_pairs` was deliberately left alone.** `tests/test_contact_atom.py:44` pins that
it takes coordinates and never an atom name, and every one of the eleven callers in
`scripts/` uses it that way. The selection logic was extracted into `select_pairs`, which
takes an already-computed matrix; `contact_pairs` is now a thin wrapper. Definitions are
columns rather than an argument to one rule because they are genuinely not
interchangeable: CA and CB are per-residue representative *points*, heavy-min is a
*minimum over atom pairs*, and a mixed protein/ligand structure needs both at once.

**`min_heavy_distances` is the piece tier 3 needs.** It is also the only quantity here
that is not n²: at ~8 heavy atoms per residue, a full atom-atom matrix is 512 MB at
n=1000 and ~13 GB at n=5000, so it is chunked over residues. The reduction is two segment
minima via `np.minimum.reduceat`, atoms → residues along each axis in turn.

One trap that would have been silent: **`reduceat` does not return the identity for an
empty segment** — it returns the element at that index. A residue with no heavy atoms
would therefore have reported some other residue's distance rather than raising. Refused
explicitly.

**Two real bugs found by mapping, both in `contacts.py`:**

- **The O(n²) memory comment undercounted by 7×.** It costed the surviving `dist` array
  and ignored the two temporaries: `delta` is 24n² bytes and `delta**2` another 24n²,
  against 8n² for the result. Measured peak was **56 MB at n=1000 and 224 MB at n=2000**,
  where the comment claimed 8 MB and "200 MB at n=5000" — the true figure there is ~1.4 GB.
  The chunked kernel now measures 22 MB and 61 MB for the same cases, and the ratio
  improves with n as the surviving array comes to dominate.
- **`load_ca`'s atom validation ran after the parse loop and against a hardcoded literal.**
  `if atom not in ("CA", "CB")` sat below the loop that had already indexed `res[atom]`, so
  an unknown atom name that happened to exist in the structure was silently collected
  first and only rejected at the end — and the literal was free to drift from
  `config.CONTACT_ATOMS`, which is what `frustration.py` checks against. Both fixed.

**Bit-exactness held.** The chunked kernels are checked against the naive forms they
replace with `array_equal`, not `allclose`, because this project holds serial and parallel
runs to bit-identical agreement and "the same to 1e-15" is not the same. `min_heavy`
likewise matches a brute-force double loop bitwise, on ragged residues of 1 to 33 atoms.

**Verified on a real ligand.** `heavy_atom_coords_from_pose` and `residue_kinds_from_pose`
were checked against a peptide with ATP appended: heavy counts 4 for glycine and 33 for
ATP, `is_protein` False for ATP alone, and the ligand gets finite heavy-atom distances to
every protein residue where the CA rule gives it no coordinate at all.

## Tier 3: ligands

Designed by three parallel agents and adversarially verified by three more. One design
came back **flawed** and two **sound-with-fixes**, which is the reason to record what was
implemented rather than what was proposed.

### What tier 3 actually requires, established by probe

Rosetta **ships params** for ATP (33 heavy-indexed atoms, no CA), ZN, MG and other
coenzymes in the default `fa_standard` set. No vendoring of `molfile_to_params.py`, no
`-extra_res_fa`, no licensing decision. That removes what looked like the gating item:
params generation is needed only for *arbitrary novel* ligands, not to build or test any
of the machinery.

With a ligand in the pose:

| Layer | Status |
|---|---|
| Pair energies | **Already work.** `pair_energy_matrix` is kind-agnostic; 8 of 10 protein-ligand pairs carried nonzero energy on first probe. |
| Contact geometry | Blocked — a ligand has no CA. **Now fixed.** |
| Decoy generation | Blocked — `pose.sequence()` returns `Z` for ATP and `shuffle_sequence` permutes it like an amino acid, so `make_decoy` raises `IndexError`. **Not yet fixed.** |

### A bug in the tier-2 geometry, caught by the review

`heavy_atom_coords_from_pose` used `nheavyatoms()` as the atom count. **That counts
VIRTUAL atoms**, and for exactly the residues this function exists to serve: Rosetta's
metal params carry a shell of them at the coordination positions — ZN reports 5 heavy
atoms and has **1 real**, MG reports 7 and has 1. They sit 1.0–2.2 A off the metal, so
feeding them to a minimum-distance rule shrinks every distance to that ion and invents
contacts no atom supports. Measured: **2.04 A of phantom shortening and 6 reported
contacts against 5 real** at 6 A on a zinc.

ATP carries two virtual atoms as well, but they sit 0.00 A from real atoms and cannot move
a minimum — **which is why testing with a ligand and not an ion hid this completely**.

The test that should have caught it was circular: it asserted the extracted count equalled
`nheavyatoms()`, the very quantity under suspicion. It now asserts against the ion's
chemistry — a zinc is one atom — and a second test builds the buggy coordinates explicitly
and requires the two to disagree.

### The contact rule

    protein-protein   dist_protein <= cutoff          (the paper's Ca-Ca rule, 10.0 A)
    anything else     dist_heavy   <= ligand_cutoff   (minimum heavy-atom, 6.0 A)

**6.0 A is not arbitrary** — it is Rosetta's etable interaction radius (`fa_max_dis`), so
it is where REF2015 pair terms stop firing between heavy atoms. Every one of 186
protein-protein pairs within 6.0 A minimum-heavy on 1XTQ carries nonzero `e_ij` (0.0%
dead), against 7.5% dead at 7.0 A and 30.9% at 8.0 A.

**But it is not calibrated to match the protein branch, and that is stated rather than
hidden.** Only 73% of the pairs `CA <= 10.0` admits also satisfy `heavy-min <= 6.0`, and
the count-matched equivalent is nearer 7.0 A. Hydrogens also carry REF2015 past 6.0 A of
heavy separation — protein-protein pairs show nonzero `e_ij` out to 7.85 A. So 6.0 A is
the *conservative* choice, not the count-matched one, and it is a parameter because the
paper says nothing about ligands at all.

### Sequence separation was measured on the wrong quantity

`min_seq_sep` counted **flat list position**. Rosetta gives a HETATM the same chain letter
as the protein it sits in, so a ligand parsed between two covalently adjacent residues
inflated their separation from 1 to 2 and **un-dropped a pair `min_seq_sep=2` exists to
drop**. Measured on a real structure: moving a MG from the end of a chain to mid-chain,
coordinates unchanged, changed which pairs survived.

**Two fixes were needed, and they correct opposite errors:**

1. Count separation in **polymer ordinals** (`cumsum(is_protein) - 1`), so a heteroatom
   contributes zero separation instead of one.
2. Apply the mask **only to protein-protein pairs**. The ordinal alone is not enough — it
   gives a ligand its predecessor's ordinal, so separation from the neighbouring residue
   becomes 0 and every protein-ligand contact at the insertion point silently vanishes.

Fixing one and not the other leaves the opposite bug, and both are silent. The second was
found by the test suite, not by the review: the review's recommendation was fix (1) alone
and explicitly called the conjunction "redundant".

### Ligand-ligand pairs are excluded by default

Not because they are uninteresting. **Eq. 1 cannot be computed for them.** A decoy differs
from the native only by a shuffled protein sequence and a repack, so a pair with no
protein endpoint has no shuffleable identity and often no movable chi; its decoy spread is
floating-point residue rather than a sampled distribution.

**That is worse than a missing row.** `np.where(std > 0, ...)` (`frustration.py:141`,
`:653`) is a BIT-LEVEL test: a sigma of 1e-16 passes it and Eq. 1 returns a clean finite
number that `classify()` will label minimally or highly frustrated. Protein-ligand pairs
are kept — those have a varying endpoint.

### Open, and NOT changed unilaterally: the `std > 0` guard

The review argues `std > 0` should be `std > tol`, with `tol` tied to the 0.02 dead-contact
threshold `config.py` already documents. The mechanism is real and reproducible —
catastrophic cancellation in `total_sq/N - mean^2` leaves a spurious nonzero sigma (1.7e-07
measured on synthetic identical energies), which passes the bit-level guard.

**This is left alone deliberately.** `config.py` records that 1824 of 7687 contacts (23.7%)
across the six GTPase runs have sigma <= 0.02. Changing the guard would turn those from
finite indices into NaN — a change to published numbers and to what the tool claims, which
is a scientific decision, not a bug fix. Recorded here for that decision to be made
explicitly.

### Tier 3: decoy generation with a ligand present

The remaining blocker. `pose.sequence()` returns `Z` for ATP, `shuffle_sequence` permuted
it like an amino acid, and `make_decoy` died with a bare `IndexError`. **The crash was the
lucky outcome** — the same mechanism would otherwise have tried to mutate the ligand into
an amino acid and some amino acid into ATP.

**Three kinds of residue, not two.** Conflating them is how a modified residue gets
silently mutated:

| Kind | Predicate | Treatment |
|---|---|---|
| canonical amino acid | `type().is_canonical_aa()` | designed — the shuffle may change its identity |
| non-canonical **protein** | protein but not canonical | identity fixed, side chain still repacks |
| everything else | `not is_protein()` | frozen |

`is_canonical_aa()` rather than `is_protein()`: a spin label like R1A reports
`is_protein=True` and `name1 'Z'`, so it crashes `apply_sequence` exactly as a ligand does
**with no ligand anywhere in the file**. And `is_protein()` rather than `not is_ligand()`:
Rosetta reports `is_ligand()` **True for water**, so `is_ligand` cannot discriminate.

**Freezing takes three separate mechanisms, and none is redundant:**

1. `prevent_repacking()` on the packer task. `restrict_to_repacking()` alone leaves a
   ligand *packable* — measured, not assumed.
2. `set_chi(i, False)` on the MoveMap, for minimisation.
3. A `TaskFactory` with `PreventRepackingRLT` for FastRelax. **The MoveMap does not bind
   FastRelax's internal packer** — with chi explicitly off, FastRelax still repacked the
   ligand. A future refactor that drops this "because the movemap already freezes chi"
   reintroduces the bug, and it is invisible: the ligand simply lands in a different
   rotamer and every energy involving it shifts.

**The silent asymmetry to watch for** is freezing the decoys but not the native reference.
`repack()` and `apply_sequence()` are separate code paths, so fixing one and not the other
is easy, and the result is an E0 measured against an ensemble prepared differently — with
nothing raising. Both have their own test.

**`freeze_ligand` is a parameter, defaulting True, because the science is genuinely
unsettled** — the EGFR branch's two mutation routes disagree about whether ligand torsions
minimise, and nothing there asserts invariance. Freezing means the index answers "how
frustrated is this protein around a ligand held where the crystal put it", which is the
question a fixed backbone already commits us to. Letting it relax answers a different
question and makes the decoy ensemble's ligand pose a free variable. There is a test that
`freeze_ligand=False` **actually lets the ligand move**, so the freeze tests cannot pass
for the wrong reason.

**Verified end to end**: a full `compute_frustration` on a peptide-ATP complex produces
protein-ligand contacts with finite indices and classifications, ligand atom displacement
exactly 0.00 A across the native reference and every decoy, backbone unchanged, and the
all-protein path **bit-identical** to before the change.

### What the provenance work caught immediately

Adding `freeze_ligand` to `STAGE_INPUTS` broke six checkpoint tests at once, because
`regeneration_key` raises on a missing setting rather than defaulting. That is the module
behaving exactly as designed — the alternative is a key computed without the field
comparing EQUAL to one computed with it at its default.

`STAGE_INPUTS` now also asserts at import that **`measurement` contains every `ensemble`
field**. A measured artefact is produced *from* decoy structures, so anything changing
what a decoy IS must invalidate the measurement too; adding a field to one and not the
other would let a measurement be reused across structurally different ensembles.

## Novel ligands: parametrising arbitrary chemistry

The ligand work up to this point assumed the ligand was one Rosetta already knows. That
covers ATP, ZN, MG, SAH, UDP and the other coenzymes in the default `fa_standard` set,
and it covers nothing a virtual screen or a co-folding model produces. Those are the
actual use case: score a pocket *after* docking, on the compound that was docked.

### `molfile_to_params.py` is not needed, and that was the surprise

The conventional route to Rosetta parameters is `molfile_to_params.py`, which lives in the
Rosetta source tree, is not in the PyRosetta wheel, and is licensed such that vendoring it
into this repo would be redistribution. That was recorded as a blocker needing a licensing
decision. It is not a blocker, because it is not needed.

PyRosetta exposes Rosetta's own C++ SDF/MOL reader:

```python
from pyrosetta.rosetta.core.chemical import sdf
mutable = sdf.convert_to_ResidueTypes("AQ4.sdf", False)[1]   # vector1, 1-INDEXED
```

which does the entire job — atom types, partial charges, connectivity, internal
coordinates. Measured on erlotinib: `name 'AQ4', natoms 52, nheavyatoms 29,
is_ligand True`. `pyrosetta/toolbox/load_ligand.py:96` does `import molfile_to_params`
and is therefore dead on a stock install; that dead import is what made the tool look
mandatory.

### Registration is per-pose, and that is what makes fork work

The type goes into a `PoseResidueTypeSet` layered over `fa_standard`, not into the global
set. Three separate reasons, each independently sufficient:

* The ChemicalManager's `fa_standard` set is a const owning pointer. It cannot be added to.
* `-extra_res_fa` is honoured only at the **first** `pyrosetta.init` of a process. FrustX's
  decoy workers re-init after fork (`frustration.py:381-384`), so a flag-based route would
  have to be threaded into `DEFAULT_INIT_FLAGS` and would change the protein-only path.
* Because the type travels *inside the pose*, the workers need to know nothing about
  ligands. `DEFAULT_INIT_FLAGS` is untouched and the all-protein path is unchanged.

Verified: a runtime-built type survives `fork` plus the worker's own re-init, bit-identical
at n_jobs 2, 3 and 8.

### Atom naming: the trap, and why it does not bite

Rosetta matches PDB `HETATM` records to a residue type **by atom name**. An SDF carries no
PDB atom names, so Rosetta invents `C1, C2, ...` while 1M17's HETATM block says `C13, O2,
N4`. Nothing matches. The fix is `remap_pdb_atom_names(True)`, which matches by geometry
instead, and the failure without it is loud rather than silent — measured:

```
remap=True  -> ligand coordinates match the HETATM block to 0.0000 A
remap=False -> RuntimeError: too many tries in fill_missing_atoms!
```

The constructor default is already `True`. It is set explicitly anyway, because the whole
topology mode depends on it and a default is not a guarantee.

### Two indices, not one: `name` and `name3`

Rosetta indexes residue types by `name` **and** by `name3`, separately, and conflating them
is the easiest way to write this module wrong. An SDF title longer than three characters
gets a `name3` that is only a truncation, and the truncation is not a lookup key:

```
fix/longtitle.sdf -> name='ligand_1'  name3='lig'
prts.has_name('ligand_1') = True
prts.has_name('lig')      = False
prts.name_map('lig')      -> RuntimeError: The residue lig could not be generated.
```

Every real docking output has a title longer than three characters, so an implementation
keyed on `name3` fails on all of them and works on hand-made test fixtures.

The same split governs the shadowing guard. Registering a type whose name collides with a
curated one is accepted **without complaint**, and the new type wins the lookup:

```
'ATP'  has_name=True   has_name3=True
'SAH'  has_name=False  has_name3=True    <- a name-only guard misses this
' ZN'  has_name=False  has_name3=True       (note the padding)

fix/atpname.sdf registered without complaint; name_map('ATP').natoms() = 6
                                             (the curated ATP has 45)
```

So a user who fetches `ATP_ideal.sdf` from RCSB would silently score the neutral tetraacid
in place of Rosetta's fitted, charged nucleotide — in a kinase P-loop, where it matters
most. `refuse_shadowing` checks **both** indices and refuses.

### Hydrogens: a global count is not a gate

Rosetta types atoms from their connectivity, so a ligand missing hydrogens is typed wrongly
and every energy computed from it is wrong in a way nothing downstream can detect. The
obvious gate — "does this file have any hydrogens" — is defeated by a single polar
hydrogen, which is exactly what AutoDock and PDBQT-derived pipelines emit. All three of
these parse without complaint:

| file | H | C1 type | C1 charge | O1 type | O1 charge |
|---|---|---|---|---|---|
| `methanol.sdf` | 4 | CH3 | −0.234 | OH | −0.624 |
| `methanol_polarH.sdf` | 1 | **CH1** | **+0.017** | OH | −0.553 |
| `methanol_noH.sdf` | 0 | **CH1** | **+0.335** | **OOC** | **−0.335** |

So the test is per-atom valence saturation, summing bond **orders** — a triple bond counts
3, or erlotinib's terminal alkyne `C#CH` would be flagged as missing hydrogens. Verified to
flag nothing across all 29 heavy atoms of erlotinib, and to catch both corrupt methanols.
FrustX refuses rather than repairing: adding hydrogens is a choice about pH belonging to
whoever prepared the ligand, and it would mean an RDKit or OpenBabel dependency.

### Where the typing is genuinely not faithful — OPEN

`fa_standard` has no sp (linear) carbon type. Erlotinib's alkyne carbons come back as
**COO — a carboxyl carbon — carrying +0.683** where the true partial charge is near zero:

```
C1  Carbon  type=COO  q=+0.683  nH=1   (this is C#CH)
C2  Carbon  type=COO  q=+0.683  nH=0
```

That is a ~0.7 e error on two atoms, and `fa_elec` is a pairwise product of charges, so it
does not stay local. `check_chemistry` warns rather than refusing.

**Whether it matters for a frustration index is not settled, and the argument cuts both
ways.** The ligand is frozen and identically typed in the native and in every decoy, so a
constant offset cancels in the Z-score of Eq. 1. But the cancellation is exact only if the
mutated protein residue does not change how that atom is screened, and a shuffled pocket
changes the local dielectric environment by construction. **Not resolved. Do not put a
protein–ligand frustration index in a paper until it is.** The escape hatch for any contact
that matters is a curated `.params` file, which registers through the same call.

### Validation on the EGFR set

`scripts/egfr_ligand_check.py`, output in `results/egfr_ligand/`. Four kinase-domain
complexes with drug-like inhibitors, none of which Rosetta knows: 1M17/erlotinib,
2ITY/gefitinib, 1XKK/lapatinib, 3POZ/TAK-285. All four load with **max coordinate
deviation 0.0000 Å** from the structure's own HETATM block, and FrustX's protein–ligand
contact count agrees exactly with an independent numpy calculation done from raw PDB text:

| structure | ligand | heavy | chi | contacts | independent | frozen | unfrozen | serial=parallel |
|---|---|---|---|---|---|---|---|---|
| 1M17 | AQ4 erlotinib | 29 | 10 | 25 | 25 | 0.0000 Å | 2.7333 Å | bit-identical |
| 2ITY | IRE gefitinib | 31 | 8 | 27 | 27 | 0.0000 Å | 1.7559 Å | bit-identical |
| 1XKK | FMM lapatinib | 40 | 11 | 37 | 37 | 0.0000 Å | 2.3067 Å | bit-identical |
| 3POZ | 03P TAK-285 | 38 | 12 | 37 | 37 | 0.0000 Å | 3.2515 Å | bit-identical |

Every check passes on all four. The `unfrozen` column is what gives the `frozen` column its
meaning: each inhibitor has 8–12 rotatable bonds and moves 1.8–3.3 Å once the freeze is
lifted, so 0.0000 Å is evidence rather than a property of a ligand that could not move.

One residual, unrelated to ligands: 3POZ produces 36 finite indices out of 37 ligand
contacts. That is the `std > 0` guard, on a contact whose decoy sigma is zero — the same
open question recorded above, now visible on the ligand face.

The first version of that cross-check disagreed on 1M17, 25 against 26, and **FrustX was
right**. CYS A 751 is modelled in two half-occupancy conformers; Rosetta keeps altloc A at
6.017 Å, outside the 6.0 Å cutoff, while the naive numpy check pooled both altlocs and
caught conformer B at 5.267 Å. A cross-check against raw PDB text has to filter altlocs or
it is not measuring the same thing.

### Ligands are spectators, and the control has to be able to move

Ligands enter the contact map and are never mutated, repacked or moved. The machinery for
this already existed and needed no change; what was missing was evidence that it holds for
a type built at runtime rather than one from the curated set.

The freeze measures **exactly 0.0000 Å** displacement across the native reference and every
decoy. That number is worthless on its own, and the obvious control is degenerate on a
hand-built fixture: methanol has `nchi=0`, so unfreezing it moves it by 1.8e-15 Å and the
control passes with the feature deleted. A real drug fixes this — lapatinib has **11 chi
angles**, and unfrozen it moves **2.29 Å**. So the EGFR script carries the displacement
control and the unit tests carry a `PackerTask` assertion instead, pinned against a plain
`RestrictToRepacking` that *does* leave the ligand packable.

### Provenance

`ligands` is in all four stages of `STAGE_INPUTS`, valued as a sorted list of **content
hashes** — `--ligand hit.sdf` names a different molecule tomorrow if the screen was re-run,
and a key over the path would happily reuse an ensemble built for the old one. Adding the
field broke **23 tests**, which is `regeneration_key` refusing to hash a settings dict that
omits a field it depends on, working exactly as designed. Every existing `.partial.npz`
stops resuming; there is no migration, same as `freeze_ligand`.

Separately, `ligand_cutoff` was missing from the measurement key at a neighbourhood
readout, where the contact map enters the energies through `row_i + row_j`:

```
readout=neighbourhood, ligand_cutoff 6.0 vs 9.0
  contacts key differs:    True
  measurement key differs: False     <- wrong
  (control) cutoff 10 vs 8 differs:  True
```

Fixed, with a test asserting both halves — it must *not* invalidate at `readout="pair"`,
where `apply_readout` returns the energies untouched.

One more gap closed: `scripts/dump_decoy_samples.py` builds a settings dict that **no test
covered**, so it could sit dead while pytest was green. It had already broken twice this
way. There is now an AST-level test asserting its keys still satisfy `STAGE_INPUTS`.

### A ligand that touches nothing, found by using the feature

`--ligand-placed` appends a ligand at the coordinates its own file carries, which is the
post-docking case. Fed an RCSB `_ideal.sdf` it produced **zero protein-ligand contacts and
no error** -- the run completed and reported a perfectly clean protein-only answer, which
is precisely the failure class the kind-aware contact rule was written to stop.

The cause is not a bug in the loading: an `_ideal.sdf` is an idealised conformer generated
near the origin, *not* the crystallographic or docked pose. Placed correctly the same file
is fine; placed from the wrong frame it lands nowhere near the protein. Since a genuinely
non-contacting ligand is conceivable, this warns rather than refusing:

```
frustx: warning: ligand AQ4 at B1 makes NO contacts with the protein at
        --ligand-cutoff 6.0 A. If this was --ligand-placed, its file's coordinates
        are probably not the pose you meant -- an RCSB _ideal.sdf is a generated
        conformer near the origin, not the crystallographic or docked position.
```

Verified non-vacuous: it does not fire for the same ligand in topology mode, where the
structure supplies the pose and the contact count is 25.

### Still deferred, by decision

Ligand-flexible decoys and conformer libraries (decoys shuffle amino-acid identity only, so
a ligand relaxing into a shuffled pocket answers a different question); covalent-ligand
bookkeeping; `.mol2` input (Rosetta's reader returns an empty result for it and raises
nothing, so it is refused by extension); RDKit/OpenBabel as dependencies.
