# RQ1 notebook extraction record

The three notebooks remain unchanged at their historical paths. The source snapshot
is `1937995fb78d979a66e237b7ab7facbe20ca0513`. Cell indexes below are **zero-based**.
The new modules are extracted/reconstructed implementations, not the programs
originally executed to produce the saved evidence. No real-data cell was executed
during extraction. Their functions have no dataset-wide runner or output writer.

| Historical notebook/cells | Current Python representation | Boundary |
|---|---|---|
| 01, cell 3 | `dataset_audit.check_pairs` | Original stem comparison and printed diagnostics; explicit path lists |
| 01, cell 7 | `dataset_audit.rgb_pixel_digest` | Exact dimension/RGB prefix and SHA-256; accepts already decoded RGB bytes |
| 01, cell 9 | `dataset_audit.calculate_similarity` | Original normalized-array MAE/Pearson; constant arrays retain NaN correlation |
| 01, cell 12 | `dataset_audit.compare_at_shift`, `best_translation_match` | Original shift order, overlap threshold 0.85 and strict MAE tie behavior; OpenCV resize import is lazy |
| 02, cell 8 | `similarity_split.similarity_groups` | Original connected components over confirmed-pair endpoints; globals become inputs |
| 02, cell 16 | `similarity_split.select_group_allocation` | Original exhaustive 9/10-group search and stable mean/max ranking; precomputed counts/targets become arguments |
| 03, cell 20 | Existing `workflow.read_metrics` / extraction convention | First maximum logged mAP50–95 selection already represented; no duplicate implementation |

Modules are under [src/trustpcb/rq1](../../src/trustpcb/rq1/); tests under
[tests/rq1](../../tests/rq1/). Tests use synthetic arrays, path lists and counts,
including comparison with selected original pure notebook code. No saved scientific
result is recalculated. NumPy is required for array kernels. OpenCV is needed only
for unequal-size translation diagnostics; that optional branch was not exercised
locally because OpenCV is unavailable in the verification environment.

## Logic deliberately retained in notebooks

- Notebook 01 cells 0–6: environment setup, dataset traversal, image/annotation
  validation and audit-table assembly depend on raw images/labels and notebook
  state. Cell 7's full duplicate grouping and cell 8's pHash/dataset-wide screening
  remain historical code; only the isolated pixel-hash operation was extracted.
- Notebook 01 cells 9–15: candidate orchestration, tier filtering, cached image
  loading, manual-review displays/decisions and CSV writes remain historical.
  Automatic scores must not be presented as reproducing human review decisions.
- Notebook 02: class-count construction and split-manifest writes remain historical.
  In particular cell 16 constructs group counts from pair rows and assumes the
  original group structure. The extracted search does not silently generalize that
  construction. Its combinatorial real-data execution belongs on DICC. Frozen
  manifests are not regenerated. RQ2's partition algorithm is a separate workflow.
- Notebook 03: environment/class checks and the preliminary smoke/supplied/
  similarity-aware YOLO calls remain historical. Shared runtime configuration is
  already in `dataset_config.py`; matched repeated training remains canonical in
  `workflow.py`. Preliminary comparison formatting is preserved in its saved CSV.

These stateful cells were not converted into a new end-to-end pipeline: doing so
would need separate review and external data, and would risk implying unsupported
equivalence to the executed notebooks. Newly extracted helpers are not substituted
into historical provenance or source hashes.

## Reporting, not research execution

`reporting.py` is newly authored presentation code. It copies small existing tables,
selected existing preliminary figures and two existing notebook PNG panels. Three
new figures render saved image counts, preliminary metrics and matched-training
means/sample SD. It does not select epochs, aggregate metrics, calculate intervals,
fit models or execute extracted research functions.

The [RQ1 reporting area](../../outputs/rq1/README.md) records all artifact sources and
exact-byte hashes. The producer of `paired_differences.csv` and `paired_summary.csv`
remains **unresolved**. Their existing bytes are copied, not recreated or attributed
to the new Python code. Raw RQ1 training plots/checkpoints remain external.
