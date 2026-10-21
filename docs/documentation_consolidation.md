# Research journals and documentation archives

The root README is the repository homepage. Three editable research journals replace
the earlier README-derived Word guides:

- [RQ1 Research Journal](rq1/RQ1_Research_Journal.docx): dataset auditing, similarity-aware splitting, preliminary baselines and matched repeated training.
- [RQ2 Research Journal](rq2/RQ2_Research_Journal.docx): partitioning, detector and calibration preparation, transformation signals, ablations, weighted fusion and final evaluations.
- [RQ3 Research Journal](rq3/RQ3_Research_Journal.docx): selective review, its reporting extension, computational efficiency and frozen-threshold final evaluation.

These journals synthesize saved methods, configurations, original numerical records,
provenance and existing figures. They are later evidence reconstructions, not new
experiments or proof that reconstructed Python was historically executed. Missing
external/DICC inputs and mixed or negative results are explicit in each journal.
Existing Markdown protocols remain technical references.

## Evidence and source identity

[research_journal_evidence.json](research_journal_evidence.json) records journal table
selections, cited source paths, source hashes, figure paths and the wider inspected
file inventory. Inventory counts describe file inspection, not independent experiments
or independent replications; reporting copies may duplicate original evidence.

[documentation_inventory.json](documentation_inventory.json) retains all 17 retired
README identities and maps them to journal navigation bookmarks. Original README
sources remain recoverable from Git commit
`6ffecf2309519b92cbbb54293cc3aa63fe9fc868`. Original RQ1 notebook identities and
checkout transformations remain in [rq1/notebook_archive.json](rq1/notebook_archive.json).
Paths labelled historical in a journal are not active filesystem links.

## Historical hashes and current documentation

The original reporting manifests remain unchanged: **29 / 64 / 44** artifact
records for RQ1/RQ2/RQ3. Their retired README entries validate through Git. Current
local reporting artifact counts remain **25 / 57 / 39**, plus each manifest.
No scientific tables, figures, original provenance or implementations were changed
by journal reconstruction.

[documentation_archive.json](documentation_archive.json) preserves original source
paths, SHA-256 values, immutable commits, blob identities and checkout newline
transforms. Its separate current-file hashes now bind the journals and navigation.
Current documentation is not substituted for original historical evidence.
Scientific inputs, reporting copies, figures and original-file references still
require their exact local bytes. A missing Git history or changed scientific file
remains a validation failure.

Presentation builders retain archived README records without recreating those
files. Keep the original reporting manifests; deleting them is not a supported
clean rebuild. The journal evidence inventory is documentation metadata, not a
replacement scientific manifest.

## Verification

Using the existing CPU test environment and `PYTHONPATH=src`:

```text
python -B -m trustpcb.rq1.reporting --verify-only
python -B -m trustpcb.rq2.reporting --verify-only
python -B -m trustpcb.rq3.reporting --verify-only
python -B -m unittest discover -s tests
```

The journals use native editable Word tables, heading styles, numbered figures,
evidence registers and page numbers. Every rendered page was visually inspected
using background Microsoft Word PDF export and page images outside the repository.
No PDF duplicates or generated scientific results are added to the repository.
