# RelTwin — ICASSP 2027 author-review manuscript

Main file: `main.tex`. Compiler: **pdfLaTeX**. Bibliography: **BibTeX** (`IEEEbib`). Upload the ZIP contents directly to an Overleaf project; `main.tex`, `spconf.sty`, and `authors.tex` must be at the project root.

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

This is an author-review deliverable, not an automatic submission. It preserves the supplied ICASSP template files. Technical content occupies no more than four pages; a fifth page, when present, contains references only. Source tables use at least 9 pt body text.

## Paper focus and evidence roles

The paper studies query-specific timestamp answers when two inverse relations coexist in one recording. RelTwin-Cand is the existing `no_exchange` configuration: sequence supervision plus ordinary candidate cross-entropy and rehearsal. JS/RBEE and SetPO are extension controls, not independently proven central advances.

The matched-runtime SFT--Cand seed 0 comparison is separate from historical-checkpoint re-evaluation and from the archived full system. Clotho's archived score is official plus boundary refinement, not RelTwin transfer. UnAV is not included in fair ranking because its protocol remains qualified. All relational development and timing-stress results are labeled according to their actual data role.

The `notes/` directory contains Chinese and English Markdown exports, numerical evidence, source/protocol provenance, and the author checklist. These files do not enter the PDF and need not be uploaded separately as conference supplementary material. They are provided for review and reproducibility, not to evade the page limit.

## Authors

Zhicheng Tang — u202414252@hust.edu.cn

Yuehan Zhang — yuehanzhang2005@hotmail.com

Both: Huazhong University of Science and Technology, Wuhan, China. The author order is supplied by the user. No corresponding author or ORCID is invented. Authors must confirm those portal fields before submission.

## Submission checks

The [ICASSP 2027 Paper Kit](https://cmsworkshops.com/ICASSP2027/papers/paper_kit.php), checked 2026-09-13, lists the regular-paper deadline as 2026-09-16. Each author requires an ORCID; the PDF and submission form must use the same author list. Review is not double-blind unless explicitly specified otherwise. This manuscript includes the supplied names and a disclosure of AI-assisted writing, schematic preparation, and analysis code.

Authors must review all scientific claims and attribution, confirm affiliations/contact details, approve the final AI disclosure, and run the submission portal's document check. Compilation and numerical validation do not guarantee paper acceptance.
