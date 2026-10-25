# Paper and release file index

Submitted paper: [PDF](https://github.com/tzctzctzctzc/RELTWIN/blob/652c5cf420b1c0d5ab63d1620ab826a466bc00b6/paper.pdf), confirmed by the authors on 2026-10-24 as `finalpaper.pdf`.

SHA256: `942e5c14ba195325d9634bd9c89eedb155e00209cd35f70d1caf5919b262ea71`.

| Content | Location |
|---|---|
| Submitted table values and source hashes | `experiments/paper/release.json` |
| Full two-stage training entry for both backbones | `reproduction/train_paper.sh` |
| TimeAudio paper model and inference configuration | `configs/timeaudio_paper.yaml` |
| Distributed checkpoint configurations | `configs/release_profiles.json` |
| Checkpoint sizes and hashes | `checkpoints/manifest.json` |
| Archived SpotSound-A public predictions | `results/paper_spotsound/{spotsound400,clotho6649,unav100}` |
| SpotSound-A Table 2 predictions | `results/paper_spotsound/table2_{official,sft256,candidate256}` |
| TimeAudio Table 1 predictions | `results/paper_timeaudio/{spotsound400,clotho1000,unav100}` |
| TimeAudio Table 2 predictions | `results/paper_timeaudio/table2_{official,sft,reltwin}` |
| Shared 300-pair manifest | `results/paper_timeaudio/table2_manifest.json` |
| CPU-rescored tables | `results/paper_alignment_verification.json` |
| Training commands and scoring profiles | `docs/PAPER_REPRODUCTION.md` |

The TimeAudio directory named `clotho1000` contains the completed 6,649-query evaluation. Source-to-release paths and hashes are in each result directory's `provenance.json`; original summaries and predictions are retained verbatim apart from documented text-newline normalization.

`experiments/paper/release.json` defines the submitted tables, training configuration, and shared evaluation protocol. `configs/release_profiles.json` indexes downloadable artifacts.

`check_paper_release.py` validates the paper catalog. `verify_paper_evidence.py` checks 52 source files, complete index coverage, paired query/target alignment, and stored metrics. `verify_release_checkpoints.py` checks the distributed binaries against their checksum manifest.
