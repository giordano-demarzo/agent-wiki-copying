# The data

This directory should hold the data release of the OpenAI agent wiki incident,
published on 4 September 2026 at [collusion.wiki](https://collusion.wiki) by
S. Von Arx, C. Slade Byrd, S. Kitts and T. Larsen. We do not redistribute it.

Download it and place these files here:

| file | what it holds |
| --- | --- |
| `revisions.jsonl` | every stored revision: full page text after the edit, the diff hunks, the username, the time of the save |
| `pages.jsonl` | page metadata, including the page family and the deletion counts |
| `labels.jsonl` | one record per username, including the flag marking human accounts |
| `events.jsonl` | the server-side request log |

The release also ships `manifest.json` and `SHA256SUMS`. After downloading, check
the files against the release's own checksums:

```bash
cd data && sha256sum -c SHA256SUMS
```

Then build the parsed dataset:

```bash
python src/build_dataset.py
```

which writes `cache/data.pkl` and prints the totals. On the release we used,
those are 14,591 revisions of 4,579 pages, of which 13,661 edits are under the
3,099 non-human handles.
