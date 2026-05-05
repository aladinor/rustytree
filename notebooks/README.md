# notebooks

Short demos of `rustytree` against real public datasets.

| Notebook | What it shows | Data |
|---|---|---|
| `klot_demo.ipynb` | Open the same NEXRAD radar DataTree with `engine="zarr"` and `engine="rustytree"` side-by-side; compare wall time + verify structural / `.sel()` parity. | `s3://nexrad-arco/KLOT` (anonymous icechunk on S3, 107 groups) |

## Running

```bash
# In the repo root, with maturin develop already done:
.venv/bin/pip install jupyter
.venv/bin/jupyter lab notebooks/
```

The S3 reads are anonymous so no AWS credentials are needed; the
notebook hard-codes `storage_options={"region": "us-east-1", "anon": True}`.

Cold-cache wall times are sensitive to your network and AWS S3 prefix
performance. The numbers in the notebook are from a single run on a
home connection; expect ~30-60 s for `engine="zarr"` and ~2-5 s for
`engine="rustytree"`.
