# progscore — Web demo (Streamlit)

A browser interface for `progscore`: single-sample programme scoring and
deconvolution for bulk omics, including the cross-modal programme similarity
score (CMPS).

## Run locally

```bash
cd progscore_web
pip install -r requirements.txt
streamlit run app.py
# open http://localhost:8501
```

## Deploy to your own server

```bash
# on the server
cd progscore_web
pip install -r requirements.txt
nohup streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
# then open http://<server-ip>:8501
```

Behind a reverse proxy (nginx):

```nginx
location /progscore/ {
    proxy_pass http://127.0.0.1:8501/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

(Streamlit requires websocket support, hence the Upgrade headers.)

## Features

- Upload a bulk matrix (TSV/CSV, genes × samples).
- Paste signature genes (or use the bundled example).
- Score with mean-z / ssGSEA / GSVA / PLAGE / singscore / **CMPS**.
- Deconvolve with NNLS / MuSiC / CIBERSORT / BayesPrism.
- Download results as CSV.

## Files

- `app.py` — the Streamlit app (self-contained; imports `progscore` from this folder).
- `progscore/` — the package (copy of the GitHub repo's `progscore/`).
- `example/` — example bulk matrix + signature.
