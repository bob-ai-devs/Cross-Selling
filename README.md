# AI Cross-Selling Recommendation Engine (Streamlit)

Streamlit port of the Flask "Cross-Selling" module, extended with an analytics dashboard.

| Page | What it does |
|---|---|
| Home | Model/secret status, how the engine works |
| Single Customer | Form → top-3 products with affinity, confidence and a model-interest chart; sample profiles; CSV/JSON download |
| Batch Prediction | CSV/Excel upload (or demo data), auto column mapping, data-quality profile, pre-trained **or** custom model, filters, CSV/Excel download |
| Train Custom Model | Dynamic product/date column pairs, ID-column exclusion, live loss chart, metrics, confusion matrix, cluster/elbow view, report downloads |
| Insights Dashboard | Product mix, affinity distribution, segment heat-maps, held-vs-recommended gap, co-holding lift, Customer 360, campaign lists, Excel pack |
| Templates & Demo Data | Schema, starter CSVs, synthetic data generator |

**Nothing is stored.** Uploaded files, predictions and trained models exist only in the user's browser
session (Streamlit session state). There is no write to disk, the Gist or any database — only view and download.

## Project layout

```
streamlit_app.py            entry point (navigation)
app_pages/                  one module per page
core/                       config, storage (Gist loader), preprocessing, inference, training, analytics, demo
tools/publish_artifacts_to_gist.py   run locally to upload the model/encoders to a Gist
models/ , encodes/          optional: bundle artifacts in the repo instead of (or as fallback to) the Gist
.streamlit/config.toml      theme; secrets.toml.example shows the secret format
```

## 1 · Put the pre-trained artifacts in a Gist

Files expected (same names as the Flask app): `cross_selling.keras`, and `scaler.pkl`, `le_gen.pkl`, `le_mar.pkl`,
`le_emp.pkl`, `le_inc.pkl`, `le_prop.pkl`, `le_job.pkl`, `le_com.pkl`, `le_pro.pkl`, `aff_sc.pkl`.

```bash
mkdir -p models encodes
cp <your>/cross_selling.keras models/
cp <your>/*.pkl encodes/
export GITHUB_TOKEN=ghp_xxx            # classic token, "gist" scope
python tools/publish_artifacts_to_gist.py     # prints the gist id
```

Gists only hold text, so each file is stored base64-encoded as `<name>.b64`. To rotate later, run the tool with
`--gist-id <ID>` to update, or create a new Gist and change the secret.

## 2 · Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (do **not** commit `.streamlit/secrets.toml`; do not commit the `.pkl`/`.keras`
   files if you rely on the Gist).
2. share.streamlit.io → *New app* → pick the repo → main file `streamlit_app.py` (Python 3.12).
3. *Settings → Secrets*:

```toml
[github]
token   = "ghp_your_temporary_token"
gist_id = "your_gist_id"
```

Flat keys `GITHUB_TOKEN` / `GIST_ID` are accepted too. When you change the secrets later, press **Reload** in the
app sidebar (or reboot the app) to fetch the new artifacts.

Load order: Gist → repo files (`models/`, `encodes/`) → clear error message.

## Compatibility notes

* Use the **same TensorFlow/Keras and scikit-learn versions** you trained with (the `.keras` file and the pickles are
  version-sensitive). Pin them in `requirements.txt`; `tensorflow-cpu` is used to keep memory low.
* Pickles can run code when loaded — only load artifacts from a Gist you control.
* The scaler is fed columns in the scaler's own `feature_names_in_` order when available; otherwise in the order
  `Age, Gender, …, Years_of_Employment, 1st_product … 5th_product`.

## Running locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
