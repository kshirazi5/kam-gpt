# Kam-GPT

An AI-powered portfolio that represents **Kamran Shirazi** for recruiters and
engineers. The core logic lives in `kam_gpt.generator` and powers a simple
Streamlit application you can share publicly.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Open the printed URL in your browser to ask portfolio questions.

## Shareable deployment

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and select
   **New app**.
3. Point Streamlit to your GitHub repository and choose `app/app.py` as the app
   entry point.
4. Click **Deploy** to publish a URL you can share on LinkedIn or elsewhere.

Streamlit will automatically install the dependencies listed in
`requirements.txt` and keep the app updated every time you push changes.
