# Kam-GPT

A conversational, AI-powered portfolio that represents **Kamran Shirazi** for recruiters and engineers.

## Preview

The Streamlit application exposes a chat interface so visitors can ask Kam-GPT about Kamran's experience, focus areas, and how to get in touch.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."  # Optional, enables GPT-4.1 responses
streamlit run app/app.py
```

## Project Structure

```
├── app
│   └── app.py        # Streamlit chat experience
├── requirements.txt  # Python dependencies
└── README.md
```

## Development Notes

- Provide an `OPENAI_API_KEY` environment variable to let Kam-GPT answer with OpenAI's
  GPT-4.1 model. Without the key the app falls back to the built-in rule-based
  knowledge base in `app/app.py`.
- Update the knowledge entries in `app/app.py` to reflect the latest achievements or
  contact details when relying on the built-in experience.
