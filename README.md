# Amazon Review Sentiment Analyzer

Simple sentiment classification for Amazon reviews using a Bidirectional LSTM model in PyTorch, served with FastAPI and a lightweight frontend.

## Project structure

```
amazon_sentiment_analysis/
├── checkpoints/
│   ├── best_lstm.pt
│   └── history_lstm.json
├── data/
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       ├── test.csv
│       └── vocab.json
├── frontend/
│   └── index.html
├── src/
│   ├── api/
│   │   └── app.py
│   ├── models/
│   │   └── sentiment_model.py
│   ├── utils/
│   │   ├── amazon_loader.py
│   │   └── dataset.py
│   ├── train.py
│   └── inference.py
├── README.md
└── requirements.txt
```


## Quick start

1. create and activate venv
   - `python -m venv venv`
   - `source venv/bin/activate`

2. install dependencies
   - `pip install -r requirements.txt`

3. prepare data
   - `python -m src.utils.amazon_loader --samples 60000`

4. train model
   - `python -m src.train`

5. run API
   - `uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload`

6. open UI
   - open `frontend/index.html` in browser

## API endpoints

- `POST /api/predict` (single text)
- `POST /api/predict/batch` (list of texts)
- `GET /api/health`

Example:

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is great"}'
```

## Notes

- Use Python 3.11+.
- GPU speeds up training.
- Checkpoints are stored in `checkpoints/best_lstm.pt`.

