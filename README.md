# SEO Summarisation API

This project exposes a small FastAPI service that leverages **Azure OpenAI** to generate concise SEO summaries for a set of web‐page inputs.

## 🛠️ Requirements

```bash
python 3.9+
```

All Python dependencies are listed in `requirements.txt`.

## 🔐 Required Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Your Azure OpenAI resource endpoint.<br/>Format: `https://<resource-name>.openai.azure.com` |
| `AZURE_OPENAI_KEY` | Primary/secondary key for the Azure OpenAI resource. |
| `AZURE_OPENAI_DEPLOYMENT` | Name of the deployed ChatGPT (GPT-3.5/4) model inside Azure. |
| `AZURE_OPENAI_API_VERSION` | API version to target _(default: `2023-07-01-preview`)_ |
| `OPENAI_TEMPERATURE` | Generation temperature _(default: `0.3`)_ |
| `OPENAI_MAX_TOKENS` | Max tokens for each summary _(default: `256`)_ |

You can place these in a `.env` file during local development.

```dotenv
AZURE_OPENAI_ENDPOINT="https://my-resource.openai.azure.com"
AZURE_OPENAI_KEY="<secret>"
AZURE_OPENAI_DEPLOYMENT="gpt-35-turbo"
```

## 🚀 Running Locally

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Export environment variables** or simply create a local `.env` file — the
   application automatically loads it via `python-dotenv` on startup.

3. **Start the server**:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. Navigate to `http://localhost:8000/docs` for the interactive Swagger UI.

## ➡️ Example Requests

### SEO Page Summaries

```json
POST /summaries
{
  "items": [
    {
      "url": "https://example.com",
      "title": "Amazing Product – Example.com",
      "meta_description": "Buy amazing products at Example.com",
      "content": "<h1>Amazing Product</h1> ..."
    }
  ],
  "language": "English"
}
```

### Project Overview Summaries

```json
POST /overview-summaries
{
  "projects": [
    {
      "url": "https://example.com",
      "domain_authority": 45,
      "domain_authority_change": 2,
      "no_of_kw": 150,
      "avg_pos_change": -0.5,
      "avg_pos_desktop": 15.2,
      "avg_pos_mobile": 14.8,
      "increased_keywords": 35,
      "decreased_keywords": 20,
      "range_distribution": {
        "1-3": 15,
        "4-10": 45,
        "11-20": 60,
        "21-50": 30
      },
      "top_keywords": [
        {
          "id": 1,
          "keyword": "example product",
          "position": 3,
          "position_change": 2,
          "volume": 1200,
          "cpc": 1.5,
          "kd": 35,
          "intent": "Transactional"
        }
      ]
    }
  ],
  "language": "English"
}
```

### Keyword Summaries

```json
POST /keyword-summaries
{
  "keywords": [
    {
      "id": 1846,
      "keyword": "body wash",
      "volume": 0,
      "cpc": 0,
      "kd": 0,
      "intent": "Transactional",
      "cur_desk": 101,
      "prev_desk": 101,
      "cur_mobile": 101,
      "prev_mobile": 101,
      "project_id": 25,
      "owner_id": 1,
      "track_freq": "weekly",
      "location": "India",
      "last_updated": "2025-06-10T16:57:36.768346+00:00",
      "no_of_tracks": 2,
      "advanced": false,
      "ranking_page_for_kw_desktop": "",
      "ranking_page_for_kw_mobile": ""
    }
  ],
  "language": "English"
}
```

### Historical Rank Summaries

```json
POST /historical-rank-summaries
{
  "project_id": 25,
  "page": 1,
  "limit": 10,
  "total_keywords": 86,
  "keywords": [
    {
      "id": 1846,
      "keyword": "body wash",
      "intents": "Transactional",
      "volume": 0,
      "position_desktop_change": 0,
      "position_mobile_change": 0,
      "rank_history": [
        {
          "mobile": [
            {
              "url": "https://www.funworldblr.com/tickets",
              "Rank": 1,
              "Type": "My Site",
              "Domain": "funworldblr.com",
              "Keyword": "Fun World Bangalore tickets"
            },
            {
              "url": "https://www.funworldblr.com/offers",
              "Rank": 2,
              "Type": "My Site",
              "Domain": "funworldblr.com",
              "Keyword": "Fun World Bangalore tickets"
            }
          ],
          "desktop": [
            {
              "url": "https://www.funworldblr.com/tickets",
              "Rank": 1,
              "Type": "My Site",
              "Domain": "funworldblr.com",
              "Keyword": "Fun World Bangalore tickets"
            },
            {
              "url": "https://www.funworldblr.com/offers",
              "Rank": 2,
              "Type": "My Site",
              "Domain": "funworldblr.com",
              "Keyword": "Fun World Bangalore tickets"
            }
          ]
        }
      ]
    }
  ],
  "language": "English"
}
```

The response will contain a `summaries` array with one summary per input item.

---

Feel free to tailor the prompt or parameters to your exact SEO requirements. 