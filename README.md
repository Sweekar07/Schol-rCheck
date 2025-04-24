# ScholárCheck Fact-Check Pipeline

A Flask API that implements a core fact-checking pipeline by retrieving academic research from Semantic Scholar and analyzing claims using Gemini.

Name: Sweekar Bangera
Email: sweekar.786b@gmail.com

## Features

- Accepts claims for fact-checking via a simple REST API
- Retrieves relevant research papers from Semantic Scholar API
- Analyzes claims against research evidence using Gemini
- Returns structured assessment with explanation and references

## Setup Instructions

### Prerequisites

- Python 3.8+
- An Gemini API key
- Semantic scholar API key (optional)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd clone-directory
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running the Application

For development:
```
python app.py
```

For production:
```
gunicorn app:app
```

## API Documentation

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy"
}
```

### Fact Check

**Endpoint:** `POST /api/fact-check`

**Request Body:**
```json
{
  "claim": "Intermittent fasting improves brain function"
}
```

**Response:**
```json
{
  "assessment": "Supported",
  "explanation": "Multiple studies in the provided abstracts indicate positive effects of intermittent fasting on cognitive function and neuroplasticity. Research shows IF increases BDNF levels and improves memory performance in both animal and human studies.",
  "references": [
    {
      "title": "Intermittent fasting enhances cognitive functions and brain structure through BDNF signaling in mice",
      "url": "https://www.semanticscholar.org/paper/abc123"
    },
    {
      "title": "Impact of intermittent fasting on human cognitive function: a systematic review",
      "url": "https://www.semanticscholar.org/paper/def456"
    },
    {
      "title": "Effects of caloric restriction on brain function and cognitive performance",
      "url": "https://www.semanticscholar.org/paper/ghi789"
    }
  ]
}
```

## Test Cases

### Test Case 1: Supported Claim

**Claim:** "Intermittent fasting improves brain function"

**Expected Result:** Assessment will likely be "Supported" as there is substantial research showing benefits of intermittent fasting on cognitive function.

### Test Case 2: Refuted Claim

**Claim:** "Vitamin C megadoses cure cancer"

**Expected Result:** Assessment will likely be "Refuted" as research does not support vitamin C as a cancer cure.

### Test Case 3: Insufficient Evidence

**Claim:** "Listening to Mozart increases IQ permanently"

**Expected Result:** Assessment will likely be "Lacks Sufficient Evidence" as research on the Mozart effect is mixed and limited.

## Error Handling

The API returns appropriate error messages with corresponding HTTP status codes:

- 400: Bad Request (missing claim)
- 500: Internal Server Error (processing errors)

## Limitations

- The fact-checking quality depends on the available research in Semantic Scholar
- Analysis is limited to the top 5 papers retrieved
- Assessment depends on GEMINI interpretation capabilities