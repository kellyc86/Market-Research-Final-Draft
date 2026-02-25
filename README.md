# Market Research Assistant

AI-powered industry research tool that generates analyst-grade reports from Wikipedia sources.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Enter your Google AI API key in the sidebar (get one free at [Google AI Studio](https://aistudio.google.com/apikey))

## Architecture

- **Step 1**: Industry input with LLM-based validation
- **Step 2**: Wikipedia retrieval (broad fetch + LLM-ranked filtering to top 5)
- **Step 3**: Structured industry report generation (<500 words)

## Tech Stack

- Streamlit (UI)
- LangChain + Google Gemini Flash (LLM)
- WikipediaRetriever (data retrieval)
