# AI-Driven Compliance Checking for Safety Procedures

Audit safety procedure documents against OSHA/ANSI/CSA standards using LLM-powered analysis and RAG retrieval.

## Setup

```bash
# 1. Create virtual environment (requires Python 3.12+)
python3.12 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
echo 'GOOGLE_API_KEY=your-key-here' > .env
```

## Usage

**Run the full pipeline** (ingest regulations → audit procedure → generate reports):

```bash
python main.py pipeline "data/procedures/HSP 331.pdf"
```

Change this PDF with any PDF you choose

Reports are saved to `reports/` as JSON, HTML, and Markdown. Use MDviewer to view the markdown

**Fetch a new OSHA regulation from the eCFR API:**

```bash
python main.py fetch 1910.147
```

**Ingest regulations only, if you want to add more regulations data/regulations** (without running an audit):

```bash
python main.py ingest
```

## Project Structure

```
data/procedures/    ← Safety procedure documents (PDF, TXT, DOCX)
data/regulations/   ← Regulation texts (TXT, PDF, DOCX, HTML)
src/                ← Core modules (loader, mapper, store, auditor, reporter)
reports/            ← Generated compliance reports
config.py           ← Settings (model, chunk size, API keys via .env)
main.py             ← CLI entry point
```
