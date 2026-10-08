# MedQuAD Medical Q&A Chatbot

A local Streamlit app for semantic search across the MedQuAD medical
question-and-answer dataset and a configurable list of trusted health webpages.
The app stores embeddings in ChromaDB and can refresh configured pages
periodically so newly published page content can appear in later search results.

This is an information-retrieval demo, not a diagnostic or treatment tool. It
retrieves source material; it does not generate or independently verify medical
advice.

## Technology

- Python 3.10+
- Streamlit for the user interface
- Sentence Transformers (`all-MiniLM-L6-v2`) for text embeddings
- ChromaDB for persistent vector storage and semantic search
- `schedule` for periodic trusted-source updates
- MedQuAD XML dataset for the initial question-and-answer collection

## How it works

```text
MedQuAD XML ─► XML parsing ─► Sentence Transformers ─► persistent ChromaDB
Trusted URLs ─► scheduled updater ─► HTML text extraction ─► embeddings ─► ChromaDB
Question ─► vector retrieval ─► source-linked results ─► Streamlit
```

MedQuAD XML files are parsed into answerable question-and-answer records;
records without answers are skipped. The app embeds these records and keeps
them in `chroma_db/`. When a user asks a question, the same embedding model
searches the collection for relevant records or trusted-source text passages.
Results include the original source link when available.

The existing lightweight phrase matcher also recognizes selected dataset
conditions and common symptom and treatment terms.

## Run locally

From the project directory, use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Keep the included `MedQuAD-master copy` folder beside `app.py`. On first
startup, the app creates `chroma_db/`, indexes MedQuAD, and may download the
Sentence Transformers model. Initial indexing can take time; subsequent
launches reuse the persistent collection and add records that are not yet
indexed.

## Configure trusted-source updates

Add one HTTPS webpage URL per line to [`trusted_sources.txt`](./trusted_sources.txt).
Only these official health domains and their subdomains are accepted:

- `cdc.gov`
- `medlineplus.gov`
- `nih.gov`
- `who.int`

The included configuration lists a MedlinePlus page. Remove or replace it if
you prefer another supported page. The updater fetches the configured pages
only; it does not crawl links or discover additional pages. Only HTML and
XHTML pages with enough readable text are indexed.

The background scheduler refreshes the configured pages every 24 hours by
default. To change the interval, set `UPDATE_INTERVAL_HOURS` to an integer of at
least 1 before starting Streamlit. For example:

```bash
UPDATE_INTERVAL_HOURS=12 streamlit run app.py
```

The scheduler's first automatic refresh occurs after the configured interval.
Use **Update trusted sources now** in the sidebar to refresh immediately.
When page text changes, its indexed passages are replaced; stale passages for
that page are removed. A fetch or parsing failure is logged and reported; the
last successfully indexed content remains available.

The scheduler runs inside the Streamlit process. Use a single app process for
this built-in scheduler; multiple app replicas can run duplicate schedules.
For multi-replica deployments, use a dedicated scheduler or external job
runner. Restart the app after editing `trusted_sources.txt`.

## Data and storage

- `chroma_db/` contains the local persistent vector database and is ignored by
  Git.
- MedQuAD data remains in the included `MedQuAD-master copy` folder.
- The embedding model is provided by Sentence Transformers and may be
  downloaded on first use.
- Do not delete `chroma_db/` unless you intend to rebuild the vector index.

## Tests

Run the unit tests from the project directory:

```bash
python -m unittest discover -s tests -v
```

The tests cover MedQuAD parsing and retrieval, duplicate-safe vector indexing,
stale MedQuAD ID cleanup, trusted URL validation, and source refresh behavior.

## Safety and attribution

The chatbot is for general educational information only. Retrieved content can
be incomplete or out of date and is not independently validated. Check the
original source and consult a qualified healthcare professional for personal
medical decisions. For emergencies, contact local emergency services.

MedQuAD is published under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
If you use the dataset, cite “A Question-Entailment Approach to Question
Answering” by Asma Ben Abacha and Dina Demner-Fushman (BMC Bioinformatics,
2019), as requested in the dataset README.
