# MedQuAD Medical Q&A Chatbot — Project Report

## Abstract

The MedQuAD Medical Q&A Chatbot is a local web application for retrieving
medical information from the MedQuAD dataset and selected trusted health
webpages. It uses Sentence Transformers to create semantic embeddings and
ChromaDB to persist and search those vectors. A scheduled updater can refresh
configured webpages and replace their indexed passages, allowing later
questions to retrieve updated page content. The interface is built with
Streamlit. The application is an educational retrieval tool and does not
diagnose conditions or generate medical advice.

## Problem statement

A static question-and-answer dataset cannot reflect changes made to external
health information after the dataset was published. The project adds a
controlled update mechanism so an operator can specify trusted webpages and
the chatbot can periodically incorporate their current text into its searchable
knowledge base.

## Objectives

- Load answerable medical question-and-answer pairs from the included MedQuAD
  XML files.
- Retrieve relevant MedQuAD records and trusted-source passages using semantic
  similarity.
- Persist embeddings locally so the knowledge base can be reused after
  application restarts.
- Periodically refresh operator-configured webpages from an explicit allowlist
  of trusted medical domains.
- Display the source and link for retrieved information.
- Preserve clear medical safety and information-source notices.

## Technology stack

| Component | Role |
| --- | --- |
| Python | Data loading, vector retrieval, and update logic |
| MedQuAD | Initial medical question-and-answer dataset |
| Sentence Transformers | Embedding model `all-MiniLM-L6-v2` |
| ChromaDB | Persistent vector collection and semantic search |
| Streamlit | Chatbot search interface and manual update control |
| Schedule | In-process periodic update scheduling |
| XML / HTML parsing | MedQuAD ingestion and trusted-page text extraction |

## System architecture

```text
                           ┌──────────────┐
                           │ MedQuAD XML │
                           └──────┬───────┘
                                  ▼
                           XML preprocessing
                                  ▼
                       Sentence Transformers
                                  ▼
                         ┌─────────────┐
                         │  ChromaDB   │
                         └──────┬──────┘
                                ▲
                                │ new page embeddings
┌────────────────────┐   ┌─────┴──────┐
│ trusted_sources.txt│──►│ Scheduled  │
└────────────────────┘   │  updater   │
                         └─────┬──────┘
                               │ fetch and extract page text
                               ▼
                       Trusted health pages

Question ─► Sentence Transformers ─► ChromaDB retrieval
         ─► source-linked results ─► Streamlit UI
```

## Implementation

### MedQuAD ingestion

The loader parses XML documents under `MedQuAD-master copy`, extracts
question-answer pairs and their source metadata, and ignores entries without an
answer. The knowledge-base initializer creates or opens a persistent
`medical_knowledge` ChromaDB collection and indexes records in batches. Stable
content-derived IDs prevent duplicate IDs for repeated questions with distinct
answers or metadata. Exact duplicate records are indexed once. Stale MedQuAD
IDs left by earlier index formats are removed without deleting trusted-source
records.

### Semantic retrieval

The same Sentence Transformers embedding function is used for indexed content
and user queries. ChromaDB searches the persistent collection using cosine
distance and returns the most relevant MedQuAD answers or trusted-page text
passages. Result metadata provides a source label and original URL. The
application also retains lightweight phrase matching for selected diseases,
symptoms, and treatments.

### Dynamic knowledge updates

The `trusted_sources.txt` configuration accepts one HTTPS URL per line. The
updater validates each hostname against `cdc.gov`, `medlineplus.gov`, `nih.gov`,
and `who.int`, including their subdomains, and checks redirects against the
same allowlist. It fetches HTML/XHTML pages, extracts readable text, divides it
into chunks, and upserts the new chunks into ChromaDB. Older extra chunks for
the same URL are removed. Fetch and parsing errors are logged; a failed refresh
does not replace previously indexed page text.

The Streamlit sidebar supports a manual refresh. Automatic updates run in a
background scheduler thread within the Streamlit process. The default interval
is 24 hours and can be changed using `UPDATE_INTERVAL_HOURS` (minimum 1 hour).
The first scheduled update occurs after that interval. The updater processes
only configured pages; it does not crawl links or discover new URLs.

### User interface and safety

The Streamlit page accepts a natural-language health question, displays
retrieved information and source links, and shows the indexed item count and
configured update status. It warns users that the application is educational
and not a substitute for professional healthcare.

## Running the project

From the project directory, using Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The included `MedQuAD-master copy` dataset must remain beside `app.py`. The
vector index is persisted in `chroma_db/`. To change the refresh period:

```bash
UPDATE_INTERVAL_HOURS=12 streamlit run app.py
```

Configure trusted pages in `trusted_sources.txt`; restart the app after editing
that file. Use the sidebar control for an immediate refresh.

## Testing and verification

Run tests with:

```bash
python -m unittest discover -s tests -v
```

The test suite covers:

- XML parsing, answer filtering, retrieval, and entity recognition.
- Duplicate-safe MedQuAD indexing and removal of stale MedQuAD IDs.
- Persistence-facing source chunk replacement and vector result metadata.
- Trusted URL allowlist validation and URL de-duplication.
- Successful and failed trusted-source refresh outcomes.

During implementation verification, all 10 unit tests passed. The integrated
application initialized against its local ChromaDB collection, displayed
16,377 indexed knowledge chunks, and returned a relevant result for an asthma
symptoms query.

## Limitations

- Only explicitly configured HTML/XHTML pages on the supported domains are
  fetched; pages are not crawled or automatically discovered.
- A successful update requires network access and readable page text.
- The built-in scheduler runs in the app process. Multiple Streamlit replicas
  can execute duplicate schedules; a dedicated external scheduler is
  recommended for multi-instance deployments.
- The model and vector database are local application dependencies; first-time
  model download and dataset indexing can take time.
- Retrieved source content may be incomplete, outdated, or unsuitable for an
  individual's situation. The tool does not validate clinical accuracy.

## Future enhancements

- Add configurable ingestion support for additional trusted formats and
  official APIs.
- Add monitoring and a visible timestamp/status for the last successful
  refresh.
- Use an external job scheduler for multi-instance deployments.
- Add source review and content-quality checks before indexing.
- Expand automated integration testing against a temporary real ChromaDB
  collection.

## Medical safety and dataset attribution

This project is an educational information-retrieval system, not a diagnostic
or treatment service. Users should check cited sources and consult a qualified
healthcare professional before making health decisions. For emergencies,
contact local emergency services.

MedQuAD is distributed under the
[Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
Please cite “A Question-Entailment Approach to Question Answering” by Asma Ben
Abacha and Dina Demner-Fushman (BMC Bioinformatics, 2019), as requested by the
dataset authors.
