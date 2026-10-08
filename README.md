# MedQuAD Medical Q&A Chatbot

A local, retrieval-based medical question-answering app powered by the included
[MedQuAD dataset](https://github.com/abachaa/MedQuAD). It uses TF-IDF similarity
to retrieve relevant dataset answers and lightweight phrase matching to identify
dataset conditions plus common symptoms and treatments.

## Run locally

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Keep the `MedQuAD-master copy` dataset folder beside `app.py`. The app indexes
answerable question-answer pairs on first load; pairs with blank answers are
excluded. The dataset can contain answers that were removed by its authors, so
an empty or incomplete result is not evidence that no medical information exists.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Medical safety and attribution

This is an educational information-retrieval demo, not a diagnostic or treatment
tool. Answers are quoted from retrieved dataset records; consult a qualified
healthcare professional for personal medical advice.

MedQuAD is published under the [Creative Commons Attribution 4.0 International
License](https://creativecommons.org/licenses/by/4.0/). If you use the dataset,
please cite “A Question-Entailment Approach to Question Answering” by Asma Ben
Abacha and Dina Demner-Fushman (BMC Bioinformatics, 2019), as requested in the
dataset README.
