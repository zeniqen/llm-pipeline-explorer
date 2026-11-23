# LLM Pipeline Explorer

An interactive classroom visualizer for the language model processing pipeline. Built with vanilla HTML5, CSS3, and JavaScript without external libraries or dependencies.

## The pipeline

The app demonstrates six stages of text processing in an NLP system:

1. **Corpus Text** — Raw text used to build the model's vocabulary.
2. **Vocabulary** — Unique word and punctuation lookup table. ID `0` is reserved for `[UNK]`. Corpus items receive IDs starting at `1`.
3. **Sentence Tokenization** — Splitting incoming text into lexical tokens.
4. **Token IDs** — Mapping known tokens to vocabulary IDs. Unknown words fall back to a subword character rule (`c`, `##h`, `##a`, `##i`, `##r`) assigned ID `0`.
5. **Sentence Embeddings** — Projecting sentences into numerical vector space.
6. **3D Visualization** — Interactive 3D scatter plot rendered on HTML5 canvas with metric multidimensional scaling (MDS).

## Running the app

Start a local HTTP server from this directory:

```bash
python3 -m http.server 8000
```

Open `http://localhost:8000` in any modern web browser. It doesn't need Node, bundlers, or internet access.

You can also open `index.html` directly in your browser.

## Embedding modes

The 3D visualization offers two contrasting projection modes:

- **Spelling-based (static):** Uses character bigram Jaccard distance and normalized Levenshtein string edit distance. Sentences with nearly identical letters (like *"The cat sat on the..."* and *"The bat sat on the..."*) group together, even though their meanings differ.
- **Meaning-based (contextual):** Uses topic concept vectors and cosine distance. Sentences with similar meaning (like *"The cat sat on the mat"* and *"Kitten rests on the rug"*) cluster together, despite sharing zero words.

## Architecture

- `index.html` — Semantic structure, accessible landmarks, and concept notes.
- `style.css` — High-contrast light theme, accessible focus states, and responsive layout.
- `app.js` — Tokenizer logic, vocabulary manager, metric MDS projection, and 3D canvas renderer with drag rotation and zoom.
