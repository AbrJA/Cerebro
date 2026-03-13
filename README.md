<div align="center">

# 🧠 Cerebro

**A blazing-fast, fully local RAG application built in Julia**

[![Julia](https://img.shields.io/badge/Julia-1.11+-9558B2?logo=julia&logoColor=white)](https://julialang.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-000000?logo=ollama)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

*Ask questions about your documents using local AI — no cloud, no API keys, no data leaves your machine.*

</div>

---

## ✨ Features

- **🔒 100% Local & Private** — All models run via [Ollama](https://ollama.com). Your documents never leave your machine.
- **⚡ Binary Quantized Search** — Embeddings are compressed to binary vectors. Search uses SIMD `popcount` hamming distance — orders of magnitude faster than cosine similarity.
- **🧩 Hierarchical Chunking** — Documents are split into large "parent" chunks (for context) and small "child" chunks (for precise retrieval). Searches match on small chunks, but the LLM receives the full parent context.
- **🔀 Multithreaded Retrieval** — The search engine splits the vector database across all available CPU threads using a zero-allocation MaxHeap for top-k selection.
- **💬 Web Chat Interface** — A clean, modern chat UI powered by [HTTP.jl](https://github.com/JuliaWeb/HTTP.jl), complete with typing indicators and source citations.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP.jl Web UI                        │
│              (Chat Interface @ localhost)                 │
└──────────────────────┬──────────────────────────────────┘
                       │ POST /api/chat
┌──────────────────────▼──────────────────────────────────┐
│                   Backend.jl                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Embed Query  │→│ Hamming KNN  │→│ Prompt Builder │  │
│  │ (Ollama)     │  │ (Parallel)   │  │ + Generation   │  │
│  └─────────────┘  └──────────────┘  └────────────────┘  │
└──────────────────────▲──────────────────────────────────┘
                       │ BinaryIndex
┌──────────────────────┴──────────────────────────────────┐
│                  Ingestion.jl                            │
│  Load Docs → Hierarchical Chunk → Embed → Quantize      │
│                                    (Ollama)   (Binary)   │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Julia 1.11+** — [Install Julia](https://julialang.org/downloads/)
2. **Ollama** — [Install Ollama](https://ollama.com/download)
3. Pull the required models:
   ```bash
   ollama pull gemma3:1b               # chat model
   ollama pull embeddinggemma:300m-qat-q4_0  # embedding model
   ```

### Installation

```bash
git clone https://github.com/AbrJA/Cerebro.git
cd Cerebro
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Usage

1. **Add your documents** — Place `.txt` or `.md` files into the `data/documents/` folder.

2. **Start the server** — Launch Julia with multiple threads for maximum search speed:
   ```bash
   julia -t auto --project=.
   ```
   Then in the REPL:
   ```julia
   using Cerebro
   Cerebro.start_server()  # uses default CerebroConfig()
   ```

3. **Chat** — Open [http://localhost:8000](http://localhost:8000) in your browser and start asking questions!

> [!TIP]
> On first launch, Cerebro will automatically ingest all documents, generate embeddings via Ollama, quantize them to binary, and save the index to `vector_db.jld2`. Subsequent launches load the pre-built index instantly.

## 📁 Project Structure

```
Cerebro/
├── src/
│   ├── Cerebro.jl        # Main module entry point
│   ├── Config.jl          # Centralized CerebroConfig struct
│   ├── Ingestion.jl       # Document loading, chunking, embedding & quantization
│   ├── Backend.jl         # Hamming search, MaxHeap top-k, prompt generation
│   └── App.jl             # HTTP.jl web server, chat UI, API routes
├── public/
│   └── index.html         # Chat UI frontend
├── data/
│   └── documents/         # Place your .txt and .md files here
├── Project.toml
└── vector_db.jld2         # Auto-generated binary index (after first run)
```

## ⚙️ How It Works

### Binary Quantization

Traditional vector search compares hundreds of `Float32` values per embedding. Cerebro compresses each embedding to a binary vector where each float becomes a **single bit** (positive → 1, negative → 0). The dimension is auto-detected from your embedding model:

```
e.g. 768 floats × 4 bytes = 3,072 bytes  →  768 bits ÷ 8 = 96 bytes  (32× smaller)
```

### Hamming Distance with SIMD

Distance between binary vectors is computed via XOR + popcount — a single CPU instruction per byte:

```julia
@inline hamming_distance(x1::T, x2::T) where T<:Integer = count_ones(x1 ⊻ x2)
```

### MaxHeap Top-K Selection

Instead of sorting all N distances, a MaxHeap maintains only the top-k closest results in O(N log k) time with zero allocations after initialization.

### Parallel Search

The database is partitioned across CPU threads. Each thread maintains its own MaxHeap, and results are merged at the end:

```julia
k_closest_parallel(db, query, k)  # Automatically uses all available threads
```

## 🔧 Configuration

All parameters are centralized in the `CerebroConfig` struct (`src/Config.jl`). Pass a custom config to `start_server`:

```julia
config = CerebroConfig(
    port = 3000,
    docs_folder = "my/custom/docs",
    chat_model = "gemma3:1b",
    parent_words = 200,
)
Cerebro.start_server(config=config)
```

| Parameter | Default | Description |
|---|---|---|
| `parent_words` | 100 | Words per parent chunk |
| `child_words` | 25 | Words per child chunk |
| `parent_overlap` | 25 | Word overlap between parent chunks |
| `child_overlap` | 5 | Word overlap between child chunks |
| `embedding_model` | `embeddinggemma:300m-qat-q4_0` | Ollama embedding model |
| `chat_model` | `gemma3:1b` | Ollama chat model |
| `port` | 8000 | HTTP server port |
| `retrieval_k` | 2 | Number of parent chunks to retrieve |
| `db_path` | `vector_db.jld2` | Path to the persisted index |
| `docs_folder` | `data/documents` | Folder to scan for documents |

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
  <sub>Built with ❤️ in Julia</sub>
</div>
