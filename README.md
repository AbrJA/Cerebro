<div align="center">

# 🧠 Cerebro

**A blazing-fast, fully local RAG application built in Julia**

[![Julia](https://img.shields.io/badge/Julia-1.12+-9558B2?logo=julia&logoColor=white)](https://julialang.org)
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
- **💬 Web Chat Interface** — A clean, modern chat UI powered by [Genie.jl](https://genieframework.com/), complete with typing indicators and source citations.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Genie.jl Web UI                       │
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

1. **Julia 1.12+** — [Install Julia](https://julialang.org/downloads/)
2. **Ollama** — [Install Ollama](https://ollama.com/download)
3. Pull the required models:
   ```bash
   ollama pull gemma3:1b
   ollama pull embeddinggemma:300m-qat-q4_0
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
   Cerebro.App.start_server(port=8000)
   ```

3. **Chat** — Open [http://localhost:8000](http://localhost:8000) in your browser and start asking questions!

> [!TIP]
> On first launch, Cerebro will automatically ingest all documents, generate embeddings via Ollama, quantize them to binary, and save the index to `vector_db.json`. Subsequent launches load the pre-built index instantly.

## 📁 Project Structure

```
Cerebro/
├── src/
│   ├── Cerebro.jl        # Main module entry point
│   ├── Ingestion.jl       # Document loading, chunking, embedding & quantization
│   ├── Backend.jl         # Hamming search, MaxHeap top-k, prompt generation
│   └── App.jl             # Genie web server, chat UI, API routes
├── data/
│   └── documents/         # Place your .txt and .md files here
├── Project.toml
└── vector_db.json         # Auto-generated binary index (after first run)
```

## ⚙️ How It Works

### Binary Quantization

Traditional vector search compares 768+ `Float32` values per embedding (~3 KB each). Cerebro compresses each embedding to a binary vector where each float becomes a **single bit** (positive → 1, negative → 0):

```
768 floats × 4 bytes = 3,072 bytes   →   768 bits ÷ 8 = 96 bytes (32× smaller)
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

You can customize the server and chunking parameters:

```julia
# Custom port and document folder
Cerebro.App.start_server(port=3000, folder_path="my/custom/docs")
```

Chunking parameters can be adjusted in `Ingestion.jl`:

| Parameter | Default | Description |
|---|---|---|
| `parent_words` | 300 | Words per parent chunk |
| `child_words` | 75 | Words per child chunk |
| `parent_overlap` | 50 | Word overlap between parent chunks |
| `child_overlap` | 15 | Word overlap between child chunks |

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
  <sub>Built with ❤️ in Julia</sub>
</div>
