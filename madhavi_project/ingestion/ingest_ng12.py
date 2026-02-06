from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import chromadb
import vertexai
from pypdf import PdfReader
from vertexai.language_models import TextEmbeddingModel


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > 0 else end
    return [c.strip() for c in chunks if c.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="data/ng12.pdf")
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--collection", default="ng12")
    parser.add_argument("--chroma-path", default="data/chroma")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit("NG12 PDF not found")

    vertexai.init(project=args.project, location=args.location)
    embed_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

    reader = PdfReader(str(pdf_path))
    pages = reader.pages

    client = chromadb.PersistentClient(path=args.chroma_path)
    if args.force:
        try:
            client.delete_collection(args.collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(args.collection)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[dict] = []
    embeds: List[List[float]] = []

    for i, page in enumerate(pages, start=1):
        text = page.extract_text() or ""
        for j, chunk in enumerate(chunk_text(text, args.chunk_size, args.overlap)):
            chunk_id = f"ng12_p{i}_c{j}"
            ids.append(chunk_id)
            docs.append(chunk)
            metas.append({"source": "NG12 PDF", "page": i, "chunk_id": chunk_id})

    if not docs:
        raise SystemExit("No text extracted from NG12 PDF")

    for doc in docs:
        emb = embed_model.get_embeddings([doc])[0].values
        embeds.append(emb)

    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeds)

    stats = {
        "chunks": len(ids),
        "pages": len(pages),
        "collection": args.collection,
    }
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
