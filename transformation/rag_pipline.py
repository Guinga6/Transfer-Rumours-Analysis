from typing import List, Tuple
import hashlib

def split_text(text: str, max_tokens: int = 100, overlap: int = 15) -> List[str]:
    """
    Split text into overlapping chunks based on word count.

    :param text: The input transcript text.
    :param max_tokens: Maximum words per chunk.
    :param overlap: Number of overlapping words between chunks.
    :return: List of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_tokens)
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

def filter_chunks(chunks: List[str], names: List[str]) -> List[str]:
    """
    Filter chunks to retain only those mentioning any of the given names.

    :param chunks: List of text chunks.
    :param names: List of target names (players, coaches, clubs).
    :return: List of relevant chunks.
    """
    lower_names = [name.lower() for name in names]
    return [chunk for chunk in chunks if any(name in chunk.lower() for name in lower_names)]

def deduplicate_chunks(chunks: List[str]) -> List[str]:
    """
    Remove near-duplicate chunks based on hashing.

    :param chunks: List of text chunks.
    :return: List of deduplicated chunks.
    """
    seen_hashes = set()
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hashlib.md5(chunk.strip().lower().encode()).hexdigest()
        if chunk_hash not in seen_hashes:
            seen_hashes.add(chunk_hash)
            unique_chunks.append(chunk)
    return unique_chunks

def build_rag_input(names: List[str], text: str) -> Tuple[str, str]:
    """
    Build system and user prompt using a simple RAG-style approach.

    :param names: List of target names.
    :param text: Full transcript text.
    :return: Tuple containing the system and user prompts for the chat model.
    """
    chunks = split_text(text)
    relevant_chunks = filter_chunks(chunks, names)
    deduplicated_chunks = deduplicate_chunks(relevant_chunks)
    merged_text = '\n'.join(deduplicated_chunks)
    return merged_text
