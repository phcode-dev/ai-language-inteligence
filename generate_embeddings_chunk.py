#!/usr/bin/env python3
"""
Enhanced Code Search System with Analytics
Supports semantic code search with gitignore integration and detailed analytics
"""

import sys
import os
import json
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant_client.http.models import VectorParams
from transformers import AutoTokenizer, AutoModel
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.preview.tokenization import get_tokenizer_for_model
import numpy as np
import time
import warnings
import fnmatch
from pathlib import Path
import psutil

# Suppress warnings
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")

# Initialize Vertex AI
model_name = "gemini-1.5-pro-002"
tokenizer = get_tokenizer_for_model(model_name)

# Initialize Qdrant client
client = QdrantClient(url="http://localhost:6333")
text_collection_name = "text_embeddings"

# Use a lightweight embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Loading text embedding model: {embedding_model_name}")

# Initialize embedding models
text_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
text_tokenizer.model_max_length = int(1e12)  # Set to a large value
text_model = AutoModel.from_pretrained(embedding_model_name)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_model.to(device)

# Initialize Vertex AI (Replace with your project ID)
vertexai.init(project="the-racer-353515", location="us-central1")
vertex_model = GenerativeModel(model_name)
generation_config = {"max_output_tokens": 8192, "temperature": 0.5, "top_p": 0.95}

# Supported file extensions
supported_extensions = {'.js', '.ts', '.json', '.md', '.py', '.html'}

class Analytics:
    """Track and report analytics for code search operations."""
    def __init__(self):
        self.start_time = time.time()
        self.checkpoints = {}
        self.token_counts = {
            'refined_queries': 0,
            'context': 0,
            'llm_response': 0
        }
        self.operation_times = {}
        self.embedding_count = 0
        self.total_tokens = 0
        self.memory_usage = {}
        self.index_stats = {}

    def checkpoint(self, name):
        """Record a timing checkpoint."""
        current = time.time()
        if self.checkpoints:
            last_time = max(self.checkpoints.values())
            self.operation_times[name] = current - last_time
        self.checkpoints[name] = current

    def track_tokens(self, category, count):
        """Track token usage by category."""
        self.token_counts[category] = count
        self.total_tokens += count

    def get_memory_usage(self):
        """Get current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def generate_report(self, operation_type):
        """Generate analytics report."""
        total_time = time.time() - self.start_time
        current_memory = self.get_memory_usage()

        report = f"""
=== {operation_type} Analytics Report ===

Timing Information:
------------------
Total Time: {total_time:.2f} seconds
"""
        for op, duration in self.operation_times.items():
            report += f"{op}: {duration:.2f} seconds\n"

        if operation_type == "Query":
            report += f"""
Token Usage:
-----------
Refined Queries: {self.token_counts['refined_queries']}
Context: {self.token_counts['context']}
LLM Response: {self.token_counts['llm_response']}
Total Tokens: {self.total_tokens}
"""

        report += f"""
Resource Usage:
--------------
Memory Usage: {current_memory:.2f} MB
Embedding Operations: {self.embedding_count}
"""
        return report

def parse_gitignore(project_path):
    """Parse .gitignore file and return patterns to exclude."""
    gitignore_patterns = set()
    gitignore_path = os.path.join(project_path, '.gitignore')
    
    if os.path.exists(gitignore_path):
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if line.endswith('/'):
                        line = line + '**'
                    if not line.startswith('!'):  # Skip negation patterns
                        if not line.startswith('/'):
                            line = f"**/{line}"
                        gitignore_patterns.add(line)
    
    return gitignore_patterns

def should_ignore(file_path, base_path, gitignore_patterns):
    """Check if a file should be ignored based on gitignore patterns."""
    relative_path = os.path.relpath(file_path, base_path)
    relative_path = relative_path.replace(os.sep, '/')
    
    for pattern in gitignore_patterns:
        if pattern.startswith('/'):
            pattern = pattern[1:]
        if fnmatch.fnmatch(relative_path, pattern):
            return True
        path_parts = relative_path.split('/')
        for i in range(len(path_parts)):
            if fnmatch.fnmatch('/'.join(path_parts[i:]), pattern):
                return True
    return False

def generate_file_tree(project_path, gitignore_patterns, max_depth=4):
    """Generate a hierarchical file tree structure."""
    tree_structure = []
    base_path = Path(project_path)
    
    def _build_tree(directory, depth=0):
        if depth > max_depth:
            return ["    " * depth + "..."]
        
        items = []
        try:
            paths = sorted(directory.iterdir(), 
                         key=lambda x: (not x.is_dir(), x.name.lower()))
            
            for path in paths:
                relative_path = str(path.relative_to(base_path))
                
                if should_ignore(str(path), str(base_path), gitignore_patterns):
                    continue
                
                if path.name in {'.git', 'node_modules', 'build', '.nyc_output', 'coverage'}:
                    continue
                
                prefix = "    " * depth
                if path.is_dir():
                    items.append(f"{prefix}└── 📁 {path.name}/")
                    items.extend(_build_tree(path, depth + 1))
                else:
                    if path.suffix in {'.js', '.ts', '.py', '.java', '.cpp', '.h', '.hpp', '.md', '.json'}:
                        items.append(f"{prefix}└── 📄 {path.name}")
        except PermissionError:
            items.append("    " * depth + "Permission denied")
        return items

    tree_structure = _build_tree(base_path)
    return "\n".join(tree_structure)

def split_text_into_chunks(text, max_length=384, stride=128):
    """Split text into overlapping chunks."""
    tokens = text_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_length - stride):
        chunk_tokens = tokens[i:i + max_length]
        chunks.append(chunk_tokens)
    return chunks

def generate_embedding_dual(token_ids):
    """Generate embedding vector for token IDs."""
    max_length = 384
    token_ids = token_ids[:max_length]
    tokens_tensor = torch.tensor([token_ids], device=device)
    attention_mask = torch.ones(tokens_tensor.shape, dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = text_model(input_ids=tokens_tensor, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]

    embedding_vector = embeddings[0].cpu().numpy()
    embedding_vector /= (np.linalg.norm(embedding_vector) + 1e-12)

    return embedding_vector, get_vector_size()

def get_vector_size():
    """Return embedding vector size."""
    return 384

def is_valid_vector(vector, expected_size):
    """Check if vector is valid."""
    return vector is not None and len(vector) == expected_size

def create_collection(name, vector_size):
    """Create or recreate Qdrant collection."""
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=vector_size, distance="Cosine")
    )

def exclude_minified_code(file_path):
    """Check if file contains minified code."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) < 3:
                return True, "File has fewer than 3 lines"

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if "parcelRequire" in content:
                return True, "Contains 'parcelRequire' (bundled code)"

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(line) > 2000:
                    return True, "Contains very long lines (>2000 characters)"

        if file_path.endswith(".min.js"):
            return True, "File is a minified JavaScript file"

    except Exception as e:
        return True, f"Error during file filtering: {e}"

    return False, None

def fetch_metadata_for_query(project_path):
    """Fetch metadata from README and package.json."""
    metadata_files = []
    readme_path = os.path.join(project_path, "README.md")
    package_path = os.path.join(project_path, "package.json")

    if os.path.isfile(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            metadata_files.append({"file": "README.md", "content": f.read()})

    if os.path.isfile(package_path):
        with open(package_path, "r", encoding="utf-8") as f:
            metadata_files.append({"file": "package.json", "content": json.dumps(json.load(f), indent=2)})

    return metadata_files

def get_adjacent_chunks(chunk_id, file_path, num_adjacent=1):
    """Get adjacent chunks for context."""
    adjacent_chunk_ids = [chunk_id - i for i in range(num_adjacent, 0, -1)] + \
                        [chunk_id] + \
                        [chunk_id + i for i in range(1, num_adjacent + 1)]
    adjacent_chunk_ids = [cid for cid in adjacent_chunk_ids if cid > 0]

    adjacent_chunks = []
    for cid in adjacent_chunk_ids:
        res = client.retrieve(
            collection_name=text_collection_name,
            ids=[cid]
        )
        if res and res[0].payload["file"] == file_path:
            adjacent_chunks.append(res[0])

    return adjacent_chunks

def create_llm_prompt(query, context):
    """Create the LLM prompt for code analysis."""
    return f"""
You are an expert software developer. The following are code snippets from a project. Some snippets may be partial due to context limitations.

Original Query: {query}

Below is the project structure and relevant code sections found. Analyze the code and provide
a comprehensive response that:

1. Summarizes the relevant code sections found
2. Explains how they relate to the query
3. Analyzes the code location in the project structure
4. Suggests any related files worth exploring based on the project layout
5. Provides specific code examples or modifications if applicable
6. Highlights any patterns or architectural decisions visible in the code

Project Context:
{context}

Please provide a detailed response, including code examples if applicable, and make reasonable assumptions to fill in any missing context.
"""

def prepare_context(results, file_tree, project_path):
    """Prepare context for LLM analysis."""
    relevant_files = set()
    context_pieces = []
    
    for res in results:
        file_path = res.payload["file"]
        relevant_files.add(file_path)
        score = res.score
        relative_path = os.path.relpath(file_path, project_path)
        
        context_pieces.append(f"\nFile: {relative_path} (Similarity: {score:.4f})")
        context_pieces.append(res.payload["text"])
        context_pieces.append("-" * 40)

    context = f"""
Project Structure:
{file_tree}

Relevant Files:
{chr(10).join(f"  - {os.path.relpath(f, project_path)}" for f in relevant_files)}

Code Sections:
{chr(10).join(context_pieces)}
"""
    return context

def process_file_and_store_embeddings(project_path):
    """Enhanced indexing with comprehensive analytics."""
    analytics = Analytics()
    analytics.checkpoint("start")

    if not os.path.isdir(project_path):
        print(f"Directory '{project_path}' does not exist.")
        sys.exit(1)

    # Initialize analytics counters
    analytics.index_stats = {
        'total_files': 0,
        'processed_files': 0,
        'skipped_files': 0,
        'error_files': 0,
        'total_chunks': 0,
        'total_embeddings': 0,
        'total_tokens': 0,
        'batch_operations': 0,
        'file_types': {},
        'skip_reasons': {},
        'file_sizes': {
            '0-10KB': 0,
            '10-100KB': 0,
            '100KB-1MB': 0,
            '>1MB': 0
        }
    }

    # Parse gitignore patterns
    gitignore_patterns = parse_gitignore(project_path)
    analytics.index_stats['gitignore_patterns'] = len(gitignore_patterns)
    analytics.checkpoint("gitignore_parsing")

    try:
        create_collection(text_collection_name, get_vector_size())
        analytics.checkpoint("collection_creation")
    except Exception as e:
        print(f"Error creating collection: {e}")
        sys.exit(1)

    point_id = 1
    points_batch = []
    batch_size = 100
    indexed_files = []
    skipped_files = []
    error_files = []
    
    analytics.memory_usage['start'] = analytics.get_memory_usage()

    for root, dirs, files in os.walk(project_path):
        analytics.checkpoint(f"dir_scan_{root}")
        
        original_dir_count = len(dirs)
        dirs[:] = [d for d in dirs if not should_ignore(
            os.path.join(root, d),
            project_path,
            gitignore_patterns
        )]
        analytics.index_stats['filtered_directories'] = original_dir_count - len(dirs)

        for file in files:
            file_path = os.path.join(root, file)
            analytics.index_stats['total_files'] += 1
            
            # Track file type and size
            _, ext = os.path.splitext(file)
            analytics.index_stats['file_types'][ext] = analytics.index_stats['file_types'].get(ext, 0) + 1
            
            file_size = os.path.getsize(file_path)
            if file_size <= 10 * 1024:
                analytics.index_stats['file_sizes']['0-10KB'] += 1
            elif file_size <= 100 * 1024:
                analytics.index_stats['file_sizes']['10-100KB'] += 1
            elif file_size <= 1024 * 1024:
                analytics.index_stats['file_sizes']['100KB-1MB'] += 1
            else:
                analytics.index_stats['file_sizes']['>1MB'] += 1

            skip_reason = None

            if ext not in supported_extensions:
                skip_reason = f"Unsupported extension: {ext}"
            elif should_ignore(file_path, project_path, gitignore_patterns):
                skip_reason = "Matched gitignore pattern"
            elif file_size > 1 * 1024 * 1024:
                skip_reason = "File too large"
            else:
                is_excluded, reason = exclude_minified_code(file_path)
                if is_excluded:
                    skip_reason = f"Minified code: {reason}"

            if skip_reason:
                skipped_files.append((file_path, skip_reason))
                analytics.index_stats['skip_reasons'][skip_reason] = \
                    analytics.index_stats['skip_reasons'].get(skip_reason, 0) + 1
                analytics.index_stats['skipped_files'] += 1
                continue

            try:
                analytics.checkpoint(f"file_start_{file_path}")
                
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                file_tokens = len(content.split())
                analytics.index_stats['total_tokens'] += file_tokens

                chunks = split_text_into_chunks(content)
                analytics.index_stats['total_chunks'] += len(chunks)
                
                for chunk_tokens in chunks[:10]:
                    embedding, expected_size = generate_embedding_dual(chunk_tokens)
                    analytics.index_stats['total_embeddings'] += 1
                    
                    if not is_valid_vector(embedding, expected_size):
                        continue

                    chunk_text = text_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    point = PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload={
                            "file": file_path,
                            "chunk_id": point_id,
                            "text": chunk_text
                        }
                    )
                    points_batch.append(point)
                    point_id += 1

                    if len(points_batch) >= batch_size:
                        analytics.checkpoint(f"batch_upload_{analytics.index_stats['batch_operations']}")
                        client.upsert(collection_name=text_collection_name, points=points_batch)
                        analytics.index_stats['batch_operations'] += 1
                        points_batch = []

                indexed_files.append(file_path)
                analytics.index_stats['processed_files'] += 1
                analytics.checkpoint(f"file_end_{file_path}")

            except Exception as e:
                error_files.append((file_path, str(e)))
                analytics.index_stats['error_files'] += 1
                continue

    # Process remaining batch
    if points_batch:
        analytics.checkpoint("final_batch_upload")
        client.upsert(collection_name=text_collection_name, points=points_batch)
        analytics.index_stats['batch_operations'] += 1

    analytics.memory_usage['end'] = analytics.get_memory_usage()
    analytics.checkpoint("indexing_complete")

    # Generate report
    report = f"""=== Indexing Analytics Report ===
{analytics.generate_report("Indexing")}

Detailed Statistics:
------------------
Total Files Found: {analytics.index_stats['total_files']}
Files Successfully Processed: {analytics.index_stats['processed_files']}
Files Skipped: {analytics.index_stats['skipped_files']}
Files with Errors: {analytics.index_stats['error_files']}

Content Statistics:
-----------------
Total Chunks Generated: {analytics.index_stats['total_chunks']}
Total Embeddings Created: {analytics.index_stats['total_embeddings']}
Total Tokens Processed: {analytics.index_stats['total_tokens']}
Batch Operations: {analytics.index_stats['batch_operations']}

File Sizes:
----------
0-10KB: {analytics.index_stats['file_sizes']['0-10KB']} files
10-100KB: {analytics.index_stats['file_sizes']['10-100KB']} files
100KB-1MB: {analytics.index_stats['file_sizes']['100KB-1MB']} files
>1MB: {analytics.index_stats['file_sizes']['>1MB']} files

File Types:
----------
"""
    for ext, count in analytics.index_stats['file_types'].items():
        report += f"{ext}: {count} files\n"

    report += "\nSkip Reasons:\n-------------\n"
    for reason, count in analytics.index_stats['skip_reasons'].items():
        report += f"{reason}: {count} files\n"

    with open("indexing_analytics.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nIndexing completed successfully!")
    print(f"Processed {analytics.index_stats['processed_files']} files")
    print(f"Generated {analytics.index_stats['total_embeddings']} embeddings")
    print(f"Total time: {analytics.operation_times['indexing_complete']:.2f} seconds")
    print("\nDetailed analytics saved to: indexing_analytics.txt")

    return analytics.index_stats

def refine_query_with_llm(query, project_path):
    """Refine search query using LLM with a focus on actionable output."""
    print("Refining query with extended project context...")
    
    # Gather project metadata
    metadata_results = fetch_metadata_for_query(project_path)
    metadata_snippets = "\n\n".join(
        [f"File: {meta['file']}\nContent: {meta['content'][:500]}" for meta in metadata_results]
    )

    # Directory structure as part of the context
    directory_context = "\n".join(
        [os.path.join(dp, f) for dp, dn, filenames in os.walk(project_path) for f in filenames][:20]
    )  # Include up to 20 file paths

    # Combine query with metadata and directory structure
    extended_context = f"""
    Project Overview:
    - Key files and their content snippets:
    {metadata_snippets}

    - Project directory structure:
    {directory_context}

    User Query: {query}
    """

    # Prompt LLM to refine the query with a focus on actionable output
    prompt = (
        f"Refine the following user query for searching within a project codebase. "
        f"Consider the project context below and provide 3-5 actionable queries "
        f"that are concise and relevant to the codebase:\n\n{extended_context}"
    )

    try:
        response = vertex_model.generate_content([prompt], generation_config=generation_config)
        
        # Parse the LLM response into a list of actionable refined queries
        refined_queries = response.text.strip().split("\n")
        # Filter out any verbose or non-actionable lines
        actionable_queries = [rq.strip() for rq in refined_queries if rq.strip() and not rq.startswith(("**", "-"))]

        if not actionable_queries:
            print("Fallback: Using the original query as no refined queries were generated.")
            return [query]

        print("Generated refined queries:")
        for q in actionable_queries:
            print(f"  - {q}")
            
        return actionable_queries

    except Exception as e:
        print(f"Query refinement failed: {e}")
        print("Falling back to original query")
        return [query]
    
def query_abstract_description(query, project_path, top_k=3):
    """Search codebase with semantic understanding and display similarity scores."""
    analytics = Analytics()
    analytics.checkpoint("start")

    print("\n=== Query Execution Started ===")
    print(f"Query: {query}")

    # Get metadata and refine query
    metadata_results = fetch_metadata_for_query(project_path)
    analytics.checkpoint("metadata_fetch")
    
    # Generate project tree at the beginning
    gitignore_patterns = parse_gitignore(project_path)
    project_tree = generate_file_tree(project_path, gitignore_patterns)
    print("\n=== Project Structure ===")
    print(project_tree)
    print("\n" + "=" * 80 + "\n")
    
    refined_queries = refine_query_with_llm(query, project_path)
    analytics.track_tokens('refined_queries', sum(len(q.split()) for q in refined_queries))
    analytics.checkpoint("query_refinement")

    # Search process
    all_results = []
    relevant_files = {}  # Changed to dict to store scores
    similarity_threshold = 0.55

    print("\n=== Search Results ===")
    print("Finding relevant code sections...")

    for refined_query in refined_queries:
        query_tokens = text_tokenizer.encode(refined_query, add_special_tokens=False)
        query_embedding, _ = generate_embedding_dual(query_tokens)
        analytics.embedding_count += 1

        if not is_valid_vector(query_embedding, get_vector_size()):
            continue

        results = client.search(
            collection_name=text_collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )

        filtered_results = [res for res in results if res.score > similarity_threshold]
        all_results.extend(filtered_results)
        
        # Store best similarity score for each file
        for res in filtered_results:
            file_path = res.payload["file"]
            if file_path not in relevant_files or res.score > relevant_files[file_path]:
                relevant_files[file_path] = res.score

    analytics.checkpoint("vector_search")

    # Sort results by score and remove duplicates
    all_results.sort(key=lambda x: x.score, reverse=True)
    unique_results = []
    seen_files = set()
    
    print("\nTop matching files by similarity score:")
    print("-" * 80)
    
    # Display sorted results with similarity scores
    sorted_files = sorted(relevant_files.items(), key=lambda x: x[1], reverse=True)
    for file_path, score in sorted_files:
        rel_path = os.path.relpath(file_path, project_path)
        print(f"Score: {score:.4f} | File: {rel_path}")
        if file_path not in seen_files:
            for res in all_results:
                if res.payload["file"] == file_path:
                    unique_results.append(res)
                    seen_files.add(file_path)
                    break

    print("-" * 80)

    # Generate context with project tree
    context = f"""
Project Structure:
{project_tree}

Relevant Files and Code Sections:
"""
    for res in unique_results[:top_k]:
        file_path = res.payload["file"]
        rel_path = os.path.relpath(file_path, project_path)
        score = relevant_files[file_path]
        context += f"\n--- {rel_path} (Score: {score:.4f}) ---\n"
        context += res.payload["text"]
        context += "\n" + "-" * 80 + "\n"

    analytics.track_tokens('context', len(context.split()))
    analytics.checkpoint("context_generation")

    # Save detailed results to file
    results_dump = f"""=== Code Search Results ===
Query: {query}

Project Structure:
{project_tree}

Refined Queries:
{chr(10).join(f"- {q}" for q in refined_queries)}

Found Files (sorted by similarity):
{chr(10).join(f"Score: {score:.4f} | File: {os.path.relpath(path, project_path)}" for path, score in sorted_files)}

=== File Contents ===
"""
    
    # Add file contents with scores
    for res in unique_results[:top_k]:
        file_path = res.payload["file"]
        rel_path = os.path.relpath(file_path, project_path)
        score = relevant_files[file_path]
        results_dump += f"\n--- {rel_path} (Score: {score:.4f}) ---\n"
        results_dump += res.payload["text"]
        results_dump += "\n" + "-" * 80 + "\n"

    with open("search_results_detailed.txt", "w", encoding="utf-8") as f:
        f.write(results_dump)

    # Query LLM with enhanced context including project structure
    prompt = f"""
You are an expert software developer. The following are code snippets from a project. Some snippets may be partial due to context limitations.

Original Query: {query}

Below is the project structure and relevant code sections found. Analyze the code and provide
a comprehensive response that:

1. Summarizes the project structure and organization
2. Explains how the found code sections relate to the query
3. Analyzes the code location in the project structure
4. Suggests any related files worth exploring based on the project layout
5. Provides specific code examples or modifications if applicable
6. Highlights any patterns or architectural decisions visible in the code

Context:
{context}

Please provide a detailed response, including code examples if applicable, and make reasonable assumptions to fill in any missing context.
"""

    print("\nAnalyzing code sections with LLM...")
    llm_response = vertex_model.generate_content(
        [prompt],
        generation_config=generation_config
    )
    analytics.track_tokens('llm_response', len(llm_response.text.split()))
    analytics.checkpoint("llm_response")

    # Update analytics report with similarity scores and project structure
    report = f"""=== Query Analytics Report ===
{analytics.generate_report("Query")}

Project Structure:
----------------
{project_tree}

Query Information:
----------------
Original Query: {query}
Refined Queries: {len(refined_queries)}
Results Found: {len(all_results)}
Unique Results: {len(unique_results)}

Matched Files with Similarity Scores:
---------------------------------
{chr(10).join(f"Score: {score:.4f} | File: {os.path.relpath(path, project_path)}" for path, score in sorted_files)}

Token Usage:
----------
Query Refinement: {analytics.token_counts['refined_queries']}
Context Generation: {analytics.token_counts['context']}
LLM Response: {analytics.token_counts['llm_response']}
Total Tokens: {analytics.total_tokens}
"""

    with open("query_analytics.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSearch completed! Detailed results saved to 'search_results_detailed.txt'")
    print("Analytics saved to 'query_analytics.txt'")
    
    # Return combined response with project tree
    final_response = f"""
=== Project Structure ===
{project_tree}

=== Analysis ===
{llm_response.text}
"""
    return final_response
 
def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("""
Code Search Tool

Usage:
  python script.py index <project_path>
    Index a project directory for searching.
    
  python script.py query <project_path> "<query_string>"
    Search the indexed project with a query.

Examples:
  python script.py index ./my-project
  python script.py query ./my-project "find authentication implementation"
        """)
        sys.exit(1)

    try:
        command = sys.argv[1]
        
        if command == "index":
            if len(sys.argv) < 3:
                print("Error: Project path required for indexing")
                sys.exit(1)
            project_path = os.path.abspath(sys.argv[2])
            print(f"Indexing project: {project_path}")
            process_file_and_store_embeddings(project_path)
            
        elif command == "query":
            if len(sys.argv) < 4:
                print("Error: Project path and query string required")
                sys.exit(1)
            project_path = os.path.abspath(sys.argv[2])
            query = sys.argv[3]
            print(f"Searching project: {project_path}")
            print(f"Query: {query}")
            result = query_abstract_description(query, project_path)
            print("\nSearch Results:")
            print(result)
            print("\nDetailed analytics saved to: query_analytics.txt")
            
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()