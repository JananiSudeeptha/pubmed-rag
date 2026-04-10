# Chunking Strategy Comparison

## Implemented Strategies

### 1. Fixed-Size Chunking
- **Description**: Splits the abstract text into chunks of fixed word count (500 words) with an overlap of 50 words between consecutive chunks.
- **Implementation**: Uses simple word splitting and sliding window approach.
- **Pros**:
  - Simple and predictable chunk sizes
  - Easy to implement and control
  - Consistent processing time
- **Cons**:
  - Can split sentences or concepts in the middle
  - May not respect semantic boundaries
  - Overlap might include irrelevant context

### 2. Semantic Chunking
- **Description**: Splits the abstract into sentences using NLTK's sentence tokenizer, then groups sentences into chunks of approximately 500 words.
- **Implementation**: Uses NLTK for sentence boundary detection, then accumulates sentences until reaching the chunk size limit. Section labels are extracted directly from abstract header text when available, so the chunking is driven by the paper's own structure rather than a hardcoded section list.
- **Pros**:
  - Respects natural language boundaries
  - Preserves sentence integrity
  - Better semantic coherence within chunks
- **Cons**:
  - Variable chunk sizes (though approximately controlled)
  - Depends on sentence segmentation accuracy
  - Slightly more complex implementation

## Rationale

### Chunk Size (500 words)
- Chosen to approximate 500-1000 tokens, which fits well within common embedding model limits (e.g., BERT's 512 token limit)
- Provides enough context for meaningful embeddings while keeping chunks manageable for vector databases
- Balances between having sufficient context and avoiding overly long chunks that dilute specific information

### Overlap (50 words)
- Allows continuity of context between chunks
- Helps prevent loss of information at chunk boundaries
- Common practice in text chunking to maintain coherence

### Strategy Choice
- **Fixed-size**: Selected for its simplicity and reliability in production environments
- **Semantic**: Chosen to demonstrate the benefits of respecting linguistic structure

## Production Recommendation

For production deployment, I recommend **semantic chunking** over fixed-size chunking. Here's why:

1. **Better Retrieval Quality**: By preserving sentence boundaries, semantic chunking maintains the integrity of ideas and concepts, leading to more accurate and relevant retrieval results.

2. **Improved Embedding Quality**: Embeddings trained on coherent text units (sentences) are likely to capture semantic meaning better than embeddings from arbitrarily split text.

3. **User Experience**: Retrieved chunks that contain complete thoughts are more useful for users reading the context.

4. **Scalability**: While slightly more computationally intensive, the benefits outweigh the minimal performance cost.

The semantic approach aligns better with how humans process and understand text, making it more suitable for a medical literature RAG system where accuracy and context preservation are critical.