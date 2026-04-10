import re
import json
import nltk
from nltk.tokenize import sent_tokenize
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.config_reader import DataIngestionConfig
from ingest.pubmed_fetcher import PubMedFetcher

# Download punkt_tab if not already
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class Chunker:
    def __init__(self, chunk_size=500, overlap=50):
        """
        Initialize the chunker with chunk size (in words) and overlap.

        Args:
            chunk_size (int): Number of words per chunk
            overlap (int): Number of words to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_saved_pmids(self, file_path):
        """
        Load PMIDs from a saved JSON file.

        The file can contain:
          - a list of PMID strings
          - a list of article dictionaries with a 'pmid' field
          - a dict with keys 'pmids' or 'articles'
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return data
            return [item.get('pmid') for item in data if isinstance(item, dict) and item.get('pmid')]

        if isinstance(data, dict):
            if 'pmids' in data and isinstance(data['pmids'], list):
                return data['pmids']
            if 'articles' in data and isinstance(data['articles'], list):
                return [article.get('pmid') for article in data['articles'] if article.get('pmid')]

        raise ValueError('Saved PMID file must contain a list of PMIDs or a list of article dictionaries.')

    def fetch_articles_for_saved_pmids(self, saved_data_file):
        """
        Fetch article metadata and abstracts from PubMed using saved PMIDs.

        Returns a list of ArticleData objects.
        """
        pmids = self.load_saved_pmids(saved_data_file)
        fetcher = PubMedFetcher()
        return fetcher.fetch_article_details(pmids)

    def _normalize_section(self, header):
        normalized = header.strip().lower()
        known_sections = {
            'background': 'Background',
            'objective': 'Objective',
            'objectives': 'Objectives',
            'design': 'Design',
            'participants': 'Participants',
            'interventions': 'Interventions',
            'outcomes': 'Outcomes',
            'methods': 'Methods',
            'results': 'Results',
            'findings': 'Findings',
            'conclusion': 'Conclusions',
            'conclusions': 'Conclusions',
            'purpose': 'Purpose',
            'patients': 'Patients',
            'materials and methods': 'Materials and Methods'
        }
        return known_sections.get(normalized, header.strip().title())

    def _extract_sections(self, text):
        if not text:
            return [{'section': 'Abstract', 'text': ''}]

        header_pattern = r'(?m)^(?P<header>[A-Z][A-Za-z0-9 &/\-]{2,80}?)\s*[:\-–]\s*'
        matcher = re.compile(header_pattern)
        matches = list(matcher.finditer(text))

        if not matches:
            return [{'section': 'Abstract', 'text': text.strip()}]

        sections = []
        if matches[0].start() > 0:
            leading_text = text[:matches[0].start()].strip()
            if leading_text:
                sections.append({'section': 'Abstract', 'text': leading_text})

        for index, match in enumerate(matches):
            section_header = self._normalize_section(match.group('header'))
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()
            if section_text:
                sections.append({'section': section_header, 'text': section_text})

        return sections

    def fixed_size_chunking(self, article):
        """
        Fixed-size chunking strategy: split text into chunks of fixed word count with overlap.

        Args:
            article (dict): Article dictionary with 'pmid', 'title', 'abstract', etc.

        Returns:
            list: List of chunk dictionaries with metadata
        """
        source_text = article.get('full_text') or article.get('abstract', '')
        sections = self._extract_sections(source_text)
        chunks = []
        chunk_index = 0

        for section in sections:
            words = section['text'].split()
            start = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                chunk_text = ' '.join(words[start:end])
                chunks.append({
                    'text': chunk_text,
                    'pmid': article.get('pmid'),
                    'title': article.get('title', ''),
                    'section': section['section'],
                    'chunk_index': chunk_index
                })
                start += self.chunk_size - self.overlap
                chunk_index += 1

        return chunks

    def semantic_chunking(self, article):
        """
        Semantic chunking strategy: split by sentences and group into chunks of similar size.

        Args:
            article (dict): Article dictionary

        Returns:
            list: List of chunk dictionaries with metadata
        """
        source_text = article.get('full_text') or article.get('abstract', '')
        sections = self._extract_sections(source_text)
        chunks = []
        chunk_index = 0

        for section in sections:
            sentences = sent_tokenize(section['text'])
            current_chunk = []
            current_word_count = 0

            for sentence in sentences:
                sentence_words = sentence.split()
                if current_word_count + len(sentence_words) > self.chunk_size and current_chunk:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'pmid': article.get('pmid'),
                        'title': article.get('title', ''),
                        'section': section['section'],
                        'chunk_index': chunk_index
                    })
                    chunk_index += 1
                    current_chunk = [sentence]
                    current_word_count = len(sentence_words)
                else:
                    current_chunk.append(sentence)
                    current_word_count += len(sentence_words)

            if current_chunk:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'pmid': article.get('pmid'),
                    'title': article.get('title', ''),
                    'section': section['section'],
                    'chunk_index': chunk_index
                })
                chunk_index += 1

        return chunks

    def chunk_article(self, article, strategy='fixed'):
        """
        Chunk an article using the specified strategy.

        Args:
            article (dict): Article dictionary
            strategy (str): 'fixed' or 'semantic'

        Returns:
            list: List of chunks
        """
        if strategy == 'fixed':
            return self.fixed_size_chunking(article)
        elif strategy == 'semantic':
            return self.semantic_chunking(article)
        else:
            raise ValueError("Strategy must be 'fixed' or 'semantic'")

    def chunk_articles(self, articles, strategy='fixed'):
        """
        Chunk a list of articles.
        """
        all_chunks = []
        for article in articles:
            if not isinstance(article, dict):
                article = article.__dict__
            all_chunks.extend(self.chunk_article(article, strategy))
        return all_chunks

# Example usage
if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(level=logging.INFO)
    configs = DataIngestionConfig(config_path=None)
    saved_data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        configs.raw_ingested_data_file
    )

    chunker = Chunker(chunk_size=100, overlap=20)
    saved_pmids_file = saved_data_path

    # If the saved file already contains articles, extract PMIDs and re-fetch details.
    articles = chunker.fetch_articles_for_saved_pmids(saved_pmids_file)

    print(f"Fetched {len(articles)} articles for chunking.")
    print("\nSample chunk output:")
    semantic_chunks = chunker.chunk_articles(articles[:1], strategy='semantic')
    for chunk in semantic_chunks[:5]:
        print(f"[{chunk['section']} | {chunk['chunk_index']}] {chunk['text'][:120]}...")
