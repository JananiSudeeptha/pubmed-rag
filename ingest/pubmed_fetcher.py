import os
import sys
import time
import logging
import xml.etree.ElementTree as ET
import requests
import json
import dataclasses
from dataclasses import dataclass
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.config_reader import DataIngestionConfig

@dataclass
class ArticleData:
    pmid: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    full_text: Optional[str] = None
    pmc_id: Optional[str] = None
    authors: Optional[List[str]] = None
    publication_date: Optional[str] = None
    mesh_terms: Optional[List[str]] = None

    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.mesh_terms is None:
            self.mesh_terms = []

class PubMedFetcher:
    
    def __init__(self, config_path=None):
        self.config = DataIngestionConfig(config_path)
        self.eutils_base_url = self.config.eutils_base_url
        self.medical_topics = self.config.medical_topics
        self.retmax_per_topic = self.config.retmax_per_topic
        self.batch_size = 10  # Configurable batch size for fetching
        self.logger = logging.getLogger(__name__)
        # self.api_key = getattr(self.config, 'api_key', None)

    def fetch_pmids(self) -> List[str]:
        pmids = []
        for topic in self.medical_topics:
            self.logger.info(f"Searching for articles on: '{topic}'")
            search_url = (
                f"{self.eutils_base_url}esearch.fcgi?db=pubmed&term={topic.replace(' ', '+')}&retmax={self.retmax_per_topic}&retmode=json"
            )
            self.logger.info(f"  Searching with URL: {search_url}")
            # if self.api_key: search_url += f"&api_key={self.api_key}"

            try:
                response = requests.get(search_url)
                response.raise_for_status()  # Raise an exception for HTTP errors
                data = response.json()
                id_list = data.get('esearchresult', {}).get('idlist', [])
                pmids.extend(id_list)
                self.logger.info(f"  Found {len(id_list)} PMIDs for '{topic}'. Total PMIDs collected: {len(pmids)}")
                time.sleep(0.5)  # Be respectful to the API; wait for 0.5 seconds between requests
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error during esearch for '{topic}': {e}")

        total_pmids_before_dedup = len(pmids)
        pmids = list(set(pmids))
        self.logger.info(f"\nTotal PMIDs collected after deduplication: {len(pmids)}")
        self.logger.info(f"{total_pmids_before_dedup - len(pmids)} PMIDs were duplicates and removed.")
        return pmids

    def _find_element_by_localname(self, root: ET.Element, local_name: str) -> Optional[ET.Element]:
        for elem in root.iter():
            if elem.tag.endswith(local_name):
                return elem
        return None

    def parse_pmc_article_xml(self, xml_string: str) -> Optional[str]:
        try:
            root = ET.fromstring(xml_string)
            body = self._find_element_by_localname(root, 'body')
            if body is None:
                return None

            paragraphs = []
            def walk(node):
                tag = node.tag.lower()
                if tag.endswith('title') and node.text:
                    paragraphs.append(node.text.strip())
                elif tag.endswith('p') and node.text:
                    paragraphs.append(node.text.strip())
                for child in node:
                    walk(child)

            walk(body)
            return '\n'.join([p for p in paragraphs if p]) if paragraphs else None
        except ET.ParseError as e:
            self.logger.error(f"PMC XML parsing error: {e}")
            return None

    def fetch_pmc_full_text(self, pmc_id: str) -> Optional[str]:
        efetch_url = f"{self.eutils_base_url}efetch.fcgi?db=pmc&id={pmc_id}&retmode=xml"
        try:
            response = requests.get(efetch_url)
            response.raise_for_status()
            return self.parse_pmc_article_xml(response.text)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching PMC full text for {pmc_id}: {e}")
            return None

    def parse_pubmed_article_xml(self, xml_string: str) -> Optional[ArticleData]:
        """Parses a single PubMed article XML string and extracts relevant information."""
        try:
            root = ET.fromstring(xml_string)
            article_data = {}

            # PMID
            pmid_element = root.find('.//PMID')
            if pmid_element is not None:
                article_data['pmid'] = pmid_element.text

            # Title
            article_title_element = root.find('.//ArticleTitle')
            if article_title_element is not None:
                article_data['title'] = article_title_element.text

            # Abstract
            abstract_text_elements = root.findall('.//AbstractText')
            if abstract_text_elements:
                abstract_parts = []
                for elem in abstract_text_elements:
                    text = elem.text.strip() if elem.text else ''
                    label = elem.attrib.get('Label') or elem.attrib.get('NlmCategory')
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    elif text:
                        abstract_parts.append(text)
                article_data['abstract'] = '\n'.join(abstract_parts) if abstract_parts else None
            else:
                article_data['abstract'] = None

            # PMC ID
            pmc_id = None
            for article_id in root.findall('.//ArticleId'):
                if article_id.attrib.get('IdType', '').lower() == 'pmc' and article_id.text:
                    pmc_id = article_id.text.strip()
                    break
            article_data['pmc_id'] = pmc_id
            article_data['full_text'] = None

            # Authors
            authors = []
            for author_elem in root.findall('.//AuthorList/Author'):
                last_name = author_elem.find('LastName')
                fore_name = author_elem.find('ForeName')
                if last_name is not None and fore_name is not None:
                    authors.append(f"{fore_name.text} {last_name.text}")
            article_data['authors'] = authors

            # Publication Date
            pub_date_elem = root.find('.//PubDate')
            if pub_date_elem is not None:
                year = pub_date_elem.find('Year')
                month = pub_date_elem.find('Month')
                day = pub_date_elem.find('Day')
                date_parts = []
                if year is not None and year.text:
                    date_parts.append(year.text)
                if month is not None and month.text:
                    date_parts.append(month.text)
                if day is not None and day.text:
                    date_parts.append(day.text)
                pub_date = '-'.join(date_parts) if date_parts else None
                article_data['publication_date'] = pub_date

            # MeSH Terms
            mesh_terms = []
            for mesh_heading_elem in root.findall('.//MeshHeadingList/MeshHeading/DescriptorName'):
                if mesh_heading_elem is not None and mesh_heading_elem.text:
                    mesh_terms.append(mesh_heading_elem.text)
            article_data['mesh_terms'] = mesh_terms

            return ArticleData(**article_data)
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error: {e}")
            return None

    def fetch_article_details(self, pmids: List[str]) -> List[ArticleData]:
        """Fetches detailed information for a list of PMIDs in batches."""
        all_articles_data = []
        self.logger.info(f"Fetching details for {len(pmids)} articles...")
        for i in range(0, len(pmids), self.batch_size):
            batch_pmids = pmids[i:i + self.batch_size]
            pmid_list_str = ",".join(batch_pmids)

            efetch_url = (
                f"{self.eutils_base_url}efetch.fcgi?db=pubmed&id={pmid_list_str}&retmode=xml"
            )
            self.logger.info(f"  Fetching from url: {efetch_url}")
            # if self.api_key: efetch_url += f"&api_key={self.api_key}"

            try:
                response = requests.get(efetch_url)
                response.raise_for_status()  # Raise an exception for HTTP errors

                # PubMed returns a list of articles under <PubmedArticleSet>
                root = ET.fromstring(response.text)
                for article_elem in root.findall('.//PubmedArticle'):
                    # Convert each article element back to a string for parsing by the helper function
                    article_xml_string = ET.tostring(article_elem, encoding='unicode')
                    parsed_data = self.parse_pubmed_article_xml(article_xml_string)
                    if parsed_data:
                        if parsed_data.pmc_id:
                            full_text = self.fetch_pmc_full_text(parsed_data.pmc_id)
                            parsed_data.full_text = full_text or parsed_data.abstract
                        else:
                            parsed_data.full_text = parsed_data.abstract
                        all_articles_data.append(parsed_data)

                self.logger.info(f"  Fetched details for {len(batch_pmids)} articles in batch {i//self.batch_size + 1}. Total articles processed: {len(all_articles_data)}")
                time.sleep(1.0)  # Wait for 1 second between batch requests

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error during efetch for PMIDs {pmid_list_str}: {e}")
            except ET.ParseError as e:
                self.logger.error(f"XML parsing error for PMIDs {pmid_list_str}: {e}")

        self.logger.info(f"\nSuccessfully ingested details for {len(all_articles_data)} articles.")
        return all_articles_data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetcher = PubMedFetcher()
    pmids = fetcher.fetch_pmids()
    articles = fetcher.fetch_article_details(pmids)
    
    # Convert to dicts for JSON serialization
    articles_dicts = [dataclasses.asdict(article) for article in articles]
    
    output_filename = fetcher.config.raw_ingested_data_file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(articles_dicts, f, ensure_ascii=False, indent=4)
    
    logging.info(f"Ingested data saved to '{output_filename}'.")
    
    # Display the first 2 articles to verify
    if articles:
        fetcher.logger.info("\nFirst two ingested articles:")
        for i, article in enumerate(articles[:2]):
            fetcher.logger.info(f"\nArticle {i+1}:")
            article_dict = dataclasses.asdict(article)
            for key, value in article_dict.items():
                fetcher.logger.info(f"  {key}: {value}")
    else:
        fetcher.logger.info("No articles were ingested or parsed.")
