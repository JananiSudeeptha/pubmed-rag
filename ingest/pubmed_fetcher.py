import time
import requests
from config_reader import DataIngestionConfig

class PubMedFetcher:
    
    def __init__(self, config_path=None):
        self.config = DataIngestionConfig(config_path)
        self.eutils_base_url = self.config.eutils_base_url
        self.medical_topics = self.config.medical_topics
        self.retmax_per_topic = self.config.retmax_per_topic
        # self.api_key = getattr(self.config, 'api_key', None)

    def fetch_pmids(self):
        pmids = []
        for topic in self.medical_topics:
            print(f"Searching for articles on: '{topic}'")
            search_url = (
                f"{self.eutils_base_url}esearch.fcgi?db=pubmed&term={topic.replace(' ', '+')}&retmax={self.retmax_per_topic}&retmode=json"
            )
            # if self.api_key: search_url += f"&api_key={self.api_key}"

            try:
                response = requests.get(search_url)
                response.raise_for_status()  # Raise an exception for HTTP errors
                data = response.json()
                id_list = data.get('esearchresult', {}).get('idlist', [])
                pmids.extend(id_list)
                print(f"  Found {len(id_list)} PMIDs for '{topic}'. Total PMIDs collected: {len(pmids)}")
                time.sleep(0.5)  # Be respectful to the API; wait for 0.5 seconds between requests
            except requests.exceptions.RequestException as e:
                print(f"Error during esearch for '{topic}': {e}")

        total_pmids_before_dedup = len(pmids)
        pmids = list(set(pmids))
        print(f"\nTotal PMIDs collected after deduplication: {len(pmids)}")
        print(f"{total_pmids_before_dedup - len(pmids)} PMIDs were duplicates and removed.")
        return pmids

if __name__ == "__main__":
    fetcher = PubMedFetcher()
    fetcher.fetch_pmids()
