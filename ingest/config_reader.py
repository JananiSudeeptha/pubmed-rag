"""
config_reader.py
Reads data ingestion configuration from YAML file.
"""
import os
import yaml
from typing import Any, Dict

class DataIngestionConfig:
    """
    Class to handle reading and accessing data ingestion configuration.
    """
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "ingest",
                "configs.yaml"
            )
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file '{self.config_path}' not found.")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @property
    def eutils_base_url(self) -> str:
        return self.config.get("eutils_base_url", "")

    @property
    def medical_topics(self) -> list:
        return self.config.get("medical_topics", [])

    @property
    def retmax_per_topic(self) -> int:
        return self.config.get("retmax_per_topic", 10)

    @property
    def raw_ingested_data_file(self) -> str:
        return self.config.get("raw_ingested_data_file", "pubmed_articles_data.json")