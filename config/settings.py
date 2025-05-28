"""
Configuration settings for the AI Research Agent
Real-time validation and cross-checking configuration
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
import validators

# Load environment variables
load_dotenv()

class APIConfiguration(BaseModel):
    """API configuration with validation"""
    
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_cse_id: str = Field(default_factory=lambda: os.getenv("GOOGLE_CSE_ID", ""))
    semantic_scholar_api_key: str = Field(default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY", ""))
    crossref_email: str = Field(default_factory=lambda: os.getenv("CROSSREF_EMAIL", ""))
    ncbi_api_key: str = Field(default_factory=lambda: os.getenv("NCBI_API_KEY", ""))
    ncbi_email: str = Field(default_factory=lambda: os.getenv("NCBI_EMAIL", ""))
    scopus_api_key: str = Field(default_factory=lambda: os.getenv("SCOPUS_API_KEY", ""))
    wos_api_key: str = Field(default_factory=lambda: os.getenv("WOS_API_KEY", ""))
    
    @validator('crossref_email', 'ncbi_email')
    def validate_email(cls, v):
        if v and not validators.email(v):
            raise ValueError('Invalid email format')
        return v

class SearchConfiguration(BaseModel):
    """Search configuration with validation"""
    
    max_search_results: int = Field(default=100, ge=1, le=500)
    max_papers_per_query: int = Field(default=50, ge=1, le=200)
    search_timeout: int = Field(default=30, ge=5, le=120)
    verification_timeout: int = Field(default=10, ge=3, le=60)
    max_concurrent_requests: int = Field(default=5, ge=1, le=20)
    rate_limit_per_minute: int = Field(default=60, ge=10, le=300)
    
    # Preferred academic sources for quality control
    preferred_sources: List[str] = Field(default=[
        "arxiv.org",
        "scholar.google.com", 
        "pubmed.ncbi.nlm.nih.gov",
        "ieee.org",
        "acm.org",
        "springer.com",
        "elsevier.com",
        "nature.com",
        "science.org",
        "wiley.com",
        "cambridge.org",
        "oxford.org"
    ])
    
    # Blacklisted sources (predatory journals, unreliable sources)
    blacklisted_sources: List[str] = Field(default=[
        "scirp.org",
        "hindawi.com",
        "frontiersin.org",
        "mdpi.com",
        "omicsonline.org",
        "researchgate.net"  # Only for filtering, not completely blocked
    ])

class QualityConfiguration(BaseModel):
    """Quality control configuration"""
    
    min_citation_count: int = Field(default=3, ge=0)
    min_publication_year: int = Field(default=2015, ge=2000, le=2025)
    max_publication_year: int = Field(default=2025, ge=2020, le=2030)
    require_peer_review: bool = Field(default=True)
    require_doi: bool = Field(default=False)  # Many good papers don't have DOI
    check_author_affiliations: bool = Field(default=True)
    verify_journal_quality: bool = Field(default=True)
    
    # Journal quality thresholds
    min_impact_factor: float = Field(default=0.0, ge=0.0)
    prefer_quartile_journals: List[str] = Field(default=["Q1", "Q2"])

class VerificationConfiguration(BaseModel):
    """Reference verification configuration"""
    
    verify_doi: bool = Field(default=True)
    verify_authors: bool = Field(default=True)
    verify_publication: bool = Field(default=True)
    verify_citations: bool = Field(default=True)
    check_paper_existence: bool = Field(default=True)
    check_pdf_availability: bool = Field(default=True)
    validate_abstracts: bool = Field(default=True)
    cross_reference_multiple_sources: bool = Field(default=True)
    
    # Verification thresholds
    author_match_threshold: float = Field(default=0.8, ge=0.5, le=1.0)
    title_match_threshold: float = Field(default=0.9, ge=0.7, le=1.0)
    date_tolerance_days: int = Field(default=30, ge=0, le=365)

class ModelConfiguration(BaseModel):
    """AI model configuration"""
    
    model_name: str = Field(default="gpt-4-turbo-preview")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4000, ge=100, le=8000)
    top_p: float = Field(default=1.0, ge=0.1, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0)
    
    # Fallback models
    fallback_models: List[str] = Field(default=[
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-3.5-turbo-16k"
    ])

class Settings(BaseModel):
    """Main settings class combining all configurations"""
    
    # Sub-configurations
    api: APIConfiguration = Field(default_factory=APIConfiguration)
    search: SearchConfiguration = Field(default_factory=SearchConfiguration)
    quality: QualityConfiguration = Field(default_factory=QualityConfiguration)
    verification: VerificationConfiguration = Field(default_factory=VerificationConfiguration)
    model: ModelConfiguration = Field(default_factory=ModelConfiguration)
    
    # File system configuration
    cache_dir: str = Field(default="data/cache")
    papers_dir: str = Field(default="data/papers")
    references_dir: str = Field(default="data/references")
    logs_dir: str = Field(default="logs")
    
    # Database configuration
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///ai_research_agent.db"))
    
    # Logging configuration
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = Field(default_factory=lambda: os.getenv("LOG_FILE", "logs/research_agent.log"))
    
    # Cache configuration
    enable_cache: bool = Field(default_factory=lambda: os.getenv("ENABLE_CACHE", "true").lower() == "true")
    cache_expiry_hours: int = Field(default_factory=lambda: int(os.getenv("CACHE_EXPIRY_HOURS", "24")))
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()
        self._setup_logging()
    
    def _create_directories(self):
        """Create necessary directories"""
        import os
        directories = [
            self.cache_dir,
            self.papers_dir,
            self.references_dir,
            self.logs_dir,
            "data/cache",
            "data/papers",
            "data/references"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        # Create .gitkeep files
        for directory in directories:
            gitkeep_path = os.path.join(directory, ".gitkeep")
            if not os.path.exists(gitkeep_path):
                with open(gitkeep_path, 'w') as f:
                    f.write("")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        import logging
        import colorlog
        
        # Create logs directory
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler with colors
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level.upper(), logging.INFO))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate API keys and return status"""
        validation_results = {}
        
        # Required APIs
        required_apis = {
            'openai': self.api.openai_api_key,
            'google_search': self.api.google_api_key and self.api.google_cse_id,
        }
        
        # Optional APIs
        optional_apis = {
            'semantic_scholar': self.api.semantic_scholar_api_key,
            'crossref': self.api.crossref_email,
            'ncbi': self.api.ncbi_api_key and self.api.ncbi_email,
            'scopus': self.api.scopus_api_key,
            'wos': self.api.wos_api_key
        }
        
        # Check required APIs
        for api_name, api_key in required_apis.items():
            validation_results[api_name] = bool(api_key and api_key.strip())
        
        # Check optional APIs
        for api_name, api_key in optional_apis.items():
            validation_results[f"{api_name}_optional"] = bool(api_key and api_key.strip())
        
        return validation_results
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration dictionary"""
        return {
            "max_results": self.search.max_search_results,
            "timeout": self.search.search_timeout,
            "preferred_sources": self.search.preferred_sources,
            "blacklisted_sources": self.search.blacklisted_sources,
            "rate_limit": self.search.rate_limit_per_minute,
            "concurrent_requests": self.search.max_concurrent_requests
        }
    
    def get_verification_config(self) -> Dict[str, Any]:
        """Get verification configuration dictionary"""
        return {
            "verify_doi": self.verification.verify_doi,
            "verify_authors": self.verification.verify_authors,
            "verify_publication": self.verification.verify_publication,
            "verify_citations": self.verification.verify_citations,
            "check_existence": self.verification.check_paper_existence,
            "check_pdf": self.verification.check_pdf_availability,
            "validate_abstracts": self.verification.validate_abstracts,
            "cross_reference": self.verification.cross_reference_multiple_sources,
            "timeout": self.verification.verification_timeout,
            "author_threshold": self.verification.author_match_threshold,
            "title_threshold": self.verification.title_match_threshold,
            "date_tolerance": self.verification.date_tolerance_days
        }
    
    def get_quality_config(self) -> Dict[str, Any]:
        """Get quality control configuration dictionary"""
        return {
            "min_citations": self.quality.min_citation_count,
            "min_year": self.quality.min_publication_year,
            "max_year": self.quality.max_publication_year,
            "require_peer_review": self.quality.require_peer_review,
            "require_doi": self.quality.require_doi,
            "check_affiliations": self.quality.check_author_affiliations,
            "verify_journal": self.quality.verify_journal_quality,
            "min_impact_factor": self.quality.min_impact_factor,
            "prefer_quartiles": self.quality.prefer_quartile_journals
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get AI model configuration dictionary"""
        return {
            "model": self.model.model_name,
            "temperature": self.model.temperature,
            "max_tokens": self.model.max_tokens,
            "top_p": self.model.top_p,
            "frequency_penalty": self.model.frequency_penalty,
            "presence_penalty": self.model.presence_penalty,
            "fallback_models": self.model.fallback_models
        }
    
    def is_production_ready(self) -> bool:
        """Check if configuration is ready for production use"""
        validation_results = self.validate_api_keys()
        
        # Check if required APIs are configured
        required_keys = ['openai', 'google_search']
        return all(validation_results.get(key, False) for key in required_keys)
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of all configurations"""
        api_validation = self.validate_api_keys()
        
        return {
            "api_status": api_validation,
            "production_ready": self.is_production_ready(),
            "search_config": self.get_search_config(),
            "verification_config": self.get_verification_config(),
            "quality_config": self.get_quality_config(),
            "model_config": self.get_model_config(),
            "directories": {
                "cache": self.cache_dir,
                "papers": self.papers_dir,
                "references": self.references_dir,
                "logs": self.logs_dir
            },
            "database": self.database_url,
            "cache_enabled": self.enable_cache,
            "cache_expiry": self.cache_expiry_hours,
            "log_level": self.log_level
        }

# Global settings instance
settings = Settings()