"""
web_scraper.py - Comprehensive Medical Data Collection System
Gathers research papers, clinical trials, and statistics to improve heart disease prediction model
"""

import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time
import json
import re
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, quote
import logging
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDataScraper:
    """Comprehensive medical data scraper for heart disease research"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Rate limiting
        self.delay = 1.0  # seconds between requests
        
        # Data storage
        self.scraped_data = {
            'research_papers': [],
            'clinical_trials': [],
            'medical_statistics': [],
            'risk_factors': [],
            'biomarkers': []
        }
    
    def scrape_pubmed_papers(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Scrape research papers from PubMed
        
        Args:
            query: Search query for heart disease research
            max_results: Maximum number of papers to retrieve
            
        Returns:
            List of dictionaries containing paper information
        """
        logger.info(f"Scraping PubMed papers for query: {query}")
        
        papers = []
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        try:
            # Search for papers
            search_url = f"{base_url}/esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmode': 'json',
                'retmax': max_results
            }
            
            response = self.session.get(search_url, params=search_params)
            response.raise_for_status()
            search_data = response.json()
            
            paper_ids = search_data.get('esearchresult', {}).get('idlist', [])
            logger.info(f"Found {len(paper_ids)} papers")
            
            # Fetch paper details
            for i, paper_id in enumerate(paper_ids):
                try:
                    summary_url = f"{base_url}/esummary.fcgi"
                    summary_params = {
                        'db': 'pubmed',
                        'id': paper_id,
                        'retmode': 'json'
                    }
                    
                    summary_response = self.session.get(summary_url, params=summary_params)
                    summary_response.raise_for_status()
                    summary_data = summary_response.json()
                    
                    # Extract paper information
                    paper_info = summary_data.get('result', {}).get(paper_id, {})
                    
                    paper = {
                        'pubmed_id': paper_id,
                        'title': paper_info.get('title', ''),
                        'authors': paper_info.get('authors', []),
                        'journal': paper_info.get('fulljournalname', ''),
                        'publication_date': paper_info.get('pubdate', ''),
                        'abstract': paper_info.get('abstract', ''),
                        'doi': paper_info.get('elocationid', ''),
                        'keywords': self._extract_keywords(paper_info),
                        'scraped_date': datetime.now().isoformat()
                    }
                    
                    papers.append(paper)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(paper_ids)} papers")
                    
                    time.sleep(self.delay)
                    
                except Exception as e:
                    logger.warning(f"Error processing paper {paper_id}: {e}")
                    continue
            
            self.scraped_data['research_papers'].extend(papers)
            logger.info(f"Successfully scraped {len(papers)} papers")
            
        except Exception as e:
            logger.error(f"Error scraping PubMed: {e}")
        
        return papers
    
    def scrape_clinical_trials(self, condition: str = "heart disease", max_results: int = 50) -> List[Dict]:
        """
        Scrape clinical trial data from ClinicalTrials.gov
        
        Args:
            condition: Medical condition to search for
            max_results: Maximum number of trials to retrieve
            
        Returns:
            List of dictionaries containing trial information
        """
        logger.info(f"Scraping clinical trials for: {condition}")
        
        trials = []
        base_url = "https://clinicaltrials.gov/api/query/full_studies"
        
        try:
            # Search for trials with updated API parameters
            params = {
                'expr': condition,
                'format': 'json',
                'min_rnk': '1',
                'max_rnk': str(max_results)
            }
            
            response = self.session.get(base_url, params=params)
            
            # Try alternative URL if first one fails
            if response.status_code != 200:
                alt_url = "https://clinicaltrials.gov/api/query/study_fields"
                alt_params = {
                    'expr': condition,
                    'fields': 'NCTId,OfficialTitle,Condition,OverallStatus,Phase,EnrollmentCount,StudyType,StartDateStruct,CompletionDateStruct',
                    'min_rnk': '1',
                    'max_rnk': str(max_results),
                    'fmt': 'json'
                }
                response = self.session.get(alt_url, params=alt_params)
            
            response.raise_for_status()
            data = response.json()
            
            # Handle different response formats
            if 'FullStudiesResponse' in data:
                studies = data.get('FullStudiesResponse', {}).get('FullStudies', [])
            elif 'StudyFieldsResponse' in data:
                studies = data.get('StudyFieldsResponse', {}).get('StudyFields', [])
            else:
                studies = []
            
            logger.info(f"Found {len(studies)} trials")
            
            for i, study in enumerate(studies):
                try:
                    # Handle different study formats
                    if 'Study' in study:
                        study_info = study.get('Study', {})
                        protocol_section = study_info.get('ProtocolSection', {})
                        results_section = study_info.get('ResultsSection', {})
                        
                        # Extract trial information
                        trial = {
                            'nct_id': protocol_section.get('IdentificationModule', {}).get('NCTId', ''),
                            'title': protocol_section.get('IdentificationModule', {}).get('OfficialTitle', ''),
                            'condition': protocol_section.get('ConditionsModule', {}).get('ConditionsList', {}).get('Condition', []),
                            'status': protocol_section.get('StatusModule', {}).get('OverallStatus', ''),
                            'phase': protocol_section.get('DesignModule', {}).get('PhaseList', {}).get('Phase', []),
                            'enrollment': protocol_section.get('DesignModule', {}).get('EnrollmentInfo', {}).get('EnrollmentCount', ''),
                            'study_type': protocol_section.get('DesignModule', {}).get('StudyType', ''),
                            'start_date': protocol_section.get('StatusModule', {}).get('StartDateStruct', {}).get('StartDate', ''),
                            'completion_date': protocol_section.get('StatusModule', {}).get('CompletionDateStruct', {}).get('CompletionDate', ''),
                            'interventions': self._extract_interventions(protocol_section),
                            'eligibility': self._extract_eligibility(protocol_section),
                            'outcomes': self._extract_outcomes(results_section),
                            'scraped_date': datetime.now().isoformat()
                        }
                    else:
                        # Handle StudyFieldsResponse format
                        trial = {
                            'nct_id': study.get('NCTId', [''])[0] if study.get('NCTId') else '',
                            'title': study.get('OfficialTitle', [''])[0] if study.get('OfficialTitle') else '',
                            'condition': study.get('Condition', []),
                            'status': study.get('OverallStatus', [''])[0] if study.get('OverallStatus') else '',
                            'phase': study.get('Phase', []),
                            'enrollment': study.get('EnrollmentCount', [''])[0] if study.get('EnrollmentCount') else '',
                            'study_type': study.get('StudyType', [''])[0] if study.get('StudyType') else '',
                            'scraped_date': datetime.now().isoformat()
                        }
                    
                    trials.append(trial)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(studies)} trials")
                    
                    time.sleep(self.delay)
                    
                except Exception as e:
                    logger.warning(f"Error processing trial: {e}")
                    continue
            
            self.scraped_data['clinical_trials'].extend(trials)
            logger.info(f"Successfully scraped {len(trials)} trials")
            
        except Exception as e:
            logger.error(f"Error scraping ClinicalTrials.gov: {e}")
        
        return trials
    
    def scrape_medical_statistics(self) -> List[Dict]:
        """
        Scrape heart disease statistics from WHO and CDC
        
        Returns:
            List of dictionaries containing statistical information
        """
        logger.info("Scraping medical statistics from WHO and CDC")
        
        stats = []
        
        # WHO statistics
        who_stats = self._scrape_who_stats()
        stats.extend(who_stats)
        
        # CDC statistics
        cdc_stats = self._scrape_cdc_stats()
        stats.extend(cdc_stats)
        
        # American Heart Association statistics
        aha_stats = self._scrape_aha_stats()
        stats.extend(aha_stats)
        
        self.scraped_data['medical_statistics'].extend(stats)
        logger.info(f"Successfully scraped {len(stats)} statistics")
        
        return stats
    
    def _scrape_who_stats(self) -> List[Dict]:
        """Scrape WHO cardiovascular disease statistics"""
        stats = []
        
        try:
            url = "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract key statistics
            fact_sheet = {
                'source': 'WHO',
                'url': url,
                'title': 'Cardiovascular Diseases Fact Sheet',
                'scraped_date': datetime.now().isoformat(),
                'key_facts': []
            }
            
            # Find fact list
            fact_list = soup.find('ul', class_='sf-list')
            if fact_list:
                for li in fact_list.find_all('li'):
                    fact_text = li.get_text(strip=True)
                    if any(keyword in fact_text.lower() for keyword in ['deaths', 'million', 'percent', 'risk']):
                        fact_sheet['key_facts'].append(fact_text)
            
            stats.append(fact_sheet)
            
        except Exception as e:
            logger.warning(f"Error scraping WHO stats: {e}")
        
        return stats
    
    def _scrape_cdc_stats(self) -> List[Dict]:
        """Scrape CDC heart disease statistics"""
        stats = []
        
        try:
            url = "https://www.cdc.gov/heartdisease/about.htm"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            fact_sheet = {
                'source': 'CDC',
                'url': url,
                'title': 'Heart Disease About',
                'scraped_date': datetime.now().isoformat(),
                'key_facts': []
            }
            
            # Extract statistics from various elements
            for element in soup.find_all(['p', 'li', 'h2', 'div']):
                text = element.get_text(strip=True)
                if any(keyword in text.lower() for keyword in ['deaths', 'percent', 'adults', 'americans']):
                    if len(text) > 20 and len(text) < 200:
                        fact_sheet['key_facts'].append(text)
            
            stats.append(fact_sheet)
            
        except Exception as e:
            logger.warning(f"Error scraping CDC stats: {e}")
        
        return stats
    
    def _scrape_aha_stats(self) -> List[Dict]:
        """Scrape American Heart Association statistics"""
        stats = []
        
        try:
            url = "https://www.heart.org/en/about-us/heart-and-stroke-association-statistics"
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            fact_sheet = {
                'source': 'American Heart Association',
                'url': url,
                'title': 'Heart and Stroke Statistics',
                'scraped_date': datetime.now().isoformat(),
                'key_facts': []
            }
            
            # Extract statistics
            for element in soup.find_all(['p', 'li', 'div']):
                text = element.get_text(strip=True)
                if any(keyword in text.lower() for keyword in ['deaths', 'percent', 'million', 'adults']):
                    if len(text) > 20 and len(text) < 300:
                        fact_sheet['key_facts'].append(text)
            
            stats.append(fact_sheet)
            
        except Exception as e:
            logger.warning(f"Error scraping AHA stats: {e}")
        
        return stats
    
    def _extract_keywords(self, paper_info: Dict) -> List[str]:
        """Extract keywords from paper information"""
        keywords = []
        
        # Try to extract from various fields
        title = paper_info.get('title', '').lower()
        abstract = paper_info.get('abstract', '').lower()
        
        # Common heart disease keywords
        heart_keywords = [
            'heart disease', 'cardiovascular', 'coronary', 'myocardial',
            'infarction', 'angina', 'arrhythmia', 'hypertension',
            'cholesterol', 'blood pressure', 'risk factors', 'biomarkers'
        ]
        
        text = f"{title} {abstract}"
        for keyword in heart_keywords:
            if keyword in text:
                keywords.append(keyword)
        
        return keywords
    
    def _extract_interventions(self, protocol_section: Dict) -> List[str]:
        """Extract interventions from trial protocol"""
        interventions = []
        
        try:
            interventions_module = protocol_section.get('ArmsInterventionsModule', {})
            arms_list = interventions_module.get('ArmList', {}).get('Arm', [])
            
            for arm in arms_list:
                intervention_list = arm.get('InterventionList', {}).get('Intervention', [])
                for intervention in intervention_list:
                    intervention_type = intervention.get('InterventionType', '')
                    intervention_name = intervention.get('InterventionName', '')
                    if intervention_type and intervention_name:
                        interventions.append(f"{intervention_type}: {intervention_name}")
        
        except Exception as e:
            logger.warning(f"Error extracting interventions: {e}")
        
        return interventions
    
    def _extract_eligibility(self, protocol_section: Dict) -> Dict:
        """Extract eligibility criteria"""
        eligibility = {}
        
        try:
            eligibility_module = protocol_section.get('EligibilityModule', {})
            
            eligibility['study_pop'] = eligibility_module.get('StudyPopulation', '')
            eligibility['sampling_method'] = eligibility_module.get('SamplingMethod', '')
            eligibility['criteria'] = eligibility_module.get('EligibilityCriteria', '')
            eligibility['gender'] = eligibility_module.get('Sex', '')
            eligibility['min_age'] = eligibility_module.get('MinimumAge', '')
            eligibility['max_age'] = eligibility_module.get('MaximumAge', '')
            eligibility['healthy_volunteers'] = eligibility_module.get('HealthyVolunteers', '')
        
        except Exception as e:
            logger.warning(f"Error extracting eligibility: {e}")
        
        return eligibility
    
    def _extract_outcomes(self, results_section: Dict) -> List[Dict]:
        """Extract outcome measures"""
        outcomes = []
        
        try:
            outcomes_module = results_section.get('OutcomeMeasuresModule', {})
            outcome_list = outcomes_module.get('OutcomeList', {}).get('Outcome', [])
            
            for outcome in outcome_list:
                outcome_info = {
                    'type': outcome.get('OutcomeType', ''),
                    'title': outcome.get('OutcomeTitle', ''),
                    'description': outcome.get('OutcomeDescription', ''),
                    'time_frame': outcome.get('OutcomeTimeFrame', '')
                }
                outcomes.append(outcome_info)
        
        except Exception as e:
            logger.warning(f"Error extracting outcomes: {e}")
        
        return outcomes
    
    def save_data(self, filename_prefix: str = None):
        """Save scraped data to files"""
        if filename_prefix is None:
            filename_prefix = f"medical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save each category to separate files
        for category, data in self.scraped_data.items():
            if data:
                filename = f"{filename_prefix}_{category}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved {len(data)} {category} to {filename}")
        
        # Save combined data
        combined_filename = f"{filename_prefix}_combined.json"
        with open(combined_filename, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved combined data to {combined_filename}")
    
    def generate_summary_report(self) -> Dict:
        """Generate summary report of scraped data"""
        report = {
            'scraping_date': datetime.now().isoformat(),
            'summary': {},
            'data_quality': {},
            'recommendations': []
        }
        
        for category, data in self.scraped_data.items():
            report['summary'][category] = len(data)
            
            if category == 'research_papers' and data:
                # Analyze research trends
                years = []
                for paper in data:
                    pub_date = paper.get('publication_date', '')
                    if pub_date:
                        year_match = re.search(r'(\d{4})', pub_date)
                        if year_match:
                            years.append(int(year_match.group(1)))
                
                if years:
                    report['data_quality']['publication_year_range'] = f"{min(years)}-{max(years)}"
                    report['data_quality']['avg_publication_year'] = sum(years) / len(years)
            
            elif category == 'clinical_trials' and data:
                # Analyze trial phases
                phases = {}
                for trial in data:
                    phase_list = trial.get('phase', [])
                    for phase in phase_list:
                        phases[phase] = phases.get(phase, 0) + 1
                
                report['data_quality']['trial_phases'] = phases
        
        # Generate recommendations
        if len(self.scraped_data['research_papers']) < 50:
            report['recommendations'].append("Consider expanding PubMed search with more specific keywords")
        
        if len(self.scraped_data['clinical_trials']) < 20:
            report['recommendations'].append("Consider searching for additional clinical trial databases")
        
        return report

# =============================
# USAGE EXAMPLES
# =============================

if __name__ == "__main__":
    print("=" * 60)
    print("🏥 MEDICAL DATA SCRAPER")
    print("=" * 60)
    
    # Initialize scraper
    scraper = MedicalDataScraper()
    
    try:
        # Scrape research papers
        print("\n📚 Scraping Research Papers...")
        papers = scraper.scrape_pubmed_papers(
            query="heart disease prediction machine learning biomarkers",
            max_results=50
        )
        print(f"✅ Scraped {len(papers)} research papers")
        
        # Scrape clinical trials
        print("\n🔬 Scraping Clinical Trials...")
        trials = scraper.scrape_clinical_trials(
            condition="coronary artery disease",
            max_results=30
        )
        print(f"✅ Scraped {len(trials)} clinical trials")
        
        # Scrape medical statistics
        print("\n📊 Scraping Medical Statistics...")
        stats = scraper.scrape_medical_statistics()
        print(f"✅ Scraped {len(stats)} statistical sources")
        
        # Save data
        print("\n💾 Saving Data...")
        scraper.save_data()
        
        # Generate report
        print("\n📋 Generating Summary Report...")
        report = scraper.generate_summary_report()
        
        print("\n📊 SUMMARY:")
        print("-" * 40)
        for category, count in report['summary'].items():
            print(f"{category.replace('_', ' ').title()}: {count}")
        
        if report['recommendations']:
            print("\n💡 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "=" * 60)
        print("✅ Medical data scraping completed!")
        print("Data saved to JSON files for model improvement")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your internet connection and try again")
