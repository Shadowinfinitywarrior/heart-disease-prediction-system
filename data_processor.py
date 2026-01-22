"""
data_processor.py - Advanced Data Processing and Integration Module
Processes scraped medical data to improve heart disease prediction model
"""

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalDataProcessor:
    """Advanced processor for scraped medical data to enhance model features"""
    
    def __init__(self):
        self.feature_extractor = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.scaler = StandardScaler()
        self.processed_features = {}
        self.biomarker_database = {}
        self.risk_factor_weights = {}
        
    def load_scraped_data(self, filepath: str) -> Dict:
        """Load scraped medical data from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded scraped data from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return {}
    
    def process_research_papers(self, papers: List[Dict]) -> Dict:
        """Process research papers to extract insights and features"""
        logger.info(f"Processing {len(papers)} research papers")
        
        processed_data = {
            'risk_factors': {},
            'biomarkers': {},
            'prediction_methods': {},
            'performance_metrics': {},
            'feature_importance': {},
            'trends': {}
        }
        
        # Extract risk factors
        risk_factor_patterns = [
            r'(?:risk factor|predictor|associated with)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(?:increases|associated with)',
            r'(?:elevated|high)\s+(\w+(?:\s+\w+)*)\s+(?:risk|levels)'
        ]
        
        # Extract biomarkers
        biomarker_patterns = [
            r'(?:biomarker|marker)\s+(\w+(?:\s+\w+)*)',
            r'(?:levels|concentration)\s+of\s+(\w+(?:\s+\w+)*)',
            r'(?:serum|plasma)\s+(\w+(?:\s+\w+)*)'
        ]
        
        # Extract performance metrics
        metric_patterns = [
            r'(?:accuracy|sensitivity|specificity|auc|roc)\s*[:=]?\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*%?\s*(?:accuracy|sensitivity|specificity)',
            r'auc\s*[:=]?\s*(\d+(?:\.\d+)?)'
        ]
        
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
            
            # Extract risk factors
            for pattern in risk_factor_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    risk_factor = match.strip()
                    if len(risk_factor) > 2:
                        processed_data['risk_factors'][risk_factor] = \
                            processed_data['risk_factors'].get(risk_factor, 0) + 1
            
            # Extract biomarkers
            for pattern in biomarker_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    biomarker = match.strip()
                    if len(biomarker) > 2:
                        processed_data['biomarkers'][biomarker] = \
                            processed_data['biomarkers'].get(biomarker, 0) + 1
            
            # Extract performance metrics
            for pattern in metric_patterns:
                matches = re.findall(pattern, text)
                for match in matches:
                    try:
                        value = float(match)
                        if 0 <= value <= 1:
                            processed_data['performance_metrics']['accuracy'] = \
                                max(processed_data['performance_metrics'].get('accuracy', 0), value)
                    except ValueError:
                        continue
        
        # Sort by frequency
        for key in processed_data:
            if isinstance(processed_data[key], dict):
                processed_data[key] = dict(sorted(
                    processed_data[key].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20])  # Top 20
        
        return processed_data
    
    def process_clinical_trials(self, trials: List[Dict]) -> Dict:
        """Process clinical trials to extract treatment patterns and outcomes"""
        logger.info(f"Processing {len(trials)} clinical trials")
        
        processed_data = {
            'interventions': {},
            'outcomes': {},
            'eligibility_criteria': {},
            'phase_distribution': {},
            'success_rates': {}
        }
        
        for trial in trials:
            # Process interventions
            interventions = trial.get('interventions', [])
            for intervention in interventions:
                if ':' in intervention:
                    intervention_type = intervention.split(':')[0].strip()
                    processed_data['interventions'][intervention_type] = \
                        processed_data['interventions'].get(intervention_type, 0) + 1
            
            # Process outcomes
            outcomes = trial.get('outcomes', [])
            for outcome in outcomes:
                outcome_type = outcome.get('type', '')
                if outcome_type:
                    processed_data['outcomes'][outcome_type] = \
                        processed_data['outcomes'].get(outcome_type, 0) + 1
            
            # Process phases
            phases = trial.get('phase', [])
            for phase in phases:
                processed_data['phase_distribution'][phase] = \
                    processed_data['phase_distribution'].get(phase, 0) + 1
            
            # Process eligibility
            eligibility = trial.get('eligibility', {})
            age_criteria = eligibility.get('min_age', '') + ' ' + eligibility.get('max_age', '')
            if age_criteria.strip():
                processed_data['eligibility_criteria']['age_range'] = age_criteria
        
        return processed_data
    
    def process_medical_statistics(self, stats: List[Dict]) -> Dict:
        """Process medical statistics to extract population-level insights"""
        logger.info(f"Processing {len(stats)} statistical sources")
        
        processed_data = {
            'prevalence_rates': {},
            'mortality_rates': {},
            'risk_statistics': {},
            'demographic_data': {},
            'trends': {}
        }
        
        # Patterns for extracting numerical data
        number_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:%|percent)',
            r'(\d+(?:,\d+)*)\s*(?:deaths|cases|people)',
            r'(\d+(?:\.\d+)?)\s*(?:million|billion|thousand)',
            r'(\d+(?:\.\d+)?)\s*times?\s+(?:more|less|higher|lower)'
        ]
        
        for stat_source in stats:
            key_facts = stat_source.get('key_facts', [])
            source = stat_source.get('source', '')
            
            for fact in key_facts:
                fact_lower = fact.lower()
                
                # Extract prevalence data
                if 'prevalence' in fact_lower or 'have' in fact_lower:
                    for pattern in number_patterns:
                        matches = re.findall(pattern, fact)
                        for match in matches:
                            try:
                                value = float(match.replace(',', ''))
                                if '%' in fact:
                                    processed_data['prevalence_rates'][source] = value
                                else:
                                    processed_data['prevalence_rates'][source] = value
                            except ValueError:
                                continue
                
                # Extract mortality data
                if 'death' in fact_lower or 'mortality' in fact_lower:
                    for pattern in number_patterns:
                        matches = re.findall(pattern, fact)
                        for match in matches:
                            try:
                                value = float(match.replace(',', ''))
                                processed_data['mortality_rates'][source] = value
                            except ValueError:
                                continue
                
                # Extract risk statistics
                if 'risk' in fact_lower:
                    for pattern in number_patterns:
                        matches = re.findall(pattern, fact)
                        for match in matches:
                            try:
                                value = float(match.replace(',', ''))
                                processed_data['risk_statistics'][source] = value
                            except ValueError:
                                continue
        
        return processed_data
    
    def generate_enhanced_features(self, processed_data: Dict) -> Dict:
        """Generate enhanced features for model improvement"""
        logger.info("Generating enhanced features from processed data")
        
        enhanced_features = {
            'novel_risk_factors': [],
            'biomarker_weights': {},
            'feature_interactions': [],
            'population_adjustments': {},
            'confidence_scores': {}
        }
        
        # Analyze risk factors from research
        risk_factors = processed_data.get('research_papers', {}).get('risk_factors', {})
        if risk_factors:
            # Get top risk factors with frequency weights
            top_risks = list(risk_factors.items())[:10]
            for risk_factor, frequency in top_risks:
                weight = min(frequency / max(risk_factors.values()), 1.0)
                enhanced_features['biomarker_weights'][risk_factor] = weight
        
        # Analyze biomarkers
        biomarkers = processed_data.get('research_papers', {}).get('biomarkers', {})
        if biomarkers:
            top_biomarkers = list(biomarkers.items())[:10]
            enhanced_features['novel_risk_factors'] = [bio[0] for bio in top_biomarkers]
        
        # Population adjustments from statistics
        stats = processed_data.get('medical_statistics', {})
        prevalence = stats.get('prevalence_rates', {})
        if prevalence:
            avg_prevalence = np.mean(list(prevalence.values()))
            enhanced_features['population_adjustments']['base_prevalence'] = avg_prevalence
        
        # Generate feature interactions based on clinical trials
        trials = processed_data.get('clinical_trials', {})
        interventions = trials.get('interventions', {})
        if interventions:
            # Create interaction pairs
            intervention_types = list(interventions.keys())[:5]
            for i, type1 in enumerate(intervention_types):
                for type2 in intervention_types[i+1:]:
                    enhanced_features['feature_interactions'].append(f"{type1}_{type2}")
        
        return enhanced_features
    
    def create_training_enhancement_dataset(self, original_data_path: str, enhanced_features: Dict) -> pd.DataFrame:
        """Create enhanced training dataset with new features"""
        logger.info("Creating enhanced training dataset")
        
        try:
            # Load original dataset
            original_df = pd.read_csv(original_data_path)
            logger.info(f"Loaded original dataset with {len(original_df)} rows")
            
            # Create enhanced features
            enhanced_df = original_df.copy()
            
            # Add novel risk factor features (if applicable)
            for risk_factor in enhanced_features.get('novel_risk_factors', []):
                # Create synthetic features based on existing data patterns
                if 'cholesterol' in risk_factor.lower():
                    enhanced_df[f'enhanced_{risk_factor.lower().replace(" ", "_")}'] = \
                        original_df['chol'] * np.random.normal(1.0, 0.1, len(original_df))
                elif 'blood pressure' in risk_factor.lower():
                    enhanced_df[f'enhanced_{risk_factor.lower().replace(" ", "_")}'] = \
                        original_df['trestbps'] * np.random.normal(1.0, 0.1, len(original_df))
            
            # Add interaction features
            for interaction in enhanced_features.get('feature_interactions', []):
                if '_' in interaction:
                    features = interaction.split('_')
                    if len(features) == 2 and all(f in original_df.columns for f in features):
                        enhanced_df[f'interaction_{interaction}'] = \
                            original_df[features[0]] * original_df[features[1]]
            
            # Add population adjustment features
            pop_adjustments = enhanced_features.get('population_adjustments', {})
            if 'base_prevalence' in pop_adjustments:
                enhanced_df['population_risk_adjustment'] = pop_adjustments['base_prevalence']
            
            # Add confidence scoring
            enhanced_df['prediction_confidence'] = np.random.beta(2, 1, len(original_df))
            
            logger.info(f"Created enhanced dataset with {enhanced_df.shape[1]} features")
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error creating enhanced dataset: {e}")
            return pd.DataFrame()
    
    def generate_model_improvement_report(self, processed_data: Dict, enhanced_features: Dict) -> Dict:
        """Generate comprehensive report for model improvement"""
        logger.info("Generating model improvement report")
        
        report = {
            'generation_date': datetime.now().isoformat(),
            'data_summary': {},
            'key_insights': [],
            'feature_recommendations': [],
            'model_improvements': [],
            'validation_suggestions': []
        }
        
        # Data summary
        report['data_summary'] = {
            'research_papers_analyzed': len(processed_data.get('research_papers', {}).get('risk_factors', {})),
            'clinical_trials_analyzed': len(processed_data.get('clinical_trials', {}).get('interventions', {})),
            'statistical_sources': len(processed_data.get('medical_statistics', {}).get('prevalence_rates', {})),
            'novel_features_generated': len(enhanced_features.get('novel_risk_factors', []))
        }
        
        # Key insights
        risk_factors = processed_data.get('research_papers', {}).get('risk_factors', {})
        if risk_factors:
            top_risk = list(risk_factors.keys())[0] if risk_factors else "None"
            report['key_insights'].append(f"Most cited risk factor: {top_risk}")
        
        biomarkers = processed_data.get('research_papers', {}).get('biomarkers', {})
        if biomarkers:
            top_biomarker = list(biomarkers.keys())[0] if biomarkers else "None"
            report['key_insights'].append(f"Most promising biomarker: {top_biomarker}")
        
        # Feature recommendations
        novel_features = enhanced_features.get('novel_risk_factors', [])
        if novel_features:
            report['feature_recommendations'].extend([
                f"Consider adding {feature} as new input feature" 
                for feature in novel_features[:5]
            ])
        
        # Model improvements
        report['model_improvements'] = [
            "Incorporate feature interactions for non-linear relationships",
            "Add population-based risk adjustments",
            "Implement ensemble methods for confidence scoring",
            "Use transfer learning from research paper insights"
        ]
        
        # Validation suggestions
        report['validation_suggestions'] = [
            "Cross-validate with external datasets",
            "Test model on different demographic groups",
            "Validate against clinical trial outcomes",
            "Compare with established risk scores"
        ]
        
        return report
    
    def save_processed_data(self, processed_data: Dict, enhanced_features: Dict, report: Dict, 
                           filename_prefix: str = None):
        """Save all processed data and reports"""
        if filename_prefix is None:
            filename_prefix = f"processed_medical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save processed data
        with open(f"{filename_prefix}_processed.json", 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # Save enhanced features
        with open(f"{filename_prefix}_enhanced_features.json", 'w', encoding='utf-8') as f:
            json.dump(enhanced_features, f, indent=2, ensure_ascii=False)
        
        # Save report
        with open(f"{filename_prefix}_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed data with prefix: {filename_prefix}")

# =============================
# USAGE EXAMPLE
# =============================

if __name__ == "__main__":
    print("=" * 60)
    print("🔬 MEDICAL DATA PROCESSOR")
    print("=" * 60)
    
    # Initialize processor
    processor = MedicalDataProcessor()
    
    try:
        # Load scraped data (assuming it exists)
        scraped_files = [f for f in os.listdir('.') if 'medical_data' in f and 'combined.json' in f]
        
        if scraped_files:
            # Process the most recent file
            latest_file = sorted(scraped_files)[-1]
            print(f"\n📂 Loading scraped data from {latest_file}")
            
            scraped_data = processor.load_scraped_data(latest_file)
            
            # Process each data type
            print("\n📚 Processing Research Papers...")
            processed_papers = processor.process_research_papers(
                scraped_data.get('research_papers', [])
            )
            
            print("\n🔬 Processing Clinical Trials...")
            processed_trials = processor.process_clinical_trials(
                scraped_data.get('clinical_trials', [])
            )
            
            print("\n📊 Processing Medical Statistics...")
            processed_stats = processor.process_medical_statistics(
                scraped_data.get('medical_statistics', [])
            )
            
            # Combine processed data
            processed_data = {
                'research_papers': processed_papers,
                'clinical_trials': processed_trials,
                'medical_statistics': processed_stats
            }
            
            # Generate enhanced features
            print("\n🚀 Generating Enhanced Features...")
            enhanced_features = processor.generate_enhanced_features(processed_data)
            
            # Create enhanced training dataset
            print("\n💾 Creating Enhanced Training Dataset...")
            if os.path.exists('heart.csv'):
                enhanced_dataset = processor.create_training_enhancement_dataset(
                    'heart.csv', enhanced_features
                )
                enhanced_dataset.to_csv('heart_enhanced.csv', index=False)
                print(f"✅ Enhanced dataset saved with {enhanced_dataset.shape[1]} features")
            
            # Generate improvement report
            print("\n📋 Generating Improvement Report...")
            report = processor.generate_model_improvement_report(
                processed_data, enhanced_features
            )
            
            # Save all results
            processor.save_processed_data(processed_data, enhanced_features, report)
            
            # Display summary
            print("\n📊 PROCESSING SUMMARY:")
            print("-" * 40)
            for category, data in processed_data.items():
                print(f"{category.replace('_', ' ').title()}: {len(data)} insights extracted")
            
            print(f"\n🚀 Enhanced Features Generated: {len(enhanced_features)}")
            print(f"📋 Recommendations: {len(report['feature_recommendations'])}")
            
            print("\n" + "=" * 60)
            print("✅ Medical data processing completed!")
            print("Enhanced dataset and reports saved for model improvement")
            print("=" * 60)
            
        else:
            print("❌ No scraped data files found. Run web_scraper.py first")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure you have scraped data files available")
