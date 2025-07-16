#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN


class SimilarityAnalyzer:
    def __init__(self, results_file):
        """Initialize analyzer with results file"""
        self.results_file = results_file
        self.results = self.load_results()
        self.df = pd.DataFrame(self.results)
    
    def load_results(self):
        """Load results from JSON file"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def basic_statistics(self):
        """Print basic statistics about the similarities"""
        similarities = self.df['similarity'].dropna()
        alignment_scores = self.df['alignment_score'].dropna()
        
        print("="*60)
        print("SIMILARITY ANALYSIS REPORT")
        print("="*60)
        
        print(f"\nBasic Statistics:")
        print(f"Total samples: {len(self.df)}")
        print(f"Valid similarities: {len(similarities)}")
        print(f"Valid alignment scores: {len(alignment_scores)}")
        
        if len(similarities) > 0:
            print(f"\nSimilarity Statistics:")
            print(f"  Mean: {similarities.mean():.4f}")
            print(f"  Median: {similarities.median():.4f}")
            print(f"  Std: {similarities.std():.4f}")
            print(f"  Min: {similarities.min():.4f}")
            print(f"  Max: {similarities.max():.4f}")
            print(f"  25th percentile: {similarities.quantile(0.25):.4f}")
            print(f"  75th percentile: {similarities.quantile(0.75):.4f}")
        
        if len(alignment_scores) > 0:
            print(f"\nAlignment Score Statistics:")
            print(f"  Mean: {alignment_scores.mean():.4f}")
            print(f"  Median: {alignment_scores.median():.4f}")
            print(f"  Std: {alignment_scores.std():.4f}")
            print(f"  Min: {alignment_scores.min():.4f}")
            print(f"  Max: {alignment_scores.max():.4f}")
    
    def similarity_distribution(self):
        """Analyze similarity distribution"""
        similarities = self.df['similarity'].dropna()
        
        if len(similarities) == 0:
            print("No valid similarities found.")
            return
        
        # Define categories
        high_sim = similarities[similarities > 0.8]
        med_sim = similarities[(similarities >= 0.5) & (similarities <= 0.8)]
        low_sim = similarities[similarities < 0.5]
        
        print(f"\nSimilarity Distribution:")
        print(f"  High similarity (>0.8): {len(high_sim)} ({len(high_sim)/len(similarities)*100:.1f}%)")
        print(f"  Medium similarity (0.5-0.8): {len(med_sim)} ({len(med_sim)/len(similarities)*100:.1f}%)")
        print(f"  Low similarity (<0.5): {len(low_sim)} ({len(low_sim)/len(similarities)*100:.1f}%)")
        
        return {
            'high': len(high_sim),
            'medium': len(med_sim),
            'low': len(low_sim)
        }
    
    def find_outliers(self, threshold_low=0.3, threshold_high=0.95):
        """Find samples with very low or very high similarities"""
        similarities = self.df['similarity'].dropna()
        
        low_outliers = self.df[self.df['similarity'] < threshold_low]
        high_outliers = self.df[self.df['similarity'] > threshold_high]
        
        print(f"\nOutlier Analysis:")
        print(f"  Very low similarities (<{threshold_low}): {len(low_outliers)}")
        print(f"  Very high similarities (>{threshold_high}): {len(high_outliers)}")
        
        if len(low_outliers) > 0:
            print(f"\nSamples with lowest similarities:")
            low_sorted = low_outliers.nsmallest(5, 'similarity')
            for _, row in low_sorted.iterrows():
                text_preview = row['text'][:100] + "..." if len(row['text']) > 100 else row['text']
                print(f"  Similarity: {row['similarity']:.4f} - Text: {text_preview}")
        
        return low_outliers, high_outliers
    
    def correlation_analysis(self):
        """Analyze correlations between different metrics"""
        if 'audio_duration' in self.df.columns:
            corr_duration = self.df['similarity'].corr(self.df['audio_duration'])
            print(f"\nCorrelation with audio duration: {corr_duration:.4f}")
        
        if 'alignment_score' in self.df.columns:
            corr_alignment = self.df['similarity'].corr(self.df['alignment_score'])
            print(f"Correlation between similarity and alignment: {corr_alignment:.4f}")
    
    def text_length_analysis(self):
        """Analyze relationship between text length and similarity"""
        self.df['text_length'] = self.df['text'].str.len()
        
        # Correlation with text length
        corr_text_length = self.df['similarity'].corr(self.df['text_length'])
        print(f"\nCorrelation with text length: {corr_text_length:.4f}")
        
        # Average similarity by text length bins
        self.df['text_length_bin'] = pd.cut(self.df['text_length'], 
                                           bins=5, 
                                           labels=['Very Short', 'Short', 'Medium', 'Long', 'Very Long'])
        
        length_stats = self.df.groupby('text_length_bin')['similarity'].agg(['mean', 'count', 'std'])
        print(f"\nSimilarity by text length:")
        print(length_stats)
    
    def save_filtered_datasets(self, output_dir="filtered_datasets"):
        """Save filtered datasets based on similarity thresholds"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # High quality samples (similarity > 0.8)
        high_quality = self.df[self.df['similarity'] > 0.8]
        high_quality.to_json(f"{output_dir}/high_quality_samples.json", 
                            orient='records', indent=2, force_ascii=False)
        
        # Medium quality samples (0.5 <= similarity <= 0.8)
        medium_quality = self.df[(self.df['similarity'] >= 0.5) & (self.df['similarity'] <= 0.8)]
        medium_quality.to_json(f"{output_dir}/medium_quality_samples.json", 
                              orient='records', indent=2, force_ascii=False)
        
        # Low quality samples (similarity < 0.5)
        low_quality = self.df[self.df['similarity'] < 0.5]
        low_quality.to_json(f"{output_dir}/low_quality_samples.json", 
                           orient='records', indent=2, force_ascii=False)
        
        print(f"\nFiltered datasets saved to {output_dir}/")
        print(f"  High quality: {len(high_quality)} samples")
        print(f"  Medium quality: {len(medium_quality)} samples")
        print(f"  Low quality: {len(low_quality)} samples")
    
    def create_visualizations(self, output_dir="plots"):
        """Create visualization plots"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Similarity distribution histogram
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['similarity'].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Similarity Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Similarity Scores')
        plt.axvline(self.df['similarity'].mean(), color='red', linestyle='--', label=f'Mean: {self.df["similarity"].mean():.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/similarity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Similarity vs Text Length scatter plot
        if 'text_length' not in self.df.columns:
            self.df['text_length'] = self.df['text'].str.len()
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['text_length'], self.df['similarity'], alpha=0.6, s=20)
        plt.xlabel('Text Length (characters)')
        plt.ylabel('Similarity Score')
        plt.title('Similarity vs Text Length')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/similarity_vs_text_length.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Similarity vs Audio Duration (if available)
        if 'audio_duration' in self.df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(self.df['audio_duration'], self.df['similarity'], alpha=0.6, s=20)
            plt.xlabel('Audio Duration (seconds)')
            plt.ylabel('Similarity Score')
            plt.title('Similarity vs Audio Duration')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/similarity_vs_duration.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Box plot by similarity categories
        categories = []
        for sim in self.df['similarity']:
            if sim > 0.8:
                categories.append('High')
            elif sim >= 0.5:
                categories.append('Medium')
            else:
                categories.append('Low')
        
        self.df['similarity_category'] = categories
        
        plt.figure(figsize=(8, 6))
        self.df.boxplot(column='similarity', by='similarity_category', figsize=(8, 6))
        plt.title('Similarity Distribution by Category')
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/similarity_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to {output_dir}/")
    
    def generate_report(self, output_file="similarity_analysis_report.txt"):
        """Generate a comprehensive text report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("SIMILARITY ANALYSIS REPORT\n")
            f.write("="*50 + "\n\n")
            
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            self.basic_statistics()
            self.similarity_distribution()
            self.find_outliers()
            self.correlation_analysis()
            self.text_length_analysis()
            
            sys.stdout = original_stdout
        
        print(f"\nDetailed report saved to {output_file}")

def main():
    # Configuration
    results_file = "dataset_similarities.json"  # Change to your results file
    
    if not Path(results_file).exists():
        print(f"Results file {results_file} not found!")
        return
    
    # Initialize analyzer
    analyzer = SimilarityAnalyzer(results_file)
    
    # Run analysis
    print("Running similarity analysis...")
    analyzer.basic_statistics()
    analyzer.similarity_distribution()
    analyzer.find_outliers()
    analyzer.correlation_analysis()
    analyzer.text_length_analysis()
    
    # Save filtered datasets
    analyzer.save_filtered_datasets()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Generate comprehensive report
    analyzer.generate_report()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()