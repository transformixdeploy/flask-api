import os
import pandas as pd
import numpy as np
import json
import google.generativeai as genai
from datetime import datetime
from flask import Flask, request, jsonify
from collections import defaultdict
from dotenv import load_dotenv
app = Flask(__name__)
load_dotenv()
GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)
class DataSchemaAnalyzer:
    def __init__(self, df):
        self.df = df
        
    def analyze_schema(self):
        return self.fallback_schema_analysis()
    
    def fallback_schema_analysis(self):
 
        numeric_cols = list(self.df.select_dtypes(include=[np.number]).columns)
        text_cols = list(self.df.select_dtypes(include=['object', 'string']).columns)
        
        column_names_lower = [col.lower() for col in self.df.columns]
        all_text_content = ' '.join(column_names_lower)
        
        business_domain = "general business"
        primary_entity = "record"
        
        if any(word in all_text_content for word in ['customer', 'client', 'buyer']):
            business_domain = "customer management"
            primary_entity = "customer"
        elif any(word in all_text_content for word in ['patient', 'medical', 'health']):
            business_domain = "healthcare"
            primary_entity = "patient"
        elif any(word in all_text_content for word in ['student', 'course', 'grade']):
            business_domain = "education"
            primary_entity = "student"
        elif any(word in all_text_content for word in ['product', 'inventory', 'stock']):
            business_domain = "inventory management"
            primary_entity = "product"
        elif any(word in all_text_content for word in ['diving', 'diver', 'certification', 'specialty', 'scuba']):
            business_domain = "diving education and certification"
            primary_entity = "diver"
        elif any(word in all_text_content for word in ['employee', 'staff', 'hr', 'salary']):
            business_domain = "human resources"
            primary_entity = "employee"
        elif any(word in all_text_content for word in ['sales', 'revenue', 'profit', 'order']):
            business_domain = "sales and revenue"
            primary_entity = "transaction"
        elif any(word in all_text_content for word in ['marketing', 'campaign', 'lead', 'conversion']):
            business_domain = "marketing analytics"
            primary_entity = "lead"
        
        categorical_fields = []
        for col in text_cols[:5]:
            unique_count = self.df[col].nunique()
            total_count = len(self.df)
            
            if 2 <= unique_count <= 50 or (unique_count / total_count < 0.2 and unique_count > 1):
                top_values = list(self.df[col].value_counts().head(5).index)
                categorical_fields.append({
                    "column": col,
                    "suggested_filters": [str(val) for val in top_values]
                })
        
        return {
            "business_domain": business_domain,
            "primary_entity": primary_entity,
            "categorical_fields": categorical_fields
        }

def generate_smart_questions(df, schema_analysis):
    entity = schema_analysis['primary_entity']
    domain = schema_analysis['business_domain']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_fields = schema_analysis.get('categorical_fields', [])
    
    questions = []
    
    questions.extend([
        {
            "title": "Data Overview",
            "question": f"What are the key patterns in this {domain} data?"
        },
        {
            "title": "Summary Analysis", 
            "question": f"Show me a summary of all {entity}s in the dataset"
        }
    ])
    
    if numeric_cols:
        first_numeric = numeric_cols[0]
        questions.extend([
            {
                "title": "Statistical Analysis",
                "question": f"What's the average and total {first_numeric}?"
            },
            {
                "title": "Distribution Analysis",
                "question": f"Show me statistics for {first_numeric} across different groups"
            }
        ])
    
    if categorical_fields:
        first_cat = categorical_fields[0]['column']
        questions.extend([
            {
                "title": "Category Breakdown",
                "question": f"Break down the data by {first_cat}"
            },
            {
                "title": "Popular Categories",
                "question": f"What are the most common values in {first_cat}?"
            }
        ])
    
    domain_questions = []
    
    if 'customer' in domain.lower():
        domain_questions = [
            {
                "title": "Customer Segmentation",
                "question": "Which customer segments have the highest activity?"
            },
            {
                "title": "Top Performers", 
                "question": "What are the characteristics of top-performing customers?"
            },
            {
                "title": "Trend Analysis",
                "question": "Are there any concerning trends in customer data?"
            }
        ]
    elif 'healthcare' in domain.lower():
        domain_questions = [
            {
                "title": "Condition Analysis",
                "question": "What are the most common conditions in the dataset?"
            },
            {
                "title": "Patient Demographics",
                "question": "Show me patient demographics and patterns"
            },
            {
                "title": "Data Quality Check",
                "question": "Are there any data quality issues I should know about?"
            }
        ]
    elif 'diving' in domain.lower():
        domain_questions = [
            {
                "title": "Certification Levels",
                "question": "Which divers have the most certifications?"
            },
            {
                "title": "Specialty Distribution",
                "question": "What's the distribution of specialties across divers?"
            },
            {
                "title": "Level Analysis",
                "question": "How many divers are at each certification level?"
            }
        ]
    elif 'education' in domain.lower():
        domain_questions = [
            {
                "title": "Performance Analysis",
                "question": "What are the grade distribution patterns across students?"
            },
            {
                "title": "Course Popularity",
                "question": "Which courses have the highest enrollment?"
            },
            {
                "title": "Student Success",
                "question": "What factors correlate with student success?"
            }
        ]
    elif 'sales' in domain.lower():
        domain_questions = [
            {
                "title": "Revenue Analysis",
                "question": "What are the top revenue-generating products or services?"
            },
            {
                "title": "Sales Trends",
                "question": "Show me sales trends over time"
            },
            {
                "title": "Performance Metrics",
                "question": "What are the key sales performance indicators?"
            }
        ]
    elif 'hr' in domain.lower() or 'employee' in domain.lower():
        domain_questions = [
            {
                "title": "Employee Distribution",
                "question": "How are employees distributed across departments?"
            },
            {
                "title": "Salary Analysis",
                "question": "What are the salary ranges by role and experience?"
            },
            {
                "title": "HR Metrics",
                "question": "What are the key human resources metrics?"
            }
        ]
    elif 'inventory' in domain.lower() or 'product' in domain.lower():
        domain_questions = [
            {
                "title": "Stock Analysis",
                "question": "Which products have the highest and lowest stock levels?"
            },
            {
                "title": "Product Performance",
                "question": "What are the best and worst performing products?"
            },
            {
                "title": "Inventory Trends",
                "question": "Are there any concerning inventory trends?"
            }
        ]
    elif 'marketing' in domain.lower():
        domain_questions = [
            {
                "title": "Campaign Performance",
                "question": "Which marketing campaigns are performing best?"
            },
            {
                "title": "Lead Quality",
                "question": "What are the characteristics of high-quality leads?"
            },
            {
                "title": "Conversion Analysis",
                "question": "What factors drive the highest conversion rates?"
            }
        ]
    else:
        domain_questions = [
            {
                "title": "Data Quality Assessment",
                "question": "What insights can you derive from the data quality?"
            },
            {
                "title": "Pattern Recognition",
                "question": "Analyze the distribution of records across different categories"
            },
            {
                "title": "Business Intelligence",
                "question": "What business opportunities do you see in this data?"
            }
        ]
    
    remaining_slots = 6 - len(questions)
    if remaining_slots > 0:
        questions.extend(domain_questions[:remaining_slots])
    
    while len(questions) < 6:
        generic_questions = [
            {
                "title": "Correlation Analysis",
                "question": "What correlations exist between different data fields?"
            },
            {
                "title": "Outlier Detection",
                "question": "Are there any outliers or unusual patterns in the data?"
            },
            {
                "title": "Completeness Check",
                "question": "Which data fields have the most missing information?"
            },
            {
                "title": "Value Distribution",
                "question": "Show me the distribution of values across key fields"
            }
        ]
        
        for q in generic_questions:
            if len(questions) < 6 and q not in questions:
                questions.append(q)
                break
        else:
            break  
    
    return questions[:6]  

def calculate_data_metrics(df):
    try:
        total_rows = len(df)
        total_columns = len(df.columns)
        
        total_cells = total_rows * total_columns
        missing_cells = df.isnull().sum().sum()
        missing_ratio = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        num_numeric_columns = len(numeric_columns)
        
        text_columns = df.select_dtypes(include=['object', 'string']).columns
        num_text_columns = len(text_columns)
        
        completeness_by_column = {}
        for col in df.columns:
            non_null_count = df[col].count()
            completeness_pct = (non_null_count / total_rows) * 100 if total_rows > 0 else 0
            completeness_by_column[col] = {
                'non_null_count': int(non_null_count),
                'completeness_percentage': round(completeness_pct, 1)
            }
        
        unique_analysis = {}
        for col in df.columns:
            unique_count = df[col].nunique()
            unique_ratio = (unique_count / total_rows) * 100 if total_rows > 0 else 0
            unique_analysis[col] = {
                'unique_count': int(unique_count),
                'unique_ratio': round(unique_ratio, 1)
            }
        
        return {
            'total_rows': int(total_rows),
            'total_columns': int(total_columns),
            'missing_data_ratio': round(missing_ratio, 2),
            'num_numeric_columns': int(num_numeric_columns),
            'num_text_columns': int(num_text_columns),
            'completeness_by_column': completeness_by_column,
            'unique_analysis': unique_analysis,
            'column_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    
    except Exception as e:
        return {
            'error': f'Error calculating metrics: {str(e)}',
            'total_rows': 0,
            'total_columns': 0,
            'missing_data_ratio': 0,
            'num_numeric_columns': 0
        }
def analyze_purchase_combinations(self, question):
        """Specifically handle questions about product combinations/bundles"""
        
        # First, run manual debug to see what's actually happening
        debug_results = self.debug_purchase_combinations_manual()
        
        combined_fields = self.detect_combined_fields()
        
        if not combined_fields:
            return {
                "analysis": "I couldn't find any fields that contain product combination data. The dataset might have product information, but it's not in a format that can be analyzed for purchase combinations.",
                "confidence": 0.3,
                "key_findings": ["No parseable combination fields detected"],
                "relevant_statistics": {},
                "actionable_insights": [
                    "Check if purchase data is stored in individual columns",
                    "Verify the format of purchase-related fields"
                ],
                "data_evidence": ["Analyzed field structures for combination patterns"],
                "confidence_level": "low",
                "follow_up_questions": [
                    "What format is the purchase combination data stored in?",
                    "Are there specific combination columns I should focus on?"
                ]
            }
        
        # Find the most likely product combination field
        combination_field = None
        for field_name in combined_fields.keys():
            field_lower = field_name.lower()
            if any(indicator in field_lower for indicator in ['purchase', 'product', 'item', 'buy', 'order', 'history']):
                combination_field = field_name
                break
        
        if not combination_field:
            combination_field = list(combined_fields.keys())[0]
        
        # Use the same analysis that's cached in get_combined_fields_analysis
        combined_analysis = self.get_combined_fields_analysis()
        
        if combination_field not in combined_analysis:
            return {
                "analysis": f"I found a potential combination field '{combination_field}' but couldn't extract meaningful combination data from it.",
                "confidence": 0.2,
                "key_findings": ["Combination field found but no patterns extracted"],
                "relevant_statistics": {},
                "actionable_insights": ["Check the data format in the combination field"],
                "data_evidence": [f"Attempted to parse field: {combination_field}"],
                "confidence_level": "low",
                "follow_up_questions": ["What do the purchase combinations look like in your dataset?"]
            }
        
        # Get combination data from the consistent analysis
        field_analysis = combined_analysis[combination_field]
        combination_patterns = field_analysis['combination_patterns']
        
        if not combination_patterns['top_10_combinations']:
            return {
                "analysis": f"I found the combination field '{combination_field}' but no meaningful patterns were extracted.",
                "confidence": 0.2,
                "key_findings": ["Combination field found but no patterns extracted"],
                "relevant_statistics": {},
                "actionable_insights": ["Check the data format in the combination field"],
                "data_evidence": [f"Attempted to parse field: {combination_field}"],
                "confidence_level": "low",
                "follow_up_questions": ["What do the purchase combinations look like in your dataset?"]
            }
        
        # Get top combinations
        top_combinations = combination_patterns['top_10_combinations']
        most_common_combo = top_combinations[0] if top_combinations else None
        
        analysis = f"Based on analysis of the '{combination_field}' field, "
        
        if most_common_combo:
            combo_pattern, frequency = most_common_combo
            total_records = combination_patterns['customers_with_combinations']
            percentage = (frequency / total_records) * 100 if total_records > 0 else 0
            
            # Clean up the combination display
            combo_display = combo_pattern.replace(',', ' + ').title()
            
            analysis += f"the most common purchase combination is '{combo_display}' appearing {frequency:,} times "
            analysis += f"({percentage:.1f}% of customers with purchase data). "
            
            if len(top_combinations) > 1:
                analysis += f"The top 5 most popular combinations are: "
                top_5_display = []
                for combo, count in top_combinations[:5]:
                    clean_combo = combo.replace(',', ' + ').title()
                    top_5_display.append(f"{clean_combo} ({count:,} times)")
                analysis += "; ".join(top_5_display) + ". "
        
        analysis += f"In total, there are {combination_patterns['total_unique_combinations']:,} unique purchase combinations across {combination_patterns['customers_with_combinations']:,} customers."
        
        # Add debug info to the analysis
        if debug_results and debug_results['butter_rice_variations']:
            analysis += f"\n\nDEBUG INFO: Found {len(debug_results['butter_rice_variations'])} Butter+Rice variations in manual analysis."
        
        # Build key findings
        key_findings = []
        if most_common_combo:
            combo_display = most_common_combo[0].replace(',', ' + ').title()
            key_findings.append(f"Most common combination: {combo_display} ({most_common_combo[1]:,} occurrences)")
        
        key_findings.extend([
            f"Total unique combinations: {combination_patterns['total_unique_combinations']:,}",
            f"Customers with combination data: {combination_patterns['customers_with_combinations']:,}",
            f"Average combination size: {combination_patterns['avg_items_per_combination']:.1f} items"
        ])
        
        # Build relevant statistics
        relevant_stats = {
            "most_common_combination_count": most_common_combo[1] if most_common_combo else 0,
            "total_unique_combinations": combination_patterns['total_unique_combinations'],
            "customers_with_combinations": combination_patterns['customers_with_combinations'],
            "avg_items_per_combination": combination_patterns['avg_items_per_combination']
        }
        
        return {
            "analysis": analysis,
            "confidence": 0.9,
            "key_findings": key_findings,
            "relevant_statistics": relevant_stats,
            "actionable_insights": [
                f"Create targeted bundles based on the '{most_common_combo[0].replace(',', ' + ').title()}' combination" if most_common_combo else "Analyze combination patterns for bundling opportunities",
                "Develop cross-selling strategies based on popular combinations",
                "Consider promotional pricing for frequent combinations",
                "Analyze seasonal trends in popular combinations"
            ],
            "data_evidence": [
                f"Analyzed {combination_patterns['customers_with_combinations']:,} customer purchase combinations",
                f"Identified {combination_patterns['total_unique_combinations']:,} unique combination patterns",
                f"Parsed combinations from '{combination_field}' field"
            ],
            "confidence_level": "high",
            "follow_up_questions": [
                "What customer segments prefer which combinations?",
                "How do combination preferences change over time?",
                "Which combinations have the highest profit margins?"
            ]
        }
class SmartRAGAssistant:
    def __init__(self, df, schema_analysis):
        self.df = df
        self.schema_analysis = schema_analysis
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        # Cache for processed data
        self._parsed_fields_cache = {}
    
    def detect_combined_fields(self):
        """Detect fields that might contain combined/separated data like products, tags, etc."""
        combined_fields = {}
        text_cols = self.df.select_dtypes(include=['object', 'string']).columns
        
        for col in text_cols:
            # Check if column contains common separators
            sample_values = self.df[col].dropna().head(100).astype(str)
            
            separators_found = []
            common_separators = [',', ';', '|', '/', '&', '+', '-']
            
            for sep in common_separators:
                if any(sep in str(val) for val in sample_values):
                    separators_found.append(sep)
            
            # Check if field name suggests it contains multiple items
            field_indicators = ['history', 'products', 'items', 'tags', 'categories', 'skills', 'interests', 'purchase']
            if any(indicator in col.lower() for indicator in field_indicators) or separators_found:
                # Determine the most likely separator
                separator_counts = {}
                for sep in separators_found:
                    count = sum(str(val).count(sep) for val in sample_values)
                    separator_counts[sep] = count
                
                if separator_counts:
                    most_common_sep = max(separator_counts, key=separator_counts.get)
                    combined_fields[col] = {
                        'separator': most_common_sep,
                        'type': 'combined_items'
                    }
        
        return combined_fields
    
    def parse_combined_field(self, column_name, separator=','):
        """Parse a combined field into individual items and return frequency analysis"""
        if column_name in self._parsed_fields_cache:
            return self._parsed_fields_cache[column_name]
        
        if column_name not in self.df.columns:
            return None
        
        # Extract all individual items
        all_items = []
        for value in self.df[column_name].dropna():
            if pd.isna(value):
                continue
            # Split by separator and clean items
            items = [item.strip().lower() for item in str(value).split(separator) if item.strip()]
            all_items.extend(items)
        
        if not all_items:
            return None
        
        # Count frequency of each item
        from collections import Counter
        item_counts = Counter(all_items)
        
        # Create analysis
        analysis = {
            'total_individual_items': len(all_items),
            'unique_items': len(item_counts),
            'most_common': item_counts.most_common(20),  # Top 20 items
            'item_frequency': dict(item_counts),
            'coverage_stats': {
                'records_with_data': int(self.df[column_name].count()),
                'average_items_per_record': len(all_items) / max(1, self.df[column_name].count())
            }
        }
        
        # Cache the result
        self._parsed_fields_cache[column_name] = analysis
        return analysis
    
    def analyze_combination_patterns(self, column_name, separator=','):
        """Analyze full combination patterns (not individual items)"""
        if column_name not in self.df.columns:
            return None
        
        # Get all non-null combination strings
        combinations = []
        combination_sizes = []
        
        for value in self.df[column_name].dropna():
            if pd.isna(value) or str(value).strip() == '':
                continue
            
            # Clean and normalize the combination string
            clean_combo = str(value).strip()
            # Sort items in combination for consistent matching
            items = [item.strip().lower() for item in clean_combo.split(separator) if item.strip()]
            if items:
                # Sort items to treat "A,B" and "B,A" as the same combination
                sorted_items = sorted(items)
                normalized_combo = ','.join(sorted_items)
                combinations.append(normalized_combo)
                combination_sizes.append(len(items))
        
        if not combinations:
            return None
        
        # Count frequency of each combination
        from collections import Counter
        combo_counts = Counter(combinations)
        
        # Calculate statistics
        analysis = {
            'total_customers_with_combinations': len(combinations),
            'unique_combinations': len(combo_counts),
            'most_common_combinations': combo_counts.most_common(20),
            'avg_items_per_combination': sum(combination_sizes) / len(combination_sizes) if combination_sizes else 0,
            'combination_size_distribution': Counter(combination_sizes)
        }
        
        return analysis
    
    def create_data_context(self, max_rows=100):
        context = {
            "metadata": {
                "business_domain": self.schema_analysis.get('business_domain', 'general business'),
                "primary_entity": self.schema_analysis.get('primary_entity', 'record'),
                "total_records": len(self.df),
                "columns": list(self.df.columns),
                "data_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            },
            "sample_data": self.df.head(min(max_rows, len(self.df))).to_dict('records'),
            "statistical_summary": self.get_statistical_summary(),
            "categorical_insights": self.get_categorical_insights(),
            "data_quality": self.analyze_data_quality(),
            "business_insights": self.schema_analysis.get('insights', []),
            "combined_fields_analysis": self.get_combined_fields_analysis()
        }
        return context
    
    def get_combined_fields_analysis(self):
        """Analyze fields that contain combined data like products, tags, etc."""
        combined_fields = self.detect_combined_fields()
        analysis = {}
        
        for field_name, field_info in combined_fields.items():
            parsed_data = self.parse_combined_field(field_name, field_info['separator'])
            if parsed_data:
                analysis[field_name] = {
                    'field_type': field_info['type'],
                    'separator_used': field_info['separator'],
                    'total_individual_items': parsed_data['total_individual_items'],
                    'unique_items_count': parsed_data['unique_items'],
                    'top_10_items': parsed_data['most_common'][:10],
                    'coverage_stats': parsed_data['coverage_stats']
                }
        
        return analysis
    
    def get_statistical_summary(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        summary = {}
        
        for col in numeric_cols:
            if not self.df[col].isna().all():
                try:
                    count_val = int(self.df[col].count())
                    mean_val = float(self.df[col].mean())
                    median_val = float(self.df[col].median())
                    std_val = float(self.df[col].std())
                    min_val = float(self.df[col].min())
                    max_val = float(self.df[col].max())
                    q25_val = float(self.df[col].quantile(0.25))
                    q75_val = float(self.df[col].quantile(0.75))
                    sum_val = float(self.df[col].sum())
                    unique_count = int(self.df[col].nunique())
                    
                    # Handle NaN values
                    for val_name, val in [('mean_val', mean_val), ('median_val', median_val), 
                                        ('std_val', std_val), ('min_val', min_val), 
                                        ('max_val', max_val), ('q25_val', q25_val), 
                                        ('q75_val', q75_val), ('sum_val', sum_val)]:
                        if pd.isna(val):
                            locals()[val_name] = 0.0
                    
                    summary[col] = {
                        "count": count_val,
                        "mean": mean_val,
                        "median": median_val,
                        "std": std_val,
                        "min": min_val,
                        "max": max_val,
                        "q25": q25_val,
                        "q75": q75_val,
                        "sum": sum_val,
                        "unique_count": unique_count
                    }
                except Exception as e:
                    print(f"Skipping column {col} due to error: {e}")
                    continue
        
        return summary
    
    def get_categorical_insights(self):
        text_cols = self.df.select_dtypes(include=['object', 'string']).columns
        insights = {}
        
        for col in text_cols:
            if self.df[col].nunique() < len(self.df) * 0.8:
                value_counts = self.df[col].value_counts()
                insights[col] = {
                    "unique_count": int(self.df[col].nunique()),
                    "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "top_5_values": {str(k): int(v) for k, v in value_counts.head(5).items()},
                    "distribution": {str(k): int(v) for k, v in value_counts.head(10).items()}
                }
        
        return insights
    
    def analyze_data_quality(self):
        return {
            "completeness": float((self.df.notna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100),
            "missing_by_column": {col: int(self.df[col].isnull().sum()) for col in self.df.columns},
            "duplicate_rows": int(self.df.duplicated().sum()),
            "empty_strings": {col: int((self.df[col].astype(str) == '').sum()) for col in self.df.select_dtypes(include=['object', 'string']).columns}
        }
    
    def handle_specific_question_types(self, question):
        """Handle specific question types that require special processing"""
        question_lower = question.lower()
        
        # Detect product/item-related questions
        product_keywords = ['product', 'item', 'purchase', 'bought', 'buy', 'sold', 'selling']
        ranking_keywords = ['most', 'popular', 'common', 'frequent', 'top', 'best']
        combination_keywords = ['combination', 'combo', 'together', 'bundle', 'pair', 'group', 'set']
        
        is_product_question = any(keyword in question_lower for keyword in product_keywords)
        is_ranking_question = any(keyword in question_lower for keyword in ranking_keywords)
        is_combination_question = any(keyword in question_lower for keyword in combination_keywords)
        
        if is_product_question and is_ranking_question:
            if is_combination_question:
                return self.analyze_purchase_combinations(question)
            else:
                return self.analyze_product_popularity(question)
        
        return None
    
    def analyze_product_popularity(self, question):
        """Specifically handle questions about product popularity/frequency"""
        combined_fields = self.detect_combined_fields()
        
        if not combined_fields:
            return {
                "analysis": "I couldn't find any fields that contain product or item data in a format that can be analyzed for individual product popularity. The dataset might have product information, but it's not in a separable format.",
                "confidence": 0.3,
                "key_findings": ["No parseable product fields detected"],
                "relevant_statistics": {},
                "actionable_insights": [
                    "Check if product data is stored in individual columns",
                    "Verify the format of product-related fields",
                    "Consider data preprocessing if products are encoded differently"
                ],
                "data_evidence": ["Analyzed field structures for product patterns"],
                "confidence_level": "low",
                "follow_up_questions": [
                    "What format is the product data stored in?",
                    "Are there specific product columns I should focus on?"
                ]
            }
        
        # Find the most likely product field
        product_field = None
        for field_name in combined_fields.keys():
            field_lower = field_name.lower()
            if any(indicator in field_lower for indicator in ['purchase', 'product', 'item', 'buy', 'order']):
                product_field = field_name
                break
        
        if not product_field:
            product_field = list(combined_fields.keys())[0]  # Take the first combined field
        
        # Parse the field
        parsed_data = self.parse_combined_field(product_field, combined_fields[product_field]['separator'])
        
        if not parsed_data or not parsed_data['most_common']:
            return {
                "analysis": f"I found a potential product field '{product_field}' but couldn't extract meaningful product data from it. The field might be empty or in an unexpected format.",
                "confidence": 0.2,
                "key_findings": ["Product field found but no data extracted"],
                "relevant_statistics": {},
                "actionable_insights": ["Check the data format in the product field"],
                "data_evidence": [f"Attempted to parse field: {product_field}"],
                "confidence_level": "low",
                "follow_up_questions": ["What does the product data look like in your dataset?"]
            }
        
        # Get top products
        top_products = parsed_data['most_common'][:10]
        most_popular_product = top_products[0] if top_products else None
        
        analysis = f"Based on analysis of the '{product_field}' field, "
        
        if most_popular_product:
            product_name, frequency = most_popular_product
            total_records = parsed_data['coverage_stats']['records_with_data']
            percentage = (frequency / total_records) * 100 if total_records > 0 else 0
            
            analysis += f"the most purchased product is '{product_name.title()}' with {frequency:,} purchases "
            analysis += f"appearing in {percentage:.1f}% of customer records. "
            
            if len(top_products) > 1:
                analysis += f"The top 5 most popular products are: "
                top_5 = [f"{name.title()} ({count:,} purchases)" for name, count in top_products[:5]]
                analysis += ", ".join(top_5) + ". "
        
        analysis += f"In total, there are {parsed_data['unique_items']:,} unique products across {parsed_data['total_individual_items']:,} total purchase instances."
        
        # Build key findings
        key_findings = []
        if most_popular_product:
            key_findings.append(f"Most popular product: {most_popular_product[0].title()} ({most_popular_product[1]:,} purchases)")
        key_findings.append(f"Total unique products: {parsed_data['unique_items']:,}")
        key_findings.append(f"Total purchase instances: {parsed_data['total_individual_items']:,}")
        key_findings.append(f"Average products per customer: {parsed_data['coverage_stats']['average_items_per_record']:.1f}")
        
        # Build relevant statistics
        relevant_stats = {
            "most_popular_product_count": most_popular_product[1] if most_popular_product else 0,
            "total_unique_products": parsed_data['unique_items'],
            "total_purchase_instances": parsed_data['total_individual_items'],
            "customers_with_purchase_data": parsed_data['coverage_stats']['records_with_data'],
            "avg_products_per_customer": parsed_data['coverage_stats']['average_items_per_record']
        }
        
        return {
            "analysis": analysis,
            "confidence": 0.9,
            "key_findings": key_findings,
            "relevant_statistics": relevant_stats,
            "actionable_insights": [
                f"Focus marketing efforts on promoting {most_popular_product[0].title()} variants" if most_popular_product else "Analyze product distribution",
                "Investigate why certain products are more popular",
                "Consider bundling popular products with less popular ones",
                "Analyze seasonal trends in product purchases"
            ],
            "data_evidence": [
                f"Analyzed {parsed_data['coverage_stats']['records_with_data']:,} customer purchase records",
                f"Parsed individual products from '{product_field}' field",
                f"Separated {parsed_data['total_individual_items']:,} individual product purchases"
            ],
            "confidence_level": "high",
            "follow_up_questions": [
                "What are the characteristics of customers who buy the most popular products?",
                "How do product preferences vary by customer segments?",
                "What's the seasonal trend for the top products?"
            ]
        }
    
    def analyze_purchase_combinations(self, question):
        """Specifically handle questions about product combinations/bundles"""
        combined_fields = self.detect_combined_fields()
        
        if not combined_fields:
            return {
                "analysis": "I couldn't find any fields that contain product combination data. The dataset might have product information, but it's not in a format that can be analyzed for purchase combinations.",
                "confidence": 0.3,
                "key_findings": ["No parseable combination fields detected"],
                "relevant_statistics": {},
                "actionable_insights": [
                    "Check if purchase data is stored in individual columns",
                    "Verify the format of purchase-related fields"
                ],
                "data_evidence": ["Analyzed field structures for combination patterns"],
                "confidence_level": "low",
                "follow_up_questions": [
                    "What format is the purchase combination data stored in?",
                    "Are there specific combination columns I should focus on?"
                ]
            }
        
        # Find the most likely product combination field
        combination_field = None
        for field_name in combined_fields.keys():
            field_lower = field_name.lower()
            if any(indicator in field_lower for indicator in ['purchase', 'product', 'item', 'buy', 'order', 'history']):
                combination_field = field_name
                break
        
        if not combination_field:
            combination_field = list(combined_fields.keys())[0]
        
        # Analyze combinations (full strings, not individual items)
        combination_analysis = self.analyze_combination_patterns(combination_field, combined_fields[combination_field]['separator'])
        
        if not combination_analysis or not combination_analysis['most_common_combinations']:
            return {
                "analysis": f"I found a potential combination field '{combination_field}' but couldn't extract meaningful combination data from it.",
                "confidence": 0.2,
                "key_findings": ["Combination field found but no patterns extracted"],
                "relevant_statistics": {},
                "actionable_insights": ["Check the data format in the combination field"],
                "data_evidence": [f"Attempted to parse field: {combination_field}"],
                "confidence_level": "low",
                "follow_up_questions": ["What do the purchase combinations look like in your dataset?"]
            }
        
        # Get top combinations
        top_combinations = combination_analysis['most_common_combinations'][:10]
        most_common_combo = top_combinations[0] if top_combinations else None
        
        analysis = f"Based on analysis of the '{combination_field}' field, "
        
        if most_common_combo:
            combo_pattern, frequency = most_common_combo
            total_records = combination_analysis['total_customers_with_combinations']
            percentage = (frequency / total_records) * 100 if total_records > 0 else 0
            
            # Clean up the combination display
            combo_display = combo_pattern.replace(',', ' + ').title()
            
            analysis += f"the most common purchase combination is '{combo_display}' appearing {frequency:,} times "
            analysis += f"({percentage:.1f}% of customers with purchase data). "
            
            if len(top_combinations) > 1:
                analysis += f"The top 5 most popular combinations are: "
                top_5_display = []
                for combo, count in top_combinations[:5]:
                    clean_combo = combo.replace(',', ' + ').title()
                    top_5_display.append(f"{clean_combo} ({count:,} times)")
                analysis += "; ".join(top_5_display) + ". "
        
        analysis += f"In total, there are {combination_analysis['unique_combinations']:,} unique purchase combinations across {combination_analysis['total_customers_with_combinations']:,} customers."
        
        # Build key findings
        key_findings = []
        if most_common_combo:
            combo_display = most_common_combo[0].replace(',', ' + ').title()
            key_findings.append(f"Most common combination: {combo_display} ({most_common_combo[1]:,} occurrences)")
        
        key_findings.extend([
            f"Total unique combinations: {combination_analysis['unique_combinations']:,}",
            f"Customers with combination data: {combination_analysis['total_customers_with_combinations']:,}",
            f"Average combination size: {combination_analysis['avg_items_per_combination']:.1f} items"
        ])
        
        # Build relevant statistics
        relevant_stats = {
            "most_common_combination_count": most_common_combo[1] if most_common_combo else 0,
            "total_unique_combinations": combination_analysis['unique_combinations'],
            "customers_with_combinations": combination_analysis['total_customers_with_combinations'],
            "avg_items_per_combination": combination_analysis['avg_items_per_combination']
        }
        
        return {
            "analysis": analysis,
            "confidence": 0.9,
            "key_findings": key_findings,
            "relevant_statistics": relevant_stats,
            "actionable_insights": [
                f"Create targeted bundles based on the '{most_common_combo[0].replace(',', ' + ').title()}' combination" if most_common_combo else "Analyze combination patterns for bundling opportunities",
                "Develop cross-selling strategies based on popular combinations",
                "Consider promotional pricing for frequent combinations",
                "Analyze seasonal trends in popular combinations"
            ],
            "data_evidence": [
                f"Analyzed {combination_analysis['total_customers_with_combinations']:,} customer purchase combinations",
                f"Identified {combination_analysis['unique_combinations']:,} unique combination patterns",
                f"Parsed combinations from '{combination_field}' field"
            ],
            "confidence_level": "high",
            "follow_up_questions": [
                "What customer segments prefer which combinations?",
                "How do combination preferences change over time?",
                "Which combinations have the highest profit margins?"
            ]
        }
    
    def answer_question(self, question):
        try:
            # Check if question is asking about individual items in combined fields
            specific_analysis = self.handle_specific_question_types(question)
            if specific_analysis:
                return specific_analysis
            
            # Continue with general RAG approach
            data_context = self.create_data_context()
            prompt = self.create_rag_prompt(question, data_context)
            response = self.model.generate_content(prompt)
            parsed_response = self.parse_rag_response(response.text, question)
            
            return parsed_response
            
        except Exception as e:
            return self.fallback_answer(question, str(e))
    
    def create_rag_prompt(self, question, data_context):
        prompt = f"""
You are a Smart Business Intelligence Assistant with direct access to the following dataset:

BUSINESS CONTEXT:
- Domain: {data_context['metadata']['business_domain']}
- Entity Type: {data_context['metadata']['primary_entity']}
- Total Records: {data_context['metadata']['total_records']:,}
- Columns: {', '.join(data_context['metadata']['columns'])}

STATISTICAL SUMMARY:
{json.dumps(data_context['statistical_summary'], indent=2)}

CATEGORICAL DATA INSIGHTS:
{json.dumps(data_context['categorical_insights'], indent=2)}

COMBINED FIELDS ANALYSIS (Products, Tags, etc.):
{json.dumps(data_context['combined_fields_analysis'], indent=2)}

IMPORTANT: When analyzing combined fields, use the data provided above consistently:
- For individual item popularity: Use 'individual_items' -> 'top_10_individual_items'
- For combination patterns: Use 'combination_patterns' -> 'top_10_combinations'
- Always reference the SAME data source to avoid contradictions

DATA QUALITY METRICS:
{json.dumps(data_context['data_quality'], indent=2)}

SAMPLE DATA (First {len(data_context['sample_data'])} records):
{json.dumps(data_context['sample_data'], indent=2)}

USER QUESTION: "{question}"

Instructions:
1. Pay special attention to COMBINED FIELDS ANALYSIS - this contains both individual items AND combination patterns
2. For individual products: Use the 'individual_items' section
3. For product combinations: Use the 'combination_patterns' section
4. ALWAYS reference the exact same data shown above - don't perform separate analysis
5. Be consistent with naming and counts across different questions
6. Provide specific, data-driven answers based on the actual dataset
7. Include relevant statistics, trends, and patterns
8. If you need to make assumptions, state them clearly

Respond in JSON format:
{{
    "analysis": "Detailed answer to the user's question",
    "confidence": 0.85,
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
    "relevant_statistics": {{"stat_name": "value", "description": "explanation"}},
    "actionable_insights": ["Insight 1", "Insight 2"],
    "data_evidence": ["Evidence 1", "Evidence 2"],
    "confidence_level": "high",
    "follow_up_questions": ["Question 1", "Question 2"]
}}

Be specific, use actual data values, and provide concrete insights based on the dataset provided.
"""
        
        return prompt
    
    def parse_rag_response(self, response_text, original_question):
        try:
            clean_response = response_text.strip()
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:-3]
            elif clean_response.startswith('```'):
                clean_response = clean_response[3:-3]
            
            parsed = json.loads(clean_response)
            
            confidence = parsed.get('confidence', 0.7)
            if confidence > 1:
                confidence = confidence / 100
            
            return {
                "analysis": parsed.get('analysis', ''),
                "confidence": float(confidence),
                "key_findings": parsed.get('key_findings', []),
                "relevant_statistics": parsed.get('relevant_statistics', {}),
                "actionable_insights": parsed.get('actionable_insights', []),
                "data_evidence": parsed.get('data_evidence', []),
                "confidence_level": parsed.get('confidence_level', 'medium'),
                "follow_up_questions": parsed.get('follow_up_questions', [])
            }
            
        except json.JSONDecodeError as e:
            return {
                "analysis": response_text,
                "confidence": 0.5,
                "key_findings": ["Response generated but not structured properly"],
                "relevant_statistics": {},
                "actionable_insights": [],
                "data_evidence": ["Raw AI response provided"],
                "confidence_level": "low",
                "follow_up_questions": []
            }
    
    def fallback_answer(self, question, error_msg=None):
        question_lower = question.lower()
        
        analysis = f"Based on your {self.schema_analysis['business_domain']} dataset with {len(self.df):,} {self.schema_analysis['primary_entity']} records, "
        
        key_findings = []
        relevant_stats = {}
        
        if any(word in question_lower for word in ['total', 'count', 'how many']):
            analysis += f"the dataset contains {len(self.df):,} total records. "
            key_findings.append(f"Dataset contains {len(self.df):,} records")
        
        if any(word in question_lower for word in ['average', 'mean']):
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                avg_val = self.df[col].mean()
                analysis += f"The average {col} is {avg_val:.2f}. "
                key_findings.append(f"Average {col} is {avg_val:.2f}")
                relevant_stats[f"avg_{col}"] = avg_val
        
        completeness = (self.df.notna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        analysis += f"Data completeness is {completeness:.1f}%. "
        key_findings.append(f"Data is {completeness:.1f}% complete")
        relevant_stats["data_completeness"] = completeness
        
        if error_msg:
            analysis += f"Note: Enhanced AI analysis failed ({error_msg}), providing basic analysis."
        
        return {
            "analysis": analysis,
            "confidence": 0.3,
            "key_findings": key_findings,
            "relevant_statistics": relevant_stats,
            "actionable_insights": [
                "Use Data Explorer for detailed filtering",
                "Check data quality for missing values",
                "Consider domain-specific analysis"
            ],
            "data_evidence": [f"Analysis based on {len(self.df)} records"],
            "confidence_level": "low",
            "follow_up_questions": [
                "What specific aspect would you like to explore?",
                "Would you like to see data quality details?"
            ]
        }

def format_response_structure(rag_result):    
    relevant_statistics = []
    if isinstance(rag_result.get('relevant_statistics'), dict):
        for key, value in rag_result['relevant_statistics'].items():
            # Convert value to number, skip if not possible
            numeric_value = None
            
            if isinstance(value, (int, float)):
                numeric_value = value
            elif isinstance(value, str):
                try:
                    numeric_value = float(value)
                except (ValueError, TypeError):
                    continue
            else:
                continue
            
            if numeric_value is not None:
                relevant_statistics.append({
                    "key": key.replace('_', ' ').title(),
                    "value": numeric_value
                })
    
    actionable_insights = []
    insights_list = rag_result.get('actionable_insights', [])
    for i, insight in enumerate(insights_list):
        actionable_insights.append({
            "title": f"Insight {i + 1}",
            "value": insight
        })
    
    confidence_mapping = {
        "high": "high",
        "medium": "moderate", 
        "low": "low"
    }
    
    return {
        "smartAnalysis": {
            "analysis": rag_result.get('analysis', ''),
            "confidence": rag_result.get('confidence', 0.5)
        },
        "keyFindings": rag_result.get('key_findings', []),
        "relevantStatistics": relevant_statistics,
        "dataEvidence": {
            "evidences": rag_result.get('data_evidence', []),
            "confidence": confidence_mapping.get(rag_result.get('confidence_level', 'low'), 'moderate')
        },
        "actionableInsights": actionable_insights,
        "followUpQuestions": rag_result.get('follow_up_questions', [])
    }
@app.route('/ai/upload', methods=['POST'])
def upload():
    try:
        if 'data' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file part in the request"
            }), 400

        data = request.files['data']
        if data.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected"
            }), 400

        encodings = ["utf-8", "latin-1", "utf-8-sig", "cp1252", "utf-16"]
        delimiters = [",", ";"] 

        df = None
        for encoding in encodings:
            for delimiter in delimiters:
                data.stream.seek(0)  
                try:
                    df = pd.read_csv(
                        data,
                        encoding=encoding,
                        delimiter=delimiter,
                        on_bad_lines="skip"
                    )
                    print(f"Successfully read CSV with encoding={encoding}, delimiter='{delimiter}'")
                    break 
                except Exception as e:
                    print(f"Failed with encoding={encoding}, delimiter='{delimiter}':", e)
            if df is not None:
                break  

        if df is None:
            return jsonify({
                "status": "error",
                "message": "Could not parse CSV with any tried encoding/delimiter"
            }), 400

        if df.empty:
            return jsonify({
                "status": "error",
                "message": "Converted DataFrame is empty"
            }), 400

        if len(df.columns) == 0:
            return jsonify({
                "status": "error",
                "message": "No columns found in data"
            }), 400

        schema_analyzer = DataSchemaAnalyzer(df)  
        schema_analysis = schema_analyzer.analyze_schema()

        metrics = calculate_data_metrics(df)  
        response_data = {
            "domain": schema_analysis.get("business_domain"),
            "total_rows": metrics.get("total_rows"),
            "total_columns": metrics.get("total_columns"),
            "missing_data_ratio": metrics.get("missing_data_ratio"),
            "num_numeric_columns": metrics.get("num_numeric_columns")
        }

        return jsonify(response_data), 200

    except json.JSONDecodeError as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid JSON format: {str(e)}"
        }), 400

    except pd.errors.ParserError as e:
        return jsonify({
            "status": "error",
            "message": f"Error parsing data to DataFrame: {str(e)}"
        }), 400

    except Exception as e:
        print("Unexpected error:", e)
        return jsonify({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }), 500

@app.route('/ai/smart-question-examples', methods=['POST'])
def smart_question_example():
    try:
        if 'data' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file part in the request"
            }), 400

        data = request.files['data']
        if data.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected"
            }), 400

        encodings = ["utf-8", "latin-1", "utf-8-sig", "cp1252", "utf-16"]
        delimiters = [",", ";"]

        df = None
        for encoding in encodings:
            for delimiter in delimiters:
                data.stream.seek(0)
                try:
                    df = pd.read_csv(
                        data,
                        encoding=encoding,
                        delimiter=delimiter,
                        on_bad_lines="skip"
                    )
                    print(f"Successfully read CSV with encoding={encoding}, delimiter='{delimiter}'")
                    break
                except Exception as e:
                    print(f"Failed with encoding={encoding}, delimiter='{delimiter}':", e)
            if df is not None:
                break

        if df is None:
            return jsonify({
                "status": "error",
                "message": "Could not parse CSV with any tried encoding/delimiter"
            }), 400

        if df.empty:
            return jsonify({
                "status": "error",
                "message": "Converted DataFrame is empty"
            }), 400

        if len(df.columns) == 0:
            return jsonify({
                "status": "error",
                "message": "No columns found in data"
            }), 400

        schema_analyzer = DataSchemaAnalyzer(df)
        schema_analysis = schema_analyzer.analyze_schema()

        smart_questions = generate_smart_questions(df, schema_analysis)

        return jsonify(smart_questions), 200

    except Exception as e:
        print("Unexpected error:", e)
        return jsonify({
            "status": "error", 
            "message": f"Unexpected error: {str(e)}"
        }), 500
@app.route('/ai/question-answer', methods=['POST'])
def question_answer():
    try:
        if 'data' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No file part in the request"
            }), 400

        data_file = request.files['data']
        if data_file.filename == '':
            return jsonify({
                "status": "error",
                "message": "No file selected"
            }), 400

        question = request.form.get('question')
        if not question:
            return jsonify({
                "status": "error",
                "message": "Question is required"
            }), 400

        encodings = ["utf-8", "latin-1", "utf-8-sig", "cp1252", "utf-16"]
        delimiters = [",", ";"]

        df = None
        for encoding in encodings:
            for delimiter in delimiters:
                data_file.stream.seek(0)
                try:
                    df = pd.read_csv(
                        data_file,
                        encoding=encoding,
                        delimiter=delimiter,
                        on_bad_lines="skip"
                    )
                    print(f"Successfully read CSV with encoding={encoding}, delimiter='{delimiter}'")
                    break
                except Exception as e:
                    print(f"Failed with encoding={encoding}, delimiter='{delimiter}':", e)
            if df is not None:
                break

        if df is None:
            return jsonify({
                "status": "error",
                "message": "Could not parse CSV with any tried encoding/delimiter"
            }), 400

        if df.empty:
            return jsonify({
                "status": "error",
                "message": "Converted DataFrame is empty"
            }), 400

        if len(df.columns) == 0:
            return jsonify({
                "status": "error",
                "message": "No columns found in data"
            }), 400

        schema_analyzer = DataSchemaAnalyzer(df)
        schema_analysis = schema_analyzer.analyze_schema()

        rag_assistant = SmartRAGAssistant(df, schema_analysis)
        rag_result = rag_assistant.answer_question(question)

        formatted_response = format_response_structure(rag_result)

        return jsonify(formatted_response), 200

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }), 500
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)





