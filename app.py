import os
import pandas as pd
import numpy as np
import json
import google.generativeai as genai
from datetime import datetime
from flask import Flask, request, jsonify
from collections import defaultdict
from dotenv import load_dotenv
from itertools import combinations
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
        
        combination_field = None
        for field_name in combined_fields.keys():
            field_lower = field_name.lower()
            if any(indicator in field_lower for indicator in ['purchase', 'product', 'item', 'buy', 'order', 'history']):
                combination_field = field_name
                break
        
        if not combination_field:
            combination_field = list(combined_fields.keys())[0]
        
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
        
        top_combinations = combination_patterns['top_10_combinations']
        most_common_combo = top_combinations[0] if top_combinations else None
        
        analysis = f"Based on analysis of the '{combination_field}' field, "
        
        if most_common_combo:
            combo_pattern, frequency = most_common_combo
            total_records = combination_patterns['customers_with_combinations']
            percentage = (frequency / total_records) * 100 if total_records > 0 else 0
            
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
        
        key_findings = []
        if most_common_combo:
            combo_display = most_common_combo[0].replace(',', ' + ').title()
            key_findings.append(f"Most common combination: {combo_display} ({most_common_combo[1]:,} occurrences)")
        
        key_findings.extend([
            f"Total unique combinations: {combination_patterns['total_unique_combinations']:,}",
            f"Customers with combination data: {combination_patterns['customers_with_combinations']:,}",
            f"Average combination size: {combination_patterns['avg_items_per_combination']:.1f} items"
        ])
        
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
class UniversalMarketBasketAnalyzer:
    def __init__(self, df, schema_analysis):
        self.df = df
        self.schema_analysis = schema_analysis
        self.transaction_fields = schema_analysis.get('transaction_fields', [])
        self.transactions = self.prepare_universal_transactions()
    
    def prepare_universal_transactions(self):
        """Prepare transactions from any dataset structure"""
        transactions = []
        
        # If specific transaction fields are identified
        if self.transaction_fields:
            for field in self.transaction_fields:
                if field in self.df.columns:
                    for idx, row in self.df.iterrows():
                        if pd.notna(row[field]):
                            # Split comma-separated values
                            items = [item.strip() for item in str(row[field]).split(',')]
                            items = [item for item in items if item and item != '']
                            if len(items) > 1:
                                transactions.append(items)
        else:
            # Try to detect transactional patterns automatically
            text_cols = self.df.select_dtypes(include=['object', 'string']).columns
            
            for col in text_cols:
                # Look for comma-separated values
                has_commas = self.df[col].astype(str).str.contains(',', na=False).sum()
                if has_commas > len(self.df) * 0.1:  # If >10% have commas
                    for idx, row in self.df.iterrows():
                        if pd.notna(row[col]) and ',' in str(row[col]):
                            items = [item.strip() for item in str(row[col]).split(',')]
                            items = [item for item in items if item and item != '']
                            if len(items) > 1:
                                transactions.append(items)
                    break  # Use first suitable column
        
        return transactions
    
    def analyze_patterns(self, min_support=0.05):
        """Analyze universal patterns"""
        if not self.transactions:
            return [], []
        
        # Get all unique items
        all_items = set()
        for transaction in self.transactions:
            all_items.update(transaction)
        
        all_items = list(all_items)
        frequent_itemsets = []
        
        # Generate frequent itemsets
        for item in all_items:
            support = self.calculate_support([item])
            if support >= min_support:
                frequent_itemsets.append({
                    'itemset': [item],
                    'support': support,
                    'count': int(support * len(self.transactions))
                })
        
        # Generate 2-itemsets
        for combo in combinations(all_items, 2):
            support = self.calculate_support(list(combo))
            if support >= min_support:
                frequent_itemsets.append({
                    'itemset': list(combo),
                    'support': support,
                    'count': int(support * len(self.transactions))
                })
        
        # Generate association rules
        rules = []
        for itemset_data in frequent_itemsets:
            if len(itemset_data['itemset']) == 2:
                itemset = itemset_data['itemset']
                for i in range(len(itemset)):
                    antecedent = [itemset[i]]
                    consequent = [itemset[1-i]]
                    
                    antecedent_support = self.calculate_support(antecedent)
                    if antecedent_support > 0:
                        confidence = itemset_data['support'] / antecedent_support
                        consequent_support = self.calculate_support(consequent)
                        lift = confidence / consequent_support if consequent_support > 0 else 0
                        
                        if confidence >= 0.5:
                            rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': itemset_data['support'],
                                'confidence': confidence,
                                'lift': lift,
                                'count': itemset_data['count']
                            })
        
        return frequent_itemsets, rules
    
    def calculate_support(self, itemset):
        """Calculate support for itemset"""
        count = sum(1 for transaction in self.transactions 
                   if all(item in transaction for item in itemset))
        return count / len(self.transactions) if self.transactions else 0

def generate_pattern_business_insights(rules, domain, primary_entity):
    """Generate domain-specific business insights from patterns"""
    key_findings = []
    recommendations = []
    
    if not rules:
        return key_findings, recommendations
    
    # General findings
    strongest_rule = max(rules, key=lambda x: x['confidence'])
    key_findings.append(f"Strongest pattern: {' + '.join(strongest_rule['antecedent'])} → {' + '.join(strongest_rule['consequent'])} ({strongest_rule['confidence']:.1%} confidence)")
    
    highest_lift = max(rules, key=lambda x: x['lift'])
    key_findings.append(f"Most unexpected association: {' + '.join(highest_lift['antecedent'])} → {' + '.join(highest_lift['consequent'])} (Lift: {highest_lift['lift']:.2f})")
    
    key_findings.append(f"Total significant patterns identified: {len(rules)}")
    
    # Domain-specific recommendations
    if 'customer' in domain.lower():
        recommendations.extend([
            "Use patterns for cross-selling recommendations to customers",
            "Create bundled product offerings based on associations",
            "Target marketing campaigns using pattern insights",
            "Optimize customer experience with predictive suggestions"
        ])
    elif 'diving' in domain.lower():
        recommendations.extend([
            "Recommend complementary certifications based on diver patterns",
            "Create specialty course bundles for popular combinations",
            "Target advanced courses to divers with specific certification patterns",
            "Use patterns to predict diver progression paths"
        ])
    elif 'healthcare' in domain.lower():
        recommendations.extend([
            "Identify common treatment combinations for better care protocols",
            "Use patterns to predict potential complications or comorbidities",
            "Optimize resource allocation based on treatment associations",
            "Create preventive care recommendations using pattern insights"
        ])
    elif 'education' in domain.lower():
        recommendations.extend([
            "Recommend course combinations based on student success patterns",
            "Create curriculum pathways using popular course associations",
            "Target support services to students with specific course patterns",
            "Use patterns to predict student academic outcomes"
        ])
    elif 'sales' in domain.lower():
        recommendations.extend([
            "Create product bundles based on purchase associations",
            "Target upselling opportunities using transaction patterns",
            "Optimize inventory placement based on product associations",
            "Develop cross-selling strategies from pattern insights"
        ])
    else:
        # Generic business recommendations
        recommendations.extend([
            f"Use patterns for {primary_entity} recommendations and suggestions",
            f"Create bundled offerings based on {primary_entity} associations",
            f"Target marketing using discovered {primary_entity} patterns",
            f"Optimize {primary_entity} experience with predictive insights"
        ])
    
    return key_findings, recommendations

class SmartRAGAssistant:
    def __init__(self, df, schema_analysis):
        self.df = df
        self.schema_analysis = schema_analysis
        self.model = genai.GenerativeModel('gemini-2.5-flash-exp')
        self._parsed_fields_cache = {}
    
    def detect_combined_fields(self):
        combined_fields = {}
        text_cols = self.df.select_dtypes(include=['object', 'string']).columns
        
        for col in text_cols:
            sample_values = self.df[col].dropna().head(100).astype(str)
            
            separators_found = []
            common_separators = [',', ';', '|', '/', '&', '+', '-']
            
            for sep in common_separators:
                if any(sep in str(val) for val in sample_values):
                    separators_found.append(sep)
            
            field_indicators = ['history', 'products', 'items', 'tags', 'categories', 'skills', 'interests', 'purchase']
            if any(indicator in col.lower() for indicator in field_indicators) or separators_found:
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
        if column_name in self._parsed_fields_cache:
            return self._parsed_fields_cache[column_name]
        
        if column_name not in self.df.columns:
            return None
        
        all_items = []
        for value in self.df[column_name].dropna():
            if pd.isna(value):
                continue
            items = [item.strip().lower() for item in str(value).split(separator) if item.strip()]
            all_items.extend(items)
        
        if not all_items:
            return None
        
        from collections import Counter
        item_counts = Counter(all_items)
        
        analysis = {
            'total_individual_items': len(all_items),
            'unique_items': len(item_counts),
            'most_common': item_counts.most_common(20),
            'item_frequency': dict(item_counts),
            'coverage_stats': {
                'records_with_data': int(self.df[column_name].count()),
                'average_items_per_record': len(all_items) / max(1, self.df[column_name].count())
            }
        }
        
        self._parsed_fields_cache[column_name] = analysis
        return analysis
    
    def analyze_combination_patterns(self, column_name, separator=','):
        if column_name not in self.df.columns:
            return None
        
        combinations = []
        combination_sizes = []
        
        for value in self.df[column_name].dropna():
            if pd.isna(value) or str(value).strip() == '':
                continue
            
            clean_combo = str(value).strip()
            items = [item.strip().lower() for item in clean_combo.split(separator) if item.strip()]
            if items:
                sorted_items = sorted(items)
                normalized_combo = ','.join(sorted_items)
                combinations.append(normalized_combo)
                combination_sizes.append(len(items))
        
        if not combinations:
            return None
        
        from collections import Counter
        combo_counts = Counter(combinations)
        
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
        combined_fields = self.detect_combined_fields()
        analysis = {}
        
        for field_name, field_info in combined_fields.items():
            parsed_data = self.parse_combined_field(field_name, field_info['separator'])
            if parsed_data:
                combination_patterns = self.analyze_combination_patterns(field_name, field_info['separator'])
                
                analysis[field_name] = {
                    'field_type': field_info['type'],
                    'separator_used': field_info['separator'],
                    'individual_items': {
                        'total_individual_items': parsed_data['total_individual_items'],
                        'unique_items_count': parsed_data['unique_items'],
                        'top_10_individual_items': parsed_data['most_common'][:10],
                        'coverage_stats': parsed_data['coverage_stats']
                    },
                    'combination_patterns': {
                        'customers_with_combinations': combination_patterns['total_customers_with_combinations'] if combination_patterns else 0,
                        'total_unique_combinations': combination_patterns['unique_combinations'] if combination_patterns else 0,
                        'top_10_combinations': combination_patterns['most_common_combinations'][:10] if combination_patterns else [],
                        'avg_items_per_combination': combination_patterns['avg_items_per_combination'] if combination_patterns else 0
                    }
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
        question_lower = question.lower()
        
        product_keywords = ['product', 'item', 'purchase', 'bought', 'buy', 'sold', 'selling', 'offer', 'provide', 'sell']
        ranking_keywords = ['most', 'popular', 'common', 'frequent', 'top', 'best']
        combination_keywords = ['combination', 'combo', 'together', 'bundle', 'pair', 'group', 'set']
        unique_keywords = ['unique', 'different', 'distinct', 'variety', 'types', 'kinds', 'catalog', 'range']
        
        is_product_question = any(keyword in question_lower for keyword in product_keywords)
        is_ranking_question = any(keyword in question_lower for keyword in ranking_keywords)
        is_combination_question = any(keyword in question_lower for keyword in combination_keywords)
        is_unique_question = any(keyword in question_lower for keyword in unique_keywords)
        
        if is_product_question and is_unique_question:
            return self.analyze_unique_products(question)
        
        elif is_product_question and is_ranking_question:
            if is_combination_question:
                return self.analyze_purchase_combinations(question)
            else:
                return self.analyze_product_popularity(question)
        
        return None
    
    def analyze_unique_products(self, question):
        combined_fields = self.detect_combined_fields()
        
        if not combined_fields:
            return {
                "analysis": "I couldn't find any fields that contain product or purchase data in a format that can be analyzed for unique products. The dataset might have product information stored differently.",
                "confidence": 0.3,
                "key_findings": ["No parseable product fields detected"],
                "relevant_statistics": {},
                "actionable_insights": [
                    "Check if product data is stored in separate columns",
                    "Verify the format of product-related fields",
                    "Consider if products are encoded in a different way"
                ],
                "data_evidence": ["Analyzed field structures for product patterns"],
                "confidence_level": "low",
                "follow_up_questions": [
                    "What format is the product data stored in?",
                    "Are there specific product columns I should examine?"
                ]
            }
        
        product_field = None
        for field_name in combined_fields.keys():
            field_lower = field_name.lower()
            if any(indicator in field_lower for indicator in ['purchase', 'product', 'item', 'buy', 'order', 'history']):
                product_field = field_name
                break
        
        if not product_field:
            product_field = list(combined_fields.keys())[0]
        
        parsed_data = self.parse_combined_field(product_field, combined_fields[product_field]['separator'])
        
        if not parsed_data or not parsed_data['most_common']:
            return {
                "analysis": f"I found a potential product field '{product_field}' but couldn't extract meaningful product data from it.",
                "confidence": 0.2,
                "key_findings": ["Product field found but no data extracted"],
                "relevant_statistics": {},
                "actionable_insights": ["Verify the data format in the product field"],
                "data_evidence": [f"Attempted to parse field: {product_field}"],
                "confidence_level": "low",
                "follow_up_questions": ["What does the product data look like in your dataset?"]
            }
        
        unique_products = list(parsed_data['item_frequency'].keys())
        total_unique = len(unique_products)
        top_products = parsed_data['most_common'][:10]  
        
        product_list = [product.title() for product in unique_products]
        top_10_names = [name.title() for name, count in top_products]
        
        analysis = f"Based on analysis of the '{product_field}' field, your business offers {total_unique} unique products. "
        
        if top_products:
            analysis += f"The top 10 most frequently purchased items are: {', '.join(top_10_names[:-1])}, and {top_10_names[-1]}. "
        
        analysis += f"These products appear across {parsed_data['coverage_stats']['records_with_data']:,} customer purchase records, "
        analysis += f"with an average of {parsed_data['coverage_stats']['average_items_per_record']:.1f} items per customer purchase history."
        
        total_purchases = parsed_data['total_individual_items']
        if total_purchases > 0:
            diversity_ratio = (total_unique / total_purchases) * 100
            analysis += f" The product diversity ratio is {diversity_ratio:.1f}%, indicating "
            if diversity_ratio > 20:
                analysis += "high product variety with many unique items."
            elif diversity_ratio > 10:
                analysis += "moderate product variety."
            else:
                analysis += "focused product range with some items being purchased frequently."
        
        key_findings = [
            f"Total unique products offered: {total_unique}",
            f"Most popular product: {top_products[0][0].title()} ({top_products[0][1]:,} purchases)" if top_products else "No purchase data available",
            f"Product catalog spans {parsed_data['coverage_stats']['records_with_data']:,} customer records",
            f"Average items per customer: {parsed_data['coverage_stats']['average_items_per_record']:.1f}"
        ]
        
        relevant_stats = {
            "total_unique_products": total_unique,
            "total_purchase_instances": parsed_data['total_individual_items'],
            "customers_with_purchase_data": parsed_data['coverage_stats']['records_with_data'],
            "most_popular_product_purchases": top_products[0][1] if top_products else 0,
            "avg_products_per_customer": parsed_data['coverage_stats']['average_items_per_record']
        }
        
        actionable_insights = [
            "Consider creating product categories to better organize your diverse catalog",
            f"Focus marketing on the top 5 products: {', '.join([name.title() for name, _ in top_products[:5]])}" if len(top_products) >= 5 else "Analyze top products for marketing focus",
            "Investigate low-performing products for potential discontinuation or promotion",
            "Use purchase patterns to develop product recommendation systems"
        ]
        
        if total_unique > 50:
            actionable_insights.append("Consider implementing product search and filtering due to large catalog size")
        elif total_unique < 10:
            actionable_insights.append("Explore opportunities to expand product offerings")
        
        return {
            "analysis": analysis,
            "confidence": 0.95,
            "key_findings": key_findings,
            "relevant_statistics": relevant_stats,
            "actionable_insights": actionable_insights,
            "data_evidence": [
                f"Parsed {total_unique} unique products from '{product_field}' field",
                f"Analyzed {parsed_data['coverage_stats']['records_with_data']:,} customer purchase records",
                f"Processed {total_purchases:,} total purchase instances"
            ],
            "confidence_level": "high",
            "follow_up_questions": [
                "Which product categories generate the most revenue?",
                "What are the seasonal trends for your top products?",
                "How do product preferences vary across customer segments?"
            ]
        }
    
    def analyze_product_popularity(self, question):
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
        
        product_field = None
        for field_name in combined_fields.keys():
            field_lower = field_name.lower()
            if any(indicator in field_lower for indicator in ['purchase', 'product', 'item', 'buy', 'order']):
                product_field = field_name
                break
        
        if not product_field:
            product_field = list(combined_fields.keys())[0]
        
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
        
        key_findings = []
        if most_popular_product:
            key_findings.append(f"Most popular product: {most_popular_product[0].title()} ({most_popular_product[1]:,} purchases)")
        key_findings.append(f"Total unique products: {parsed_data['unique_items']:,}")
        key_findings.append(f"Total purchase instances: {parsed_data['total_individual_items']:,}")
        key_findings.append(f"Average products per customer: {parsed_data['coverage_stats']['average_items_per_record']:.1f}")
        
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
        
        combination_field = None
        for field_name in combined_fields.keys():
            field_lower = field_name.lower()
            if any(indicator in field_lower for indicator in ['purchase', 'product', 'item', 'buy', 'order', 'history']):
                combination_field = field_name
                break
        
        if not combination_field:
            combination_field = list(combined_fields.keys())[0]
        
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
        
        top_combinations = combination_patterns['top_10_combinations']
        most_common_combo = top_combinations[0] if top_combinations else None
        
        analysis = f"Based on analysis of the '{combination_field}' field, "
        
        if most_common_combo:
            combo_pattern, frequency = most_common_combo
            total_records = combination_patterns['customers_with_combinations']
            percentage = (frequency / total_records) * 100 if total_records > 0 else 0
            
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
        
        key_findings = []
        if most_common_combo:
            combo_display = most_common_combo[0].replace(',', ' + ').title()
            key_findings.append(f"Most common combination: {combo_display} ({most_common_combo[1]:,} occurrences)")
        
        key_findings.extend([
            f"Total unique combinations: {combination_patterns['total_unique_combinations']:,}",
            f"Customers with combination data: {combination_patterns['customers_with_combinations']:,}",
            f"Average combination size: {combination_patterns['avg_items_per_combination']:.1f} items"
        ])
        
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
    
    def answer_question(self, question):
        try:
            specific_analysis = self.handle_specific_question_types(question)
            if specific_analysis:
                return specific_analysis
            
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
def generate_dashboard_insights(df, schema_analysis):
    """Generate comprehensive dashboard insights using Gemini AI"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-exp')
        
        # Create comprehensive data context
        data_context = create_dashboard_context(df, schema_analysis)
        
        # Generate insights using Gemini
        prompt = create_dashboard_prompt(data_context)
        response = model.generate_content(prompt)
        
        # Parse AI response and combine with calculated metrics
        ai_insights = parse_dashboard_response(response.text)
        calculated_metrics = calculate_dashboard_metrics(df, schema_analysis)
        
        # Merge AI insights with calculated data
        dashboard_data = merge_dashboard_data(ai_insights, calculated_metrics, df, schema_analysis)
        
        return dashboard_data
        
    except Exception as e:
        print(f"Gemini dashboard generation failed: {str(e)}")
        # Fallback to rule-based dashboard generation
        return generate_fallback_dashboard(df, schema_analysis)

def create_dashboard_context(df, schema_analysis):
    """Create comprehensive context for dashboard generation"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    text_cols = df.select_dtypes(include=['object', 'string']).columns
    
    # Statistical summary
    statistical_summary = {}
    for col in numeric_cols:
        if not df[col].isna().all():
            try:
                statistical_summary[col] = {
                    "count": int(df[col].count()),
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else 0.0,
                    "std": float(df[col].std()) if not pd.isna(df[col].std()) else 0.0,
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else 0.0,
                    "25%": float(df[col].quantile(0.25)) if not pd.isna(df[col].quantile(0.25)) else 0.0,
                    "50%": float(df[col].median()) if not pd.isna(df[col].median()) else 0.0,
                    "75%": float(df[col].quantile(0.75)) if not pd.isna(df[col].quantile(0.75)) else 0.0,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else 0.0,
                    "sum": float(df[col].sum()) if not pd.isna(df[col].sum()) else 0.0,
                    "unique_count": int(df[col].nunique())
                }
            except Exception as e:
                print(f"Error processing column {col}: {e}")
                continue
    
    # Categorical insights
    categorical_insights = {}
    for col in text_cols:
        unique_count = df[col].nunique()
        if 2 <= unique_count <= 50 or (unique_count / len(df) < 0.2 and unique_count > 1):
            value_counts = df[col].value_counts()
            categorical_insights[col] = {
                "unique_count": int(unique_count),
                "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "top_5_values": {str(k): int(v) for k, v in value_counts.head(5).items()}
            }
    
    # Data quality metrics
    completeness = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
    
    return {
        "metadata": {
            "business_domain": schema_analysis.get('business_domain', 'general business'),
            "primary_entity": schema_analysis.get('primary_entity', 'record'),
            "total_records": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "data_completeness": completeness
        },
        "statistical_summary": statistical_summary,
        "categorical_insights": categorical_insights,
        "sample_data": df.head(5).to_dict('records')
    }

def create_dashboard_prompt(data_context):
    """Create AI prompt for dashboard insights generation"""
    return f"""
You are a Business Intelligence Dashboard Generator. Analyze the provided dataset and generate comprehensive dashboard insights.

BUSINESS CONTEXT:
- Domain: {data_context['metadata']['business_domain']}
- Entity: {data_context['metadata']['primary_entity']} 
- Records: {data_context['metadata']['total_records']:,}
- Columns: {data_context['metadata']['total_columns']}
- Data Quality: {data_context['metadata']['data_completeness']:.1f}% complete

STATISTICAL DATA:
{json.dumps(data_context['statistical_summary'], indent=2)}

CATEGORICAL DATA:
{json.dumps(data_context['categorical_insights'], indent=2)}

SAMPLE RECORDS:
{json.dumps(data_context['sample_data'], indent=2)}

Generate business dashboard insights in JSON format:

{{
    "primaryInsights": [
        "Key insight about business domain and data patterns",
        "Data quality and completeness insight", 
        "Business value and opportunity insight",
        "Domain-specific trend or pattern insight"
    ],
    "actionableInsights": [
        "Specific actionable recommendation based on data",
        "Strategy suggestion for business improvement",
        "Data-driven decision making recommendation"
    ],
    "nextSteps": [
        "Immediate action item based on findings",
        "Long-term strategic recommendation",
        "Data analysis or collection suggestion"
    ]
}}

Focus on:
1. Business domain-specific insights (not just generic data observations)
2. Actionable recommendations that can drive business value
3. Specific patterns or trends visible in the actual data
4. Data quality insights that impact business decisions

Be specific and avoid generic statements. Use actual data values and percentages where relevant.
"""

def parse_dashboard_response(response_text):
    """Parse Gemini response for dashboard insights"""
    try:
        clean_response = response_text.strip()
        if clean_response.startswith('```json'):
            clean_response = clean_response[7:-3]
        elif clean_response.startswith('```'):
            clean_response = clean_response[3:-3]
        
        return json.loads(clean_response)
    except json.JSONDecodeError as e:
        print(f"Failed to parse AI response: {e}")
        return None

def calculate_dashboard_metrics(df, schema_analysis):
    """Calculate key performance metrics for dashboard"""
    metrics = []
    entity = schema_analysis.get('primary_entity', 'record')
    
    # Primary metric: Total records
    metrics.append({
        "number": len(df),
        "title": f"Total {entity.title()}s",
        "description": f"Total number of {entity}s in the dataset"
    })
    
    # Data completeness metric
    completeness = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
    metrics.append({
        "number": round(completeness, 1),
        "title": "Data Completeness",
        "description": "Percentage of non-missing data across all fields"
    })
    
    # Numeric metrics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:3]:  # Top 3 numeric columns
        if df[col].sum() > 0:
            total_value = df[col].sum()
            avg_value = df[col].mean()
            
            metrics.append({
                "number": int(total_value) if total_value == int(total_value) else round(total_value, 2),
                "title": f"Total {col.replace('_', ' ').title()}",
                "description": f"Sum of all {col.lower()} values"
            })
            
            if avg_value != total_value and len(metrics) < 6:
                metrics.append({
                    "number": round(avg_value, 2),
                    "title": f"Average {col.replace('_', ' ').title()}",
                    "description": f"Average {col.lower()} per {entity}"
                })
    
    # Categorical diversity metrics
    categorical_fields = schema_analysis.get('categorical_fields', [])
    for cat_field in categorical_fields[:2]:  # Top 2 categorical fields
        if len(metrics) < 6:
            unique_count = df[cat_field['column']].nunique()
            metrics.append({
                "number": unique_count,
                "title": f"Unique {cat_field['column'].replace('_', ' ').title()}",
                "description": f"Number of distinct {cat_field['column'].lower()} categories"
            })
    
    return metrics[:6]  # Limit to 6 metrics

def generate_quick_stats(df, schema_analysis):
    """Generate quick statistics for dashboard"""
    stats = []
    
    stats.append({
        "key": "Business Domain",
        "value": schema_analysis.get('business_domain', 'general business').title()
    })
    
    stats.append({
        "key": "Primary Entity", 
        "value": schema_analysis.get('primary_entity', 'record').title()
    })
    
    stats.append({
        "key": "Total Records",
        "value": f"{len(df):,}"
    })
    
    completeness = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
    stats.append({
        "key": "Data Completeness",
        "value": f"{completeness:.1f}%"
    })
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:2]:
            if df[col].sum() > 0:
                total = df[col].sum()
                if total > 1000:
                    stats.append({
                        "key": f"Total {col.replace('_', ' ').title()}",
                        "value": f"{total:,.0f}"
                    })
                else:
                    stats.append({
                        "key": f"Total {col.replace('_', ' ').title()}",
                        "value": f"{total:.1f}"
                    })
                break
    
    return stats[:5]  

def generate_analytics_summary(df):
    """Generate analytics summary for numeric columns"""
    analytics = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if not df[col].isna().all():
            try:
                analytics.append({
                    "name": col,
                    "count": int(df[col].count()),
                    "mean": round(float(df[col].mean()), 2) if not pd.isna(df[col].mean()) else 0.0,
                    "std": round(float(df[col].std()), 2) if not pd.isna(df[col].std()) else 0.0,
                    "min": round(float(df[col].min()), 2) if not pd.isna(df[col].min()) else 0.0,
                    "25%": round(float(df[col].quantile(0.25)), 2) if not pd.isna(df[col].quantile(0.25)) else 0.0,
                    "50%": round(float(df[col].median()), 2) if not pd.isna(df[col].median()) else 0.0,
                    "75%": round(float(df[col].quantile(0.75)), 2) if not pd.isna(df[col].quantile(0.75)) else 0.0,
                    "max": round(float(df[col].max()), 2) if not pd.isna(df[col].max()) else 0.0
                })
            except Exception as e:
                print(f"Error processing analytics for column {col}: {e}")
                continue
    
    return analytics

def merge_dashboard_data(ai_insights, calculated_metrics, df, schema_analysis):
    
    if not ai_insights:
        ai_insights = generate_fallback_insights(df, schema_analysis)
    
    return {
        "keyBusinessInsights": {
            "primaryInsights": ai_insights.get("primaryInsights", []),
            "quickStats": generate_quick_stats(df, schema_analysis)
        },
        "keyPerformanceMetrics": calculated_metrics,
        "businessRecommendations": {
            "actionableInsights": ai_insights.get("actionableInsights", []),
            "nextSteps": ai_insights.get("nextSteps", [])
        },
        "analytics": generate_analytics_summary(df)
    }

def generate_fallback_dashboard(df, schema_analysis):
    return generate_fallback_insights_full(df, schema_analysis)

def generate_fallback_insights(df, schema_analysis):
    domain = schema_analysis.get('business_domain', 'general business')
    entity = schema_analysis.get('primary_entity', 'record')
    completeness = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
    
    primary_insights = [
        f"Dataset contains {len(df):,} {entity} records from {domain} domain",
        f"Data quality is {'excellent' if completeness > 95 else 'good' if completeness > 80 else 'fair'} with {completeness:.1f}% completeness"
    ]
    
    if 'customer' in domain.lower():
        primary_insights.append("Customer relationship data detected - suitable for segmentation and retention analysis")
    elif 'sales' in domain.lower():
        primary_insights.append("Sales data structure identified - ready for revenue and performance analysis")
    elif 'healthcare' in domain.lower():
        primary_insights.append("Healthcare data patterns found - appropriate for patient outcome analysis")
    elif 'education' in domain.lower():
        primary_insights.append("Educational data detected - suitable for student performance tracking")
    else:
        primary_insights.append(f"Business data structure optimized for {entity} analysis and reporting")
    
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    if numeric_cols > 0:
        primary_insights.append(f"Contains {numeric_cols} quantitative fields suitable for statistical analysis")
    
    actionable_insights = [
        f"Implement {entity} segmentation based on key categorical fields",
        "Set up automated data quality monitoring for missing value detection",
        "Develop KPI tracking dashboard for key business metrics"
    ]
    
    if 'customer' in domain.lower():
        actionable_insights.append("Create customer lifetime value models and churn prediction analytics")
    elif 'sales' in domain.lower():
        actionable_insights.append("Establish sales forecasting and territory performance optimization")
    elif 'inventory' in domain.lower():
        actionable_insights.append("Implement inventory optimization and demand forecasting systems")
    
    next_steps = [
        "Set up regular data validation and quality checks",
        "Create automated reporting dashboards for key stakeholders",
        "Implement data-driven decision making processes"
    ]
    
    return {
        "primaryInsights": primary_insights[:4],
        "actionableInsights": actionable_insights[:3],
        "nextSteps": next_steps[:3]
    }

def generate_fallback_insights_full(df, schema_analysis):
    """Generate complete fallback dashboard when AI is unavailable"""
    insights = generate_fallback_insights(df, schema_analysis)
    metrics = calculate_dashboard_metrics(df, schema_analysis)
    
    return {
        "keyBusinessInsights": {
            "primaryInsights": insights["primaryInsights"],
            "quickStats": generate_quick_stats(df, schema_analysis)
        },
        "keyPerformanceMetrics": metrics,
        "businessRecommendations": {
            "actionableInsights": insights["actionableInsights"],
            "nextSteps": insights["nextSteps"]
        },
        "analytics": generate_analytics_summary(df)
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
@app.route('/ai/dashboard-data', methods=['POST'])
def dashboard_data():
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
        
        dashboard_insights = generate_dashboard_insights(df, schema_analysis)
        
        return jsonify(dashboard_insights), 200

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }), 500
@app.route('/ai/pattern-analysis-initial', methods=['POST'])
def pattern_analysis_initial():
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

        # Parse CSV file
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

        # Analyze schema
        schema_analyzer = DataSchemaAnalyzer(df)
        schema_analysis = schema_analyzer.analyze_schema()

        # Initialize pattern analyzer
        pattern_analyzer = UniversalMarketBasketAnalyzer(df, schema_analysis)
        
        # Calculate initial metrics
        transactions_found = len(pattern_analyzer.transactions)
        transactional_patterns_detected = transactions_found > 0
        
        avg_items_per_transaction = 0
        if transactions_found > 0:
            avg_items_per_transaction = np.mean([len(t) for t in pattern_analyzer.transactions])

        response_data = {
            "transactionalPatternsDetected": transactional_patterns_detected,
            "data": {
                "transactionsFound": transactions_found,
                "avgItemsPerTransaction": round(avg_items_per_transaction, 1)
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }), 500

@app.route('/ai/pattern-analysis-analyze', methods=['POST'])
def pattern_analysis_analyze():
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

        # Get min_support from form data (default to 0.05)
        min_support = float(request.form.get('min_support', 0.05))

        # Parse CSV file
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

        # Analyze schema
        schema_analyzer = DataSchemaAnalyzer(df)
        schema_analysis = schema_analyzer.analyze_schema()

        # Initialize pattern analyzer
        pattern_analyzer = UniversalMarketBasketAnalyzer(df, schema_analysis)
        
        # Check if patterns can be analyzed
        if not pattern_analyzer.transactions:
            return jsonify({
                "foundPatterns": False,
                "issueIfNoPatternsFound": "No suitable transactional patterns detected in this dataset. Pattern Analysis works best with data that has comma-separated values in text fields, multiple items per record, or categorical associations.",
                "data": {
                    "significantPatternsCount": 0,
                    "topAssociationPatterns": [],
                    "businessInsights": {
                        "keyFindings": ["No transactional data patterns found in the dataset"],
                        "recommendations": [
                            "Ensure data contains comma-separated values or multiple items per record",
                            "Check for fields with categorical associations",
                            "Consider restructuring data to include transactional relationships"
                        ]
                    }
                }
            }), 200

        # Analyze patterns
        frequent_itemsets, association_rules = pattern_analyzer.analyze_patterns(min_support)
        
        if not association_rules:
            issue_message = f"No significant patterns found with minimum support of {min_support*100:.0f}%. Try lowering the minimum support threshold or ensure your data contains meaningful associations."
            
            return jsonify({
                "foundPatterns": False,
                "issueIfNoPatternsFound": issue_message,
                "data": {
                    "significantPatternsCount": 0,
                    "topAssociationPatterns": [],
                    "businessInsights": {
                        "keyFindings": [
                            f"Analyzed {len(pattern_analyzer.transactions)} transactional patterns",
                            f"No patterns met the {min_support*100:.0f}% minimum support threshold"
                        ],
                        "recommendations": [
                            "Lower the minimum support threshold to discover weaker patterns",
                            "Examine data quality and ensure meaningful associations exist",
                            "Consider different data preprocessing approaches"
                        ]
                    }
                }
            }), 200

        # Sort rules by confidence (descending)
        association_rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Format top association patterns
        top_association_patterns = []
        for rule in association_rules[:10]:  # Top 10 patterns
            top_association_patterns.append({
                "whenWeSee": " + ".join(rule['antecedent']),
                "weOftenFind": " + ".join(rule['consequent']),
                "confidence": round(rule['confidence'], 3),
                "lift": round(rule['lift'], 2)
            })

        # Generate business insights
        domain = schema_analysis.get('business_domain', 'general business')
        primary_entity = schema_analysis.get('primary_entity', 'record')
        key_findings, recommendations = generate_pattern_business_insights(association_rules, domain, primary_entity)

        response_data = {
            "foundPatterns": True,
            "issueIfNoPatternsFound": "",
            "data": {
                "significantPatternsCount": len(association_rules),
                "topAssociationPatterns": top_association_patterns,
                "businessInsights": {
                    "keyFindings": key_findings,
                    "recommendations": recommendations
                }
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
