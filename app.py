import os
import pandas as pd
import numpy as np
import json
import re
import anthropic
from datetime import datetime
from flask import Flask, request, jsonify
from collections import defaultdict
from dotenv import load_dotenv
from itertools import combinations
from functools import lru_cache
import time
app = Flask(__name__)
load_dotenv()
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")  
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")  # Update message

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) 
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
        self.item_to_index = {}  # For efficient item indexing
        self.index_to_item = {}  # For reverse lookup
        self._build_item_index()
    
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
    
    def _build_item_index(self):
        """Build efficient item indexing for large datasets"""
        if not self.transactions:
            return
        
        all_items = set()
        for transaction in self.transactions:
            all_items.update(transaction)
        
        # Create bidirectional mapping
        for idx, item in enumerate(sorted(all_items)):
            self.item_to_index[item] = idx
            self.index_to_item[idx] = item
    
    def analyze_patterns(self, min_support=0.05, min_confidence=0.5, max_itemset_size=3):
        """Analyze universal patterns using efficient Apriori algorithm with optimizations"""
        if not self.transactions:
            return [], []
        
        start_time = time.time()
        
        # Convert transactions to frozensets for efficient operations
        transaction_sets = [frozenset(transaction) for transaction in self.transactions]
        total_transactions = len(transaction_sets)
        
        # Get all unique items
        all_items = set()
        for transaction in transaction_sets:
            all_items.update(transaction)
        
        # Step 1: Generate frequent 1-itemsets with vectorized counting
        frequent_itemsets = {}
        
        # Use vectorized approach for 1-itemset counting
        item_counts = self._count_items_vectorized(transaction_sets, all_items)
        
        for item, count in item_counts.items():
            support = count / total_transactions
            if support >= min_support:
                itemset = frozenset([item])
                frequent_itemsets[itemset] = {
                    'itemset': list(itemset),
                    'support': support,
                    'count': count
                }
        
        # Step 2: Generate frequent k-itemsets using Apriori principle
        k = 2
        while k <= max_itemset_size and frequent_itemsets:
            # Generate candidates for k-itemsets
            candidates = self._generate_candidates(frequent_itemsets, k)
            
            if not candidates:
                break
            
            # Count support for candidates with optimized subset checking
            candidate_counts = self._count_candidates_vectorized(
                candidates, transaction_sets, total_transactions, min_support
            )
            
            # Update frequent itemsets
            frequent_itemsets = candidate_counts
            k += 1
        
        # Step 3: Generate association rules from all frequent itemsets
        rules = self._generate_association_rules(frequent_itemsets, min_confidence, total_transactions)
        
        # Convert back to list format for compatibility
        frequent_itemsets_list = list(frequent_itemsets.values())
        
        end_time = time.time()
        print(f"Pattern analysis completed in {end_time - start_time:.2f} seconds")
        
        return frequent_itemsets_list, rules
    
    def _count_items_vectorized(self, transaction_sets, all_items):
        """Vectorized counting of 1-itemsets with optimization for large datasets"""
        item_counts = {}
        
        # For very large datasets, use bit vector approach
        if len(transaction_sets) > 10000 and len(all_items) > 100:
            return self._count_items_bit_vector(transaction_sets, all_items)
        
        # Standard approach for smaller datasets
        for item in all_items:
            count = sum(1 for transaction in transaction_sets if item in transaction)
            item_counts[item] = count
        return item_counts
    
    def _count_items_bit_vector(self, transaction_sets, all_items):
        """Bit vector approach for very large datasets"""
        item_counts = {}
        
        # Create bit vectors for each item
        for item in all_items:
            count = 0
            for transaction in transaction_sets:
                if item in transaction:
                    count += 1
            item_counts[item] = count
        
        return item_counts
    
    def _count_candidates_vectorized(self, candidates, transaction_sets, total_transactions, min_support):
        """Optimized counting of candidate itemsets"""
        candidate_counts = {}
        
        # Group candidates by size for batch processing
        candidates_by_size = defaultdict(list)
        for candidate in candidates:
            candidates_by_size[len(candidate)].append(candidate)
        
        for size, size_candidates in candidates_by_size.items():
            for candidate in size_candidates:
                count = sum(1 for transaction in transaction_sets 
                           if candidate.issubset(transaction))
                support = count / total_transactions
                if support >= min_support:
                    candidate_counts[candidate] = {
                        'itemset': list(candidate),
                        'support': support,
                        'count': count
                    }
        
        return candidate_counts
    
    def _generate_candidates(self, frequent_itemsets, k):
        """Generate candidate k-itemsets using Apriori principle"""
        candidates = set()
        
        # Get all frequent (k-1)-itemsets
        frequent_k_minus_1 = list(frequent_itemsets.keys())
        
        # Generate candidates by joining frequent (k-1)-itemsets
        for i in range(len(frequent_k_minus_1)):
            for j in range(i + 1, len(frequent_k_minus_1)):
                itemset1 = frequent_k_minus_1[i]
                itemset2 = frequent_k_minus_1[j]
                
                # Join if they share (k-2) items
                union = itemset1.union(itemset2)
                if len(union) == k:
                    # Prune: check if all (k-1)-subsets are frequent
                    if self._has_infrequent_subset(union, frequent_k_minus_1):
                        continue
                    candidates.add(union)
        
        return candidates
    
    def _has_infrequent_subset(self, candidate, frequent_itemsets):
        """Check if candidate has any infrequent subset (pruning step)"""
        for item in candidate:
            subset = candidate - {item}
            if subset not in frequent_itemsets:
                return True
        return False
    
    def _generate_association_rules(self, frequent_itemsets, min_confidence, total_transactions):
        """Generate association rules from frequent itemsets"""
        rules = []
        
        for itemset, itemset_data in frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            
            # Generate all possible rules from this itemset
            itemset_list = list(itemset)
            for i in range(1, len(itemset_list)):
                # Generate all combinations of antecedents
                for antecedent_combo in combinations(itemset_list, i):
                    antecedent = frozenset(antecedent_combo)
                    consequent = itemset - antecedent
                    
                    if not consequent:
                        continue
                    
                    # Calculate confidence
                    antecedent_support = self._get_itemset_support_from_frequent(antecedent, frequent_itemsets, total_transactions)
                    if antecedent_support > 0:
                        confidence = itemset_data['support'] / antecedent_support
                        
                        if confidence >= min_confidence:
                            # Calculate lift
                            consequent_support = self._get_itemset_support_from_frequent(consequent, frequent_itemsets, total_transactions)
                            lift = confidence / consequent_support if consequent_support > 0 else 0
                            
                            rules.append({
                                'antecedent': list(antecedent),
                                'consequent': list(consequent),
                                'support': itemset_data['support'],
                                'confidence': confidence,
                                'lift': lift,
                                'count': itemset_data['count']
                            })
        
        return rules
    
    @lru_cache(maxsize=1000)
    def _get_itemset_support(self, itemset_tuple, total_transactions):
        """Get support for an itemset with caching for performance"""
        itemset = set(itemset_tuple)
        
        # If not in frequent itemsets, calculate directly (for consequents)
        count = sum(1 for transaction in self.transactions 
                   if all(item in transaction for item in itemset))
        return count / total_transactions if total_transactions > 0 else 0
    
    def _get_itemset_support_from_frequent(self, itemset, frequent_itemsets, total_transactions):
        """Get support for an itemset from frequent itemsets or calculate if not found"""
        if itemset in frequent_itemsets:
            return frequent_itemsets[itemset]['support']
        
        # Use cached calculation for non-frequent itemsets
        return self._get_itemset_support(tuple(sorted(itemset)), total_transactions)
    
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
        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
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
            
            field_indicators = ['history', 'products', 'items', 'tags', 'categories', 'skills', 'interests', 'purchase', 'course', 'courses', 'specialty', 'specialties', 'certification', 'certifications', 'class', 'classes', 'training', 'lesson', 'lessons']
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
        
        product_keywords = ['product', 'item', 'purchase', 'purchased', 'bought', 'buy', 'sold', 'selling', 'offer', 'provide', 'sell', 'course', 'courses', 'specialty', 'specialties', 'certification', 'certifications', 'class', 'classes', 'training', 'lesson', 'lessons', 'program', 'module']
        ranking_keywords = ['most', 'popular', 'common', 'frequent', 'top', 'best']
        combination_keywords = ['combination', 'combo', 'together', 'bundle', 'pair', 'group', 'set']
        unique_keywords = ['unique', 'different', 'distinct', 'variety', 'types', 'kinds', 'catalog', 'range', 'offered', 'available']
        
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
        
        product_field = self.select_best_combined_field(combined_fields, question)
        
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
        
        product_field = self.select_best_combined_field(combined_fields, question)
        
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
        
        combination_field = self.select_best_combined_field(combined_fields, question)
        
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
    
    def select_best_combined_field(self, combined_fields, question):
        """Select the most relevant combined field based on the question and data richness.
        Prefers course-like columns when the question mentions courses, certifications, or specialties.
        Otherwise uses product/purchase indicators and falls back to the richest field.
        """
        if not combined_fields:
            return None
        q = (question or '').lower()
        priority_keywords = [
            # Highest priority: course-related
            ('courses', 5), ('course', 5), ('specialties', 5), ('specialty', 5),
            ('certifications', 5), ('certification', 5), ('class', 4), ('classes', 4),
            ('training', 4), ('lesson', 3), ('lessons', 3),
            # Product-related
            ('purchase', 3), ('purchased', 3), ('product', 3), ('item', 3), ('buy', 3), ('order', 3), ('history', 2)
        ]
        # Score fields by name relevance
        field_scores = {}
        for field_name in combined_fields.keys():
            score = 0
            lower = field_name.lower()
            for kw, pts in priority_keywords:
                if kw in q and kw.rstrip('s') in lower:
                    score += pts
                if kw in lower:
                    score += 1  # mild boost if field name contains keyword
            field_scores[field_name] = score
        # If any field has a positive score, pick the highest
        best_field = max(field_scores, key=lambda k: field_scores[k]) if field_scores else None
        if best_field and field_scores[best_field] > 0:
            return best_field
        
        # Otherwise, pick the richest field by number of parsed individual items
        richest_field = None
        richest_count = -1
        for field_name, info in combined_fields.items():
            parsed = self.parse_combined_field(field_name, info.get('separator', ','))
            count = parsed['total_individual_items'] if parsed else 0
            if count > richest_count:
                richest_count = count
                richest_field = field_name
        return richest_field or list(combined_fields.keys())[0]
    
    def answer_question(self, question):
        try:
            specific_analysis = self.handle_specific_question_types(question)
            if specific_analysis:
                return specific_analysis
            
            data_context = self.create_data_context()
            prompt = self.create_rag_prompt(question, data_context)
            
            # Updated Claude API call
            response = self.client.messages.create(
                model="claude-3-5-haiku-latest",  # Use Claude Sonnet 3.5
                max_tokens=5000,
                temperature=0.7,
                system="You are a Smart Business Intelligence Assistant.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            parsed_response = self.parse_rag_response(response.content[0].text, question)
            return parsed_response
            
        except anthropic.RateLimitError:
            return self.fallback_answer(question, "Claude rate limit exceeded")
        except anthropic.AuthenticationError:
            return self.fallback_answer(question, "Claude authentication failed")
        except anthropic.BadRequestError as e:
            return self.fallback_answer(question, f"Invalid Claude request: {str(e)}")
        except anthropic.APIConnectionError:
            return self.fallback_answer(question, "Failed to connect to Claude API")
        except anthropic.APIError as e:
            return self.fallback_answer(question, f"Claude API error: {str(e)}")
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
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Create comprehensive data context
        data_context = create_dashboard_context(df, schema_analysis)
        
        # Generate insights using Claude
        prompt = create_dashboard_prompt(data_context)
        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=5000,
            temperature=0.7,
            system="You are a Business Intelligence Dashboard Generator.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse Claude response and combine with calculated metrics
        ai_insights = parse_dashboard_response(response.content[0].text)
        calculated_metrics = calculate_dashboard_metrics(df, schema_analysis)
        
        # Merge AI insights with calculated data
        dashboard_data = merge_dashboard_data(ai_insights, calculated_metrics, df, schema_analysis)
        
        return dashboard_data
        
    except anthropic.RateLimitError:
        print("Claude rate limit exceeded for dashboard generation")
        return generate_fallback_dashboard(df, schema_analysis)
    except anthropic.AuthenticationError:
        print("Claude authentication failed for dashboard generation")
        return generate_fallback_dashboard(df, schema_analysis)
    except anthropic.BadRequestError as e:
        print(f"Invalid Claude request for dashboard: {str(e)}")
        return generate_fallback_dashboard(df, schema_analysis)
    except anthropic.APIConnectionError:
        print("Failed to connect to Claude API for dashboard generation")
        return generate_fallback_dashboard(df, schema_analysis)
    except anthropic.APIError as e:
        print(f"Claude API error for dashboard: {str(e)}")
        return generate_fallback_dashboard(df, schema_analysis)
    except Exception as e:
        print(f"Claude dashboard generation failed: {str(e)}")
        return generate_fallback_dashboard(df, schema_analysis)

def filter_meaningful_columns(df):
    """Filter out non-meaningful columns like IDs, indices, and other non-business columns"""
    
    # Common patterns for non-meaningful columns
    id_patterns = [
        # Basic ID patterns
        r'^id$', r'^ID$', r'^Id$',
        r'^.*_id$', r'^.*_ID$', r'^.*_Id$',
        r'^id_.*$', r'^ID_.*$', r'^Id_.*$',
        
        # ID patterns with spaces
        r'^.*\s+id$', r'^.*\s+ID$', r'^.*\s+Id$',
        r'^id\s+.*$', r'^ID\s+.*$', r'^Id\s+.*$',
        
        # Master ID patterns (like "SSI Master ID")
        r'^.*\s+master\s+id$', r'^.*\s+Master\s+ID$', r'^.*\s+MASTER\s+ID$',
        r'^master\s+id\s+.*$', r'^Master\s+ID\s+.*$', r'^MASTER\s+ID\s+.*$',
        
        # Other common patterns
        r'^index$', r'^Index$', r'^INDEX$',
        r'^row$', r'^Row$', r'^ROW$',
        r'^rowid$', r'^RowID$', r'^ROWID$',
        r'^key$', r'^Key$', r'^KEY$',
        r'^primary_key$', r'^Primary_Key$', r'^PRIMARY_KEY$',
        r'^primary\s+key$', r'^Primary\s+Key$', r'^PRIMARY\s+KEY$',
        r'^uuid$', r'^UUID$', r'^Uuid$',
        r'^guid$', r'^GUID$', r'^Guid$',
        r'^hash$', r'^Hash$', r'^HASH$',
        r'^token$', r'^Token$', r'^TOKEN$',
        r'^reference$', r'^Reference$', r'^REFERENCE$',
        r'^ref$', r'^Ref$', r'^REF$',
        r'^code$', r'^Code$', r'^CODE$',
        r'^number$', r'^Number$', r'^NUMBER$',
        r'^no$', r'^No$', r'^NO$',
        r'^num$', r'^Num$', r'^NUM$',
        r'^seq$', r'^Seq$', r'^SEQ$',
        r'^serial$', r'^Serial$', r'^SERIAL$'
    ]
    
    # Columns to exclude based on patterns
    excluded_cols = set()
    
    for col in df.columns:
        col_lower = col.lower().strip()
        
        # Check against ID patterns
        for pattern in id_patterns:
            if re.match(pattern, col, re.IGNORECASE):
                excluded_cols.add(col)
                break
        
        # Additional heuristics for numeric columns that look like IDs
        if df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            # Check if column has high uniqueness (like an ID)
            uniqueness_ratio = df[col].nunique() / len(df)
            
            # If it's highly unique and has sequential-like values, likely an ID
            if uniqueness_ratio > 0.95:
                # Check if values are mostly sequential
                if len(df) > 10:
                    sorted_values = df[col].dropna().sort_values()
                    if len(sorted_values) > 1:
                        # Check if values are mostly sequential
                        diffs = sorted_values.diff().dropna()
                        if len(diffs) > 0:
                            # If most differences are 1 (sequential), it's likely an ID
                            sequential_ratio = (diffs == 1).sum() / len(diffs)
                            if sequential_ratio > 0.8:
                                excluded_cols.add(col)
                                continue
            
            # Check if column name suggests it's an ID (handle spaces)
            col_normalized = col_lower.replace(' ', '_').replace('-', '_')
            id_keywords = ['id', 'key', 'index', 'row', 'serial', 'number', 'no', 'num', 'seq', 'master_id', 'masterid']
            
            if any(keyword in col_normalized for keyword in id_keywords):
                # But only exclude if it's highly unique
                if uniqueness_ratio > 0.9:
                    excluded_cols.add(col)
    
    # Return meaningful columns
    meaningful_cols = [col for col in df.columns if col not in excluded_cols]
    
    # Ensure we don't exclude all columns
    if len(meaningful_cols) == 0:
        print("Warning: All columns were filtered out. Using original columns.")
        return list(df.columns)
    
    # Log excluded columns for debugging
    if excluded_cols:
        print(f"Excluded non-meaningful columns: {list(excluded_cols)}")
        print(f"Using {len(meaningful_cols)} meaningful columns: {meaningful_cols}")
    
    return meaningful_cols

def create_dashboard_context(df, schema_analysis):
    """Create comprehensive context for dashboard generation"""
    # Filter out non-meaningful columns (IDs, indices, etc.)
    meaningful_cols = filter_meaningful_columns(df)
    
    numeric_cols = df[meaningful_cols].select_dtypes(include=[np.number]).columns
    text_cols = df[meaningful_cols].select_dtypes(include=['object', 'string']).columns
    
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
    return f"""You are a Senior Business Intelligence Consultant creating executive dashboard insights. Your analysis will directly inform strategic business decisions.

## EXECUTIVE SUMMARY
**Business Domain**: {data_context['metadata']['business_domain']}
**Primary Business Entity**: {data_context['metadata']['primary_entity']}
**Data Scale**: {data_context['metadata']['total_records']:,} records
**Data Reliability**: {data_context['metadata']['data_completeness']:.1f}% complete

## PERFORMANCE METRICS
{json.dumps(data_context['statistical_summary'], indent=2) if data_context['statistical_summary'] else "No quantitative metrics available"}

## BUSINESS SEGMENTS
{json.dumps(data_context['categorical_insights'], indent=2) if data_context['categorical_insights'] else "No segmentation data available"}

## OPERATIONAL DATA SAMPLE
```json
{json.dumps(data_context['sample_data'][:5], indent=2)}
```

---

## DASHBOARD REQUIREMENTS

Generate strategic business insights formatted as JSON:

```json
{{
    "primaryInsights": [
        "Executive-level insight about business performance and opportunities",
        "Data quality insight affecting decision-making reliability", 
        "Market positioning or competitive advantage insight from the data",
        "Operational efficiency or resource optimization finding",
        "Risk assessment or business threat identification"
    ],
    "actionableInsights": [
        "Immediate strategic action with clear ROI potential",
        "Operational improvement with measurable impact",
        "Market opportunity with specific next steps",
        "Risk mitigation strategy based on data patterns"
    ],
    "nextSteps": [
        "Priority action item for the next 30 days",
        "Strategic initiative for next quarter",
        "Data collection or analysis improvement needed",
        "Process optimization recommendation"
    ]
}}
```

## ANALYSIS FRAMEWORK
1. **Strategic Impact**: Focus on insights that affect revenue, costs, or competitive position
2. **Measurable Outcomes**: Quantify opportunities and risks where possible
3. **Actionability**: Every insight should connect to specific business actions
4. **Executive Perspective**: Frame findings for C-level decision makers
5. **Data-Driven**: Ground recommendations in actual data patterns

## BUSINESS CONTEXT REQUIREMENTS
- Address domain-specific KPIs and success metrics
- Consider industry benchmarks and best practices
- Identify growth opportunities and efficiency gains
- Highlight data-driven competitive advantages
- Assess operational risks and mitigation strategies

Generate insights that transform raw data into strategic business intelligence.
"""


def parse_dashboard_response(response_text):
    try:
        if not response_text:
            return {}

        clean_response = response_text.strip()

        # Prefer JSON inside fenced code blocks if present
        fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean_response, re.IGNORECASE)
        if fenced_match:
            candidate = fenced_match.group(1).strip()
        else:
            candidate = clean_response
            # If there is leading prose before JSON, locate the first JSON-looking start
            first_obj = candidate.find('{')
            first_arr = candidate.find('[')
            starts = [s for s in [first_obj, first_arr] if s != -1]
            if starts:
                start_idx = min(starts)
                candidate = candidate[start_idx:]

                # Extract a balanced JSON object/array to avoid trailing prose
                def extract_balanced(text, open_ch, close_ch):
                    depth = 0
                    for idx, ch in enumerate(text):
                        if ch == open_ch:
                            depth += 1
                        elif ch == close_ch:
                            depth -= 1
                            if depth == 0:
                                return text[:idx+1]
                    return text

                if candidate.startswith('{'):
                    candidate = extract_balanced(candidate, '{', '}')
                elif candidate.startswith('['):
                    candidate = extract_balanced(candidate, '[', ']')

        candidate = candidate.strip().strip('`')
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        print(f"Failed to parse AI response: {e}")
        # Return empty dict so caller can gracefully fall back
        return {}

def calculate_dashboard_metrics(df, schema_analysis):
    """Calculate key performance metrics for dashboard"""
    metrics = []
    entity = schema_analysis.get('primary_entity', 'record')
    
    # Filter out non-meaningful columns
    meaningful_cols = filter_meaningful_columns(df)
    
    # Primary metric: Total records
    metrics.append({
        "number": len(df),
        "title": f"Total {entity.title()}s",
        "description": f"Total number of {entity}s in the dataset"
    })
    
    # Data completeness metric (using meaningful columns only)
    meaningful_df = df[meaningful_cols]
    completeness = (meaningful_df.notna().sum().sum() / (len(meaningful_df) * len(meaningful_df.columns))) * 100
    metrics.append({
        "number": round(completeness, 1),
        "title": "Data Completeness",
        "description": "Percentage of non-missing data across meaningful fields"
    })
    
    # Numeric metrics (using meaningful columns only)
    numeric_cols = meaningful_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:3]:  # Top 3 numeric columns
        if meaningful_df[col].sum() > 0:
            total_value = meaningful_df[col].sum()
            avg_value = meaningful_df[col].mean()
            
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
    
    # Filter out non-meaningful columns
    meaningful_cols = filter_meaningful_columns(df)
    meaningful_df = df[meaningful_cols]
    
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
    
    completeness = (meaningful_df.notna().sum().sum() / (len(meaningful_df) * len(meaningful_df.columns))) * 100
    stats.append({
        "key": "Data Completeness",
        "value": f"{completeness:.1f}%"
    })
    
    numeric_cols = meaningful_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols[:2]:
            if meaningful_df[col].sum() > 0:
                total = meaningful_df[col].sum()
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
    
    # Filter out non-meaningful columns
    meaningful_cols = filter_meaningful_columns(df)
    meaningful_df = df[meaningful_cols]
    
    numeric_cols = meaningful_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if not meaningful_df[col].isna().all():
            try:
                analytics.append({
                    "name": col,
                    "count": int(meaningful_df[col].count()),
                    "mean": round(float(meaningful_df[col].mean()), 2) if not pd.isna(meaningful_df[col].mean()) else 0.0,
                    "std": round(float(meaningful_df[col].std()), 2) if not pd.isna(meaningful_df[col].std()) else 0.0,
                    "min": round(float(meaningful_df[col].min()), 2) if not pd.isna(meaningful_df[col].min()) else 0.0,
                    "25%": round(float(meaningful_df[col].quantile(0.25)), 2) if not pd.isna(meaningful_df[col].quantile(0.25)) else 0.0,
                    "50%": round(float(meaningful_df[col].median()), 2) if not pd.isna(meaningful_df[col].median()) else 0.0,
                    "75%": round(float(meaningful_df[col].quantile(0.75)), 2) if not pd.isna(meaningful_df[col].quantile(0.75)) else 0.0,
                    "max": round(float(meaningful_df[col].max()), 2) if not pd.isna(meaningful_df[col].max()) else 0.0
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
        "analytics": generate_analytics_summary(df),
        **build_dashboard_charts(df, schema_analysis)
    }

def generate_fallback_dashboard(df, schema_analysis):
    return generate_fallback_insights_full(df, schema_analysis)

def generate_fallback_insights(df, schema_analysis):
    domain = schema_analysis.get('business_domain', 'general business')
    entity = schema_analysis.get('primary_entity', 'record')
    
    # Use meaningful columns for completeness calculation
    meaningful_cols = filter_meaningful_columns(df)
    meaningful_df = df[meaningful_cols]
    completeness = (meaningful_df.notna().sum().sum() / (len(meaningful_df) * len(meaningful_df.columns))) * 100
    
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
    
    numeric_cols = len(meaningful_df.select_dtypes(include=[np.number]).columns)
    if numeric_cols > 0:
        primary_insights.append(f"Contains {numeric_cols} meaningful quantitative fields suitable for statistical analysis")
    
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
    
    base = {
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
    # Append charts to the fallback payload as well
    base.update(build_dashboard_charts(df, schema_analysis))
    return base

def build_dashboard_charts(df, schema_analysis):
    """Create line, bar, pie, and donut chart data with sensible defaults.

    Ensures each chart uses a different column. If there are not enough distinct
    columns to build all charts, some charts will be disabled ("false").

    Returns a dict with keys expected by the frontend: lineChart, lineChartData, barChart, barChartData,
    pieChart, pieChartData, donutChart, donutChartData. String booleans per spec.
    """
    try:
        # Use meaningful columns to avoid IDs and similar
        meaningful_cols = filter_meaningful_columns(df)
        working_df = df[meaningful_cols]

        # Identify candidate columns
        numeric_cols = list(working_df.select_dtypes(include=[np.number]).columns)
        text_cols = list(working_df.select_dtypes(include=['object', 'string']).columns)

        # Exclude unhelpful text columns for categorical charts (phones/emails/names etc.)
        unhelpful_text_keywords = [
            'phone', 'mobile', 'telephone', 'tel', 'whatsapp', 'fax',
            'contact', 'contact_no', 'contact_number', 'email', 'e-mail',
            # Names
            'name', 'first_name', 'firstname', 'last_name', 'lastname', 'full_name', 'fullname',
            'givenname', 'surename', 'surname', 'family name', 'arabic name',
            # Other often-unhelpful free-text PII or boilerplate
            'address', 'notes', 'note', 'comment', 'comments', 'description', 'remark', 'remarks'
        ]
        filtered_text_cols = []
        for col in text_cols:
            lower = col.lower().strip()
            if any(k in lower for k in unhelpful_text_keywords):
                continue
            # Skip extremely high uniqueness text columns (>95%) and very sparse columns (<10% non-null)
            try:
                total = len(working_df)
                non_null = working_df[col].notna().sum()
                if total > 0 and (non_null / total) < 0.1:
                    continue
                uniq_ratio = working_df[col].nunique(dropna=True) / total if total else 1
                if uniq_ratio > 0.95:
                    continue
                # Skip columns dominated by a single value (>90%)
                vc_preview = (
                    working_df[col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                vc_preview = vc_preview[vc_preview != ""]
                if len(vc_preview) > 0:
                    counts = vc_preview.value_counts()
                    if counts.sum() > 0 and (counts.iloc[0] / counts.sum()) > 0.9:
                        continue
            except Exception:
                pass
            filtered_text_cols.append(col)

        # Helper: choose categorical column with reasonable cardinality
        def pick_categorical(max_unique=20):
            best_col = None
            best_unique = None
            for col in text_cols:
                try:
                    uniq = working_df[col].nunique(dropna=True)
                except Exception:
                    continue
                if 2 <= uniq <= max_unique:
                    if best_unique is None or uniq < best_unique:
                        best_col = col
                        best_unique = uniq
            # Fallback: pick the text col with the lowest unique count > 1
            if not best_col:
                low_col = None
                low_uniq = None
                for col in text_cols:
                    try:
                        uniq = working_df[col].nunique(dropna=True)
                    except Exception:
                        continue
                    if uniq > 1 and (low_uniq is None or uniq < low_uniq):
                        low_col = col
                        low_uniq = uniq
                return low_col
            return best_col

        # Helper: attempt to pick a date-like column
        def pick_date_col():
            date_hints = [c for c in working_df.columns if any(h in c.lower() for h in ["date", "time", "day", "month", "year"])]
            for col in date_hints:
                try:
                    parsed = pd.to_datetime(working_df[col], errors='coerce')
                    if parsed.notna().sum() >= max(5, int(0.1 * len(working_df))):
                        return col
                except Exception:
                    continue
            return None

        # Colors palette
        palette = [
            "#8884d8", "#82ca9d", "#ffc658", "#ff8042", "#8dd1e1",
            "#a4de6c", "#d0ed57", "#d88484", "#84d8c6", "#c6a4de"
        ]

        used_columns = set()

        # Build Line from date grouping or numeric histogram (unique source)
        line_enabled = False
        line_title = "Trend"
        line_data = []
        date_col = pick_date_col()
        if date_col is not None:
            try:
                parsed = pd.to_datetime(working_df[date_col], errors='coerce')
                grp = parsed.dropna().dt.to_period('M').value_counts().sort_index()
                if len(grp) < 3:
                    grp = parsed.dropna().dt.date
                    grp = pd.Series(grp).value_counts().sort_index()
                line_data = [{"name": str(idx), "value": int(val)} for idx, val in list(grp.items())[:20]]
                if line_data:
                    line_enabled = True
                    line_title = date_col
                    used_columns.add(date_col)
            except Exception:
                pass
        if not line_enabled and numeric_cols:
            # Histogram of first numeric column not used
            hist_col = next((c for c in numeric_cols if c not in used_columns), None)
            if hist_col is not None:
                series = working_df[hist_col].dropna()
                if not series.empty:
                    try:
                        counts, bins = np.histogram(series, bins=10)
                        for i in range(min(10, len(counts))):
                            left = bins[i]
                            right = bins[i+1]
                            name = f"{round(float(left), 2)}–{round(float(right), 2)}"
                            line_data.append({"name": name, "value": int(counts[i])})
                        if line_data:
                            line_enabled = True
                            line_title = hist_col
                            used_columns.add(hist_col)
                    except Exception:
                        pass

        # Build distinct categorical charts (bar, pie, donut) using different columns
        def build_cat_data(col_name):
            series = working_df[col_name]
            series = series.dropna()
            series = series.astype(str).str.strip()
            series = series[series != ""]
            # Guard against 'nan' strings from previous conversions
            series = series[series.str.lower() != 'nan']
            vc = series.value_counts().head(10)
            return [{"name": str(k), "value": int(v)} for k, v in vc.items()]

        # Gather up to 3 distinct categorical columns
        categorical_choices = []
        # Prefer reasonable cardinality first from filtered set
        for col in filtered_text_cols:
            try:
                uniq = working_df[col].nunique(dropna=True)
            except Exception:
                continue
            if 2 <= uniq <= 15:
                categorical_choices.append((col, uniq))
        # If too few, add additional lowest-uniqueness columns (>1)
        if len(categorical_choices) < 3:
            extras = []
            for col in filtered_text_cols:
                if any(col == c for c, _ in categorical_choices):
                    continue
                try:
                    uniq = working_df[col].nunique(dropna=True)
                except Exception:
                    continue
                if uniq > 1:
                    extras.append((col, uniq))
            # Sort extras by uniqueness ascending
            extras.sort(key=lambda x: x[1])
            categorical_choices.extend(extras[: max(0, 3 - len(categorical_choices))])

        # Filter out any already used columns
        categorical_choices = [c for c in categorical_choices if c[0] not in used_columns]

        # Assign unique columns to bar, pie, donut in order
        bar_enabled = False
        bar_title = "Category"
        bar_data = []
        if categorical_choices:
            bar_col = categorical_choices.pop(0)[0]
            try:
                bar_data = build_cat_data(bar_col)
                if bar_data:
                    bar_enabled = True
                    bar_title = bar_col
                    used_columns.add(bar_col)
            except Exception:
                pass

        pie_enabled = False
        pie_title = "Category Share"
        pie_data = []
        if categorical_choices:
            pie_col = categorical_choices.pop(0)[0]
            try:
                pie_data = build_cat_data(pie_col)
                if pie_data:
                    pie_enabled = True
                    pie_title = pie_col
                    used_columns.add(pie_col)
            except Exception:
                pass

        donut_enabled = False
        donut_title = "Category Share"
        donut_data = []
        if categorical_choices:
            donut_col = categorical_choices.pop(0)[0]
            try:
                donut_data = build_cat_data(donut_col)
                if donut_data:
                    donut_enabled = True
                    donut_title = donut_col
                    used_columns.add(donut_col)
            except Exception:
                pass

        # Assemble outputs with required string booleans, disable when not possible
        charts = {
            "lineChart": "true" if line_enabled else "false",
            "lineChartData": {
                "title": line_title,
                "data": line_data if line_data else ([{"name": "No Data", "value": 0}])
            },
            "barChart": "true" if bar_enabled else "false",
            "barChartData": {
                "title": bar_title,
                "data": bar_data if bar_data else ([{"name": "No Data", "value": 0}])
            },
            "pieChart": "true" if pie_enabled else "false",
            "pieChartData": {
                "title": pie_title,
                "colorCodes": palette[:max(1, len(pie_data))] if pie_data else palette[:1],
                "data": pie_data if pie_data else ([{"name": "No Data", "value": 1}])
            },
            "donutChart": "true" if donut_enabled else "false",
            "donutChartData": {
                "title": donut_title,
                "colorCodes": palette[:max(1, len(donut_data))] if donut_data else palette[:1],
                "data": donut_data if donut_data else ([{"name": "No Data", "value": 1}])
            }
        }

        return charts
    except Exception:
        # Robust fallback structure
        return {
            "lineChart": "true",
            "lineChartData": {"title": "Trend", "data": [{"name": "No Data", "value": 0}]},
            "barChart": "true",
            "barChartData": {"title": "Category", "data": [{"name": "No Data", "value": 0}]},
            "pieChart": "true",
            "pieChartData": {"title": "Category Share", "colorCodes": ["#8884d8"], "data": [{"name": "No Data", "value": 1}]},
            "donutChart": "true",
            "donutChartData": {"title": "Category Share", "colorCodes": ["#8884d8"], "data": [{"name": "No Data", "value": 1}]}
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

        # Use only meaningful columns (exclude IDs and similar) to avoid unhelpful questions
        meaningful_cols = filter_meaningful_columns(df)
        # Additional pruning specific for smart questions: remove phones/emails and overly-unique text fields
        additional_excluded = set()
        phone_email_keywords = [
            'phone', 'mobile', 'telephone', 'tel', 'whatsapp', 'fax',
            'contact', 'contact_no', 'contact_number', 'email', 'e-mail'
        ]
        for col in meaningful_cols:
            lower = col.lower().strip()
            if any(k in lower for k in phone_email_keywords):
                additional_excluded.add(col)
                continue
            # Exclude highly-unique text columns (likely identifiers or free text not good for suggestions)
            try:
                if df[col].dtype in ['object', 'string']:
                    total = len(df)
                    if total > 0:
                        uniq_ratio = df[col].nunique(dropna=True) / total
                        if uniq_ratio > 0.95:
                            additional_excluded.add(col)
                            continue
            except Exception:
                pass

        pruned_cols = [c for c in meaningful_cols if c not in additional_excluded]
        # Ensure at least one column remains
        df_meaningful = df[pruned_cols] if pruned_cols else df[meaningful_cols]

        schema_analyzer = DataSchemaAnalyzer(df_meaningful)
        schema_analysis = schema_analyzer.analyze_schema()

        smart_questions = generate_smart_questions(df_meaningful, schema_analysis)

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

        # Get algorithm parameters from form data
        min_support = float(request.form.get('min_support', 0.05))
        min_confidence = float(request.form.get('min_confidence', 0.5))
        max_itemset_size = int(request.form.get('max_itemset_size', 3))

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

        # Analyze patterns with improved algorithm
        frequent_itemsets, association_rules = pattern_analyzer.analyze_patterns(
            min_support=min_support, 
            min_confidence=min_confidence, 
            max_itemset_size=max_itemset_size
        )
        
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
