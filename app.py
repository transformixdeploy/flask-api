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
class SmartRAGAssistant:
    def __init__(self, df, schema_analysis):
        self.df = df
        self.schema_analysis = schema_analysis
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
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
            "business_insights": self.schema_analysis.get('insights', [])
        }
        return context
    
    def get_statistical_summary(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        summary = {}
        
        for col in numeric_cols:
            if not self.df[col].isna().all():
                summary[col] = {
                    "count": int(self.df[col].count()),
                    "mean": float(self.df[col].mean()),
                    "median": float(self.df[col].median()),
                    "std": float(self.df[col].std()),
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                    "q25": float(self.df[col].quantile(0.25)),
                    "q75": float(self.df[col].quantile(0.75)),
                    "sum": float(self.df[col].sum()),
                    "unique_count": int(self.df[col].nunique())
                }
        
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
    
    def answer_question(self, question):
        try:
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

DATA QUALITY METRICS:
{json.dumps(data_context['data_quality'], indent=2)}

SAMPLE DATA (First {len(data_context['sample_data'])} records):
{json.dumps(data_context['sample_data'], indent=2)}

USER QUESTION: "{question}"

Instructions:
1. Analyze the question in the context of the available data
2. Provide specific, data-driven answers based on the actual dataset
3. Include relevant statistics, trends, and patterns
4. If the question requires filtering or specific calculations, explain what you found
5. Suggest actionable insights where appropriate
6. If you need to make assumptions, state them clearly

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
            if isinstance(value, (int, float)):
                relevant_statistics.append({
                    "key": key.replace('_', ' ').title(),
                    "value": value
                })
            else:
                # Handle non-numeric statistics
                relevant_statistics.append({
                    "key": key.replace('_', ' ').title(),
                    "value": str(value)
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

@app.route('/ai/smart-question-example', methods=['POST'])
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
