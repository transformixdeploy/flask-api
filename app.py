from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
from collections import defaultdict

app = Flask(__name__)

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
        
        return {
            "business_domain": business_domain,
            "primary_entity": primary_entity
        }

def calculate_data_metrics(df):
    try:
        # Basic metrics
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Missing data calculation
        total_cells = total_rows * total_columns
        missing_cells = df.isnull().sum().sum()
        missing_ratio = (missing_cells / total_cells) * 100 if total_cells > 0 else 0
        
        # Numeric columns count
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        num_numeric_columns = len(numeric_columns)
        
        # Additional useful metrics
        text_columns = df.select_dtypes(include=['object', 'string']).columns
        num_text_columns = len(text_columns)
        
        # Data completeness by column
        completeness_by_column = {}
        for col in df.columns:
            non_null_count = df[col].count()
            completeness_pct = (non_null_count / total_rows) * 100 if total_rows > 0 else 0
            completeness_by_column[col] = {
                'non_null_count': int(non_null_count),
                'completeness_percentage': round(completeness_pct, 1)
            }
        
        # Unique values analysis
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
                    print(f"✅ Successfully read CSV with encoding={encoding}, delimiter='{delimiter}'")
                    break 
                except Exception as e:
                    print(f"⚠️ Failed with encoding={encoding}, delimiter='{delimiter}':", e)
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
        print("❌ Unexpected error:", e)
        return jsonify({
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5001,debug=True)