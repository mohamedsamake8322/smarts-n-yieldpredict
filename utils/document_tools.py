import os
import json
import logging
import pandas as pd
import PyPDF2
import csv
from io import StringIO

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handle processing of various document formats"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.json': self._process_json,
            '.xlsx': self._process_excel,
            '.xls': self._process_excel,
            '.csv': self._process_csv,
            '.txt': self._process_text
        }
    
    def process_file(self, file_path):
        """Process a file and extract text content"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            processor = self.supported_formats[file_extension]
            text_content = processor(file_path)
            
            # Clean and validate content
            if not isinstance(text_content, str):
                text_content = str(text_content)
            
            text_content = self._clean_text(text_content)
            
            if not text_content.strip():
                raise ValueError("No text content could be extracted from the file")
            
            logger.info(f"Successfully processed {file_path}, extracted {len(text_content)} characters")
            return text_content
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def _process_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            text_content = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        text_content.append(f"Page {page_num + 1}:\n{text}")
            
            return '\n\n'.join(text_content)
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def _process_json(self, file_path):
        """Extract text from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Convert JSON to readable text
            text_content = self._json_to_text(data)
            return text_content
            
        except Exception as e:
            raise Exception(f"Error processing JSON: {str(e)}")
    
    def _process_excel(self, file_path):
        """Extract text from Excel file"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            text_content = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Convert DataFrame to text
                sheet_text = f"Sheet: {sheet_name}\n"
                sheet_text += "Columns: " + ", ".join(df.columns.tolist()) + "\n\n"
                
                # Add data rows
                for index, row in df.iterrows():
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    if row_text.strip():
                        sheet_text += row_text + "\n"
                
                text_content.append(sheet_text)
            
            return '\n\n'.join(text_content)
            
        except Exception as e:
            raise Exception(f"Error processing Excel: {str(e)}")
    
    def _process_csv(self, file_path):
        """Extract text from CSV file"""
        try:
            text_content = []
            
            with open(file_path, 'r', encoding='utf-8') as file:
                # Try to detect delimiter
                sample = file.read(1024)
                file.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter
                
                reader = csv.DictReader(file, delimiter=delimiter)
                
                # Add column headers
                text_content.append("Columns: " + ", ".join(reader.fieldnames))
                text_content.append("")
                
                # Add data rows
                for row_num, row in enumerate(reader, 1):
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if val])
                    if row_text.strip():
                        text_content.append(f"Row {row_num}: {row_text}")
            
            return '\n'.join(text_content)
            
        except Exception as e:
            raise Exception(f"Error processing CSV: {str(e)}")
    
    def _process_text(self, file_path):
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
                
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                raise Exception(f"Error reading text file with encoding: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing text file: {str(e)}")
    
    def _json_to_text(self, data, level=0):
        """Convert JSON data to readable text"""
        text_content = []
        indent = "  " * level
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    text_content.append(f"{indent}{key}:")
                    text_content.append(self._json_to_text(value, level + 1))
                else:
                    text_content.append(f"{indent}{key}: {value}")
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                text_content.append(f"{indent}[{i}]:")
                text_content.append(self._json_to_text(item, level + 1))
        
        else:
            text_content.append(f"{indent}{data}")
        
        return '\n'.join(text_content)
    
    def _clean_text(self, text):
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        # Join lines and limit excessive newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive consecutive newlines
        while '\n\n\n' in cleaned_text:
            cleaned_text = cleaned_text.replace('\n\n\n', '\n\n')
        
        return cleaned_text
