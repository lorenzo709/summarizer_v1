import pymupdf # imports the pymupdf library
import re

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class MyParserTool(BaseModel):
    """Input schema for MyCustomTool."""
    pdf_path: str = Field(str,description= "path to the pdf we need to extract text from")

class PDFParserTool(BaseTool):
    name: str = "PDF parser tool"
    description: str = "Use this tool to extract text from a PDF file."
    args_schema: Type[BaseModel] = MyParserTool

    def _run(self, pdf_path: str) -> str:
        # 1. Initialize an empty string to store all the text
        full_text = ""
        
        try:
            # 2. Open the PDF document
            pdf = pymupdf.open(pdf_path)
            
            # 3. Iterate through every page in the document
            for page in pdf:
                # 4. Extract the text from the current page
                text = page.get_text(sort=True)
                
                # 5. Append the page text to the full_text string
                #    Adding a newline character (\n) is good practice to separate pages.
                full_text += text + "\n"
                
            # 6. Close the document (good practice)
            pdf.close()
            
        except pymupdf.FileNotFoundError:
            return f"Error: PDF file '{pdf_path}' not found."
        except Exception as e:
            return f"An error occurred: {e}"

    # --- Truncation Logic ---
        
        # Define common headers, separated by the pipe (|) for OR matching
        # We include some common variations that might appear (e.g., singular, hyphenated)
        headers = r"REFERENCES|BIBLIOGRAPHY|LITERATURE CITED|APPENDIX|ACKNOWLEDGEMENTS"
        
        # Create the full search pattern
        # 1. We use re.IGNORECASE to handle "References", "REFERENCES", etc.
        # 2. \b: Ensures a word boundary (prevents matching "PREFERENCES" for "REFERENCES").
        # 3. \s*: Accounts for zero or more spaces/newlines between words and punctuation.
        # 4. [.:]*: Accounts for optional trailing punctuation like a colon or period.
        pattern = r"\b(" + headers + r")\b[.:]*\s*"

        # Find the earliest match
        match = re.search(pattern, full_text, re.IGNORECASE)

        if match:
            # The .start() method gives the exact index where the match begins.
            return full_text[:match.start()]
        else:
            # If no header is found, return the full text
            return full_text
    

# def pdf_parser(pdf_name):
#     # 1. Initialize an empty string to store all the text
#     full_text = ""
    
#     try:
#         # 2. Open the PDF document
#         pdf = pymupdf.open(pdf_name)
        
#         # 3. Iterate through every page in the document
#         for page in pdf:
#             # 4. Extract the text from the current page
#             text = page.get_text(sort=True)
            
#             # 5. Append the page text to the full_text string
#             #    Adding a newline character (\n) is good practice to separate pages.
#             full_text += text + "\n"
            
#         # 6. Close the document (good practice)
#         pdf.close()
        
#     except pymupdf.FileNotFoundError:
#         return f"Error: PDF file '{pdf_name}' not found."
#     except Exception as e:
#         return f"An error occurred: {e}"

# # --- Truncation Logic ---
    
#     # Define common headers, separated by the pipe (|) for OR matching
#     # We include some common variations that might appear (e.g., singular, hyphenated)
#     headers = r"REFERENCES|BIBLIOGRAPHY|LITERATURE CITED|APPENDIX|ACKNOWLEDGEMENTS"
    
#     # Create the full search pattern
#     # 1. We use re.IGNORECASE to handle "References", "REFERENCES", etc.
#     # 2. \b: Ensures a word boundary (prevents matching "PREFERENCES" for "REFERENCES").
#     # 3. \s*: Accounts for zero or more spaces/newlines between words and punctuation.
#     # 4. [.:]*: Accounts for optional trailing punctuation like a colon or period.
#     pattern = r"\b(" + headers + r")\b[.:]*\s*"

#     # Find the earliest match
#     match = re.search(pattern, full_text, re.IGNORECASE)

#     if match:
#         # The .start() method gives the exact index where the match begins.
#         return full_text[:match.start()]
#     else:
#         # If no header is found, return the full text
#         return full_text