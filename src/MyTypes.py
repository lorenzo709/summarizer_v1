from typing import List

from pydantic import BaseModel

class PaperFound(BaseModel):
    pdf_name: str 
    pdf_path: str

class PDFPapers(BaseModel):
    papers: List[PaperFound]

class ParsedText(BaseModel):
    pdf_name: str 
    parsed_text: str 

class ParsedPapers(BaseModel):
    parsed_papers: List[ParsedText]

class Summary(BaseModel):
    summary: str

class ProsCons(BaseModel):
    paper_name: str
    pros_and_cons: str