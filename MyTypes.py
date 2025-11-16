from typing import List

from pydantic import BaseModel

class Paths_to_Papers(BaseModel):
    paths: List[str]

class ParsedText(BaseModel):
    pdf_name: str
    parsed_text: str

class Summary(BaseModel):
    summary: str

class ProsCons(BaseModel):
    paper_name: str
    pros_and_cons: str