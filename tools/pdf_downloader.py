import requests
import os
from pathlib import Path
from urllib.parse import urlparse

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class MyPDFDownloaderTool(BaseModel):
    """Input schema for MyCustomTool."""
    url: str = Field(str,description= "The URL pointing directly to the PDF file to be downloaded.")


class PDFDownloaderTool(BaseTool):
    name: str = "PDF downloader tool"
    description: str = "Use this tool to download a PDF file from given URL in a specified folder."
    args_schema: Type[BaseModel] = MyPDFDownloaderTool

    def _run(self, url: str, folder_path: str) -> str:
        """
        Downloads a PDF from a given URL and saves it to a specified filename.
        """

        parsed_url = urlparse(url)
        path = parsed_url.path
        filename = os.path.basename(path)

        target_path = Path(folder_path)
        full_path = target_path / filename
        
        try:
            # 1. Send a GET request with streaming enabled
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # 2. Open the local file in binary write mode ('wb')
            with open(full_path, 'wb') as pdf_file:
                # Iterate over the response content in chunks
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        pdf_file.write(chunk)

            print(f"✅ Successfully downloaded '{filename}'")

        except requests.exceptions.RequestException as e:
            print(f"❌ An error occurred during the download: {e}")

# def download_pdf(url, folder_path):
#     """
#     Downloads a PDF from a given URL and saves it to a specified filename.
#     """

#     parsed_url = urlparse(url)
#     path = parsed_url.path
#     filename = os.path.basename(path)

#     target_path = Path(folder_path)
#     full_path = target_path / filename
    
#     try:
#         # 1. Send a GET request with streaming enabled
#         response = requests.get(url, stream=True)
#         response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

#         # 2. Open the local file in binary write mode ('wb')
#         with open(full_path, 'wb') as pdf_file:
#             # Iterate over the response content in chunks
#             for chunk in response.iter_content(chunk_size=8192):
#                 if chunk:  # Filter out keep-alive chunks
#                     pdf_file.write(chunk)

#         print(f"✅ Successfully downloaded '{filename}'")

#     except requests.exceptions.RequestException as e:
#         print(f"❌ An error occurred during the download: {e}")