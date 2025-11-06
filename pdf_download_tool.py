import os
import uuid
from typing import ClassVar, Type

import requests

# Using BaseTool from langchain_core.tools is a common and robust pattern
# for creating custom tools in many agent frameworks.
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# --- Pydantic Schema for Tool Input ---


class PDFDownloaderInput(BaseModel):
    """Input for the PDFDownloaderTool."""

    url: str = Field(
        description="The URL pointing directly to the PDF file to be downloaded."
    )
    title: str = Field(
        description="A short, descriptive title for the paper to be used as part of the filename."
    )


# --- Custom Tool Implementation ---


class PDFDownloaderTool(BaseTool):
    """
    A tool to download a PDF file from a given URL and save it locally.
    It checks the Content-Type header to ensure the file is indeed a PDF.
    """

    name: str = "PDF_Downloader"
    description: str = (
        "Use this tool to download a scientific paper (PDF file) from a direct URL. "
        "The input MUST be the full URL and a descriptive title."
    )
    args_schema: ClassVar[Type[BaseModel]] = Field(default=PDFDownloaderInput)

    def _run(self, url: str, title: str) -> str:
        """
        Downloads the PDF file and returns the path to the saved file or an error message.
        """
        # Define the directory to save the PDFs
        save_dir = os.path.join(os.getcwd(), "downloads", "pdfs")

        try:
            # 1. Ensure the download directory exists
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 2. Make the request to download the file
            response = requests.get(url, stream=True, timeout=15)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # 3. Check if the content is actually a PDF
            content_type = response.headers.get("Content-Type", "")
            if "application/pdf" not in content_type:
                return (
                    f"Download failed for {url}. URL did not point to a PDF. "
                    f"Content-Type found: {content_type}"
                )

            # 4. Sanitize the title and create a unique filename
            sanitized_title = "".join(
                c if c.isalnum() or c in (" ", "_", "-") else "" for c in title
            ).strip()
            # Replace spaces with underscores and limit title length
            base_filename = sanitized_title.replace(" ", "_")[:50]
            unique_id = uuid.uuid4().hex[:6]
            filename = f"{base_filename}_{unique_id}.pdf"
            file_path = os.path.join(save_dir, filename)

            # 5. Save the content to the file
            with open(file_path, "wb") as pdf_file:
                for chunk in response.iter_content(chunk_size=8192):
                    pdf_file.write(chunk)

            return f"Successfully downloaded paper to: {file_path}"

        except requests.exceptions.RequestException as e:
            return f"Network or request error while attempting to download {url}: {e}"
        except Exception as e:
            return f"An unexpected error occurred during file saving: {e}"

    # Note: The _arun method for async calls is left unimplemented as the base class handles this.
    # If your agent framework uses async tools, you would implement _arun.
