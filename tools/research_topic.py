from pathlib import Path
import re
import requests

from src.MyTypes import PaperInfos

def research_topic(topic):
    url = "https://google.serper.dev/scholar?q=apple+inc&apiKey=38b17efee83186f292b33341b7fe5528bc73005f"
    formatted_topic = topic.replace(" ", "+")

    # Regex breakdown:
    # q=      : matches the literal characters "q="
    # [^&]* : matches any character except "&" (zero or more times)
    query = re.sub(r'q=[^&]*', f'q={formatted_topic}', url)

    payload = {}
    headers = {}

    results = requests.request("GET", query, headers=headers, data=payload)
    dictionary = results.json()
    return dictionary

def download_pdf(url, filename, folder_path):
    """
    Downloads a PDF from a given URL and saves it to a specified filename.
    """

    # parsed_url = urlparse(url)
    # path = parsed_url.path
    # filename = os.path.basename(path)

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
        return None

def search_and_save_pdf(topic, folder):
    biblex_papers = []
    results = research_topic(topic)
    for result in results["organic"]:
            title = result["title"]
            publicationInfo = result["publicationInfo"]
            year = result["year"]
            pdfUrl = result.get("pdfUrl")
            if pdfUrl is None:
                continue
            filename = f"{title}.pdf"
            if download_pdf(pdfUrl,filename, folder) is None:
                continue
            info_paper = PaperInfos(title=title, publicationInfo=publicationInfo, year=year, pdfUrl=pdfUrl, filename=filename)
            biblex_papers.append(info_paper)
    return biblex_papers
