from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    ArxivPaperTool,
    DirectoryReadTool,
    FileReadTool,
    ScrapeWebsiteTool,
    SerperDevTool,
)
from src.MyTypes import ParsedText, ParsedPapers, PDFPapers,PaperFound
from dotenv import load_dotenv

from typing import List
load_dotenv()

from tools.pdf_parser import PDFParserTool
# llm = LLM(model="ollama/deepseek-r1:8b", base_url="http://localhost:11434")
llm = LLM(model="ollama/gpt-oss:120b", base_url="http://localhost:11434")

@CrewBase 
class ResearcherCrew:
    # agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            tools=[
                SerperDevTool(),
                ArxivPaperTool(
                    download_pdfs=True,
                    save_dir="./knowledge",
                    use_title_as_filename=True,
                ),
                # DirectoryReadTool()
            ],
            # llm="gemini/gemini-2.0-flash",
            llm=llm,
            role="Scientific Papers Researcher Specialist",
            goal="Accurately search the internet to find scientific papers on a"
            "{topic} and use tools such as ArxivPaperTool to download the PDF in"
            "a specific folder",
            backstory=(
                "You are a meticulous researcher, and your primary function is "
                "to use your search tool to gather data, and your "
                "pdf download tool to save it."
            ),
            verbose=True,
        )

    # @agent
    # def parser(self) -> Agent:
    #     return Agent(
    #         # llm="gemini/gemini-2.0-flash",
    #         llm = llm,
    #         tools=[
    #             # PDFParserTool(),
    #             DirectoryReadTool()
    #         ],
    #         role="Scientific Papers Parser Specialist",
    #         # goal="For each path given in the output of the previous agent,Use"
    #         # "the PDFParserTool to extract text the PDF file in the knowledge"
    #         # "folder.",
    #         goal= "for each paper found in folder 'knowledge', return both the name of that paper and the path where its saved",
    #         backstory=(
    #             """ 
    #             you are a specialist in parsing text from a PDF file into a precise, easily digestible text format
    #             for other experts to use.
    #             """
    #         ),
    #         verbose=True
    #     )   

    @agent
    def organizer(self) -> Agent:
        return Agent(
            # llm="gemini/gemini-2.0-flash",
            llm = llm,
                tools=[
                DirectoryReadTool()
            ],
                role="The Data Structure Validator",
            goal = """
                to read the file inside the folder 'knowledge' with the tool DirectoryReadTool() and return the EXACT name of the pdf file
                and the EXACT path relative to the project and produce a final, validated PDFPapers
                 Pydantic object containing the pdf_name and pdf_path for every paper found. The
                 primary objective is structured output fidelity."
                   """,
            # goal= """
            #     To receive the output from the discovery agent (a list of downloaded paper
            #     paths), extract the paper names, and produce a final, validated PDFPapers
            #     Pydantic object containing the pdf_name and pdf_path for every paper found. The
            #     primary objective is structured output fidelity."
            #     """,
            backstory=(
                """ 
                I am an expert in data engineering and file system management. My training
                focuses on Pydantic schema validation, ensuring data integrity, and confirming
                the existence of files based on provided metadata. I prioritize precision and
                strict adherence to output schemas over creative interpretation.
                """
            ),
            verbose=True
        )   

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
            agent=self.researcher(),
        )
    
    @task
    def organize_task(self) -> Task:
        return Task(
            config=self.tasks_config["organize_task"],  # type: ignore[index]
            agent=self.organizer(),
            output_pydantic= PDFPapers
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.researcher(),self.organizer()],
            tasks=[self.research_task(),self.organize_task()],
            process=Process.sequential,
        )