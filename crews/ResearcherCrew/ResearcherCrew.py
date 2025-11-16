from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    ArxivPaperTool,
    DirectoryReadTool,
    FileReadTool,
    ScrapeWebsiteTool,
    SerperDevTool,
)
from MyTypes import Paths_to_Papers, ParsedText
from dotenv import load_dotenv

from typing import List
load_dotenv()

from tools.pdf_parser import PDFParserTool
llm = LLM(model="ollama/deepseek-r1:8b", base_url="http://localhost:11434")

@CrewBase 
class ResearcherCrew:
    agents_config = "config/agents.yaml"
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
                PDFParserTool()
            ],
            # llm="gemini/gemini-2.0-flash",
            llm=llm,
            role="Scientific Papers Researcher Specialist",
            goal="Accurately search the internet to find pertinent scientific"
            "papers on a given topic and use tools such as ArxivPaperTool to download the PDF in a specific folder",
            backstory=(
                "You are a meticulous researcher, and your primary function is "
                "to use your search tool to gather data, and your "
                "pdf download tool to save it."
            ),
            verbose=True,
        )
    

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
            agent=self.researcher(),
            # output_pydantic=Paths_to_Papers
        )
    
    @task
    def scrape_task(self) -> Task:
        return Task(
            config=self.tasks_config["scraper_task"],  # type: ignore[index]
            agent=self.writer(),
            output_pydantic= List[ParsedText]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.researcher()],
            tasks=[self.research_task(),self.scrape_task()],
            process=Process.sequential,
        )