from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
    ArxivPaperTool,
    DirectoryReadTool,
    FileReadTool,
    ScrapeWebsiteTool,
    SerperDevTool,
)
from dotenv import load_dotenv
from MyTypes import Summary

load_dotenv()

llm = LLM(model="ollama/deepseek-r1:8b", base_url="http://localhost:11434")

@CrewBase
class SummarizationCrew:
    """research/writing crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def writer(self) -> Agent:
        return Agent(
            llm=llm,
            # llm="gemini/gemini-2.0-flash",
            tools=[DirectoryReadTool()],
            role="Expert Summarization Writer",
            # goal="Write a short summarization of the scientific papers found from the previous task,"
            # "make sure that all the paper is taken in consideration by using the tools (if given)",
            # goal="Write a short summarization for each scientific papers found inside the folder 'knowledge'",
            goal="Write a short summarization on the scientific paper found inside the folder 'knowledge'",
            backstory="You are an experience writer in summarizing a text, "
            "speficially scientific papers, always capable to select the most relevant informations",
            verbose=True,
        )


    @task
    def summarize_task(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_task"],  # type: ignore[index]
            agent=self.writer(),
            output_pydantic= Summary
        )


    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.writer(),
                # self.reviewer(),
            ],
            tasks=[
                # self.scrape_task(),
                self.summarize_task(),
                # self.reviewer_task(),
            ],
            process=Process.sequential,
        )