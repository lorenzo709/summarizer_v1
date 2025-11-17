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

from tools.pdf_parser import PDFParserTool
from MyTypes import Summary

load_dotenv()

# llm = LLM(model="ollama/deepseek-r1:8b", base_url="http://localhost:11434")
llm = LLM(model="ollama/gpt-oss:120b", base_url="http://localhost:11434")

@CrewBase
class AggregateCrew:

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    @agent
    def aggregator(self) -> Agent:
        return Agent(
            # config=self.agents_config["paper_writer_agent"],  # type: ignore[index]
            llm=llm,
            # llm="gemini/gemini-2.0-flash",
            role="Expert Summarization WriterAcademic Synthesizer and Narrative Editor",
            goal="""To generate a single, continuous, and easy-to-read document
            (400–600 words) that seamlessly integrates the content of 5–6
            individual paper summaries, ensuring a logical progression of ideas
            from start to finish.
            """,
            backstory="""you are the Lead Synthesis Editor for a top-tier
            science communication firm. Your sole job is to take fragmented,
            specialized research summaries and fuse them into a single,
            cohesive, and compelling narrative for executive and public
            audiences""",
            verbose=True,
        )

    @task
    def aggregate_task(self) -> Task:
        return Task(
            config=self.tasks_config["aggregate_task"],  # type: ignore[index]
            agent=self.aggregator(),
            output_pydantic=Summary
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.aggregator(),
            ],
            tasks=[
                self.aggregate_task(),
            ],
            process=Process.sequential,
        )