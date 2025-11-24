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

from src.MyTypes import Summary
load_dotenv()

# llm = LLM(model="ollama/deepseek-r1:8b", base_url="http://localhost:11434")
llm = LLM(model="ollama/gpt-oss:120b", base_url="http://localhost:11434")
@CrewBase
class ReviewerCrew:
    @agent
    def reviewer(self) -> Agent:
        return Agent(
            llm=llm,
            # llm="gemini/gemini-2.0-flash",
            role="Academic Reviewer Specialist",
            goal="""To meticulously analyze a given academic paper, identifying
            its core contributions, strengths (pros), and areas for improvement
            or shortcomings (limitations), and structure these findings into two
            distinct, well-reasoned lists""",
            backstory="""You spent a decade as a peer reviewer for top-tier
            academic journals in your field. You are known for your objective,
            balanced, and sharp analytical mind. You believe that all research,
            no matter how groundbreaking, has inherent boundaries, and your sole
            purpose is to provide an honest, unbiased, and constructive
            critique. Your reviews are concise, insightful, and strictly focus
            on the scientific merit and applicability of the work""",
            verbose=True,
        )
    
    @task
    def reviewer_task(self) -> Task:
        return Task(
            config=self.tasks_config["reviewer_task"],  # type: ignore[index]
            agent=self.reviewer(),
            output_pydantic= Summary
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.reviewer(),
            ],
            tasks=[
                self.reviewer_task(),
            ],
            process=Process.sequential,
        )