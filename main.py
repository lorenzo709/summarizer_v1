from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    # DirectoryReadTool,
    # FileWriterTool,
    # FileReadTool,
)

from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="ollama/deepseek-r1:8b", base_url="http://localhost:11434")


@CrewBase
class BlogCrew:
    """ "Blog writing crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            # tools=[SerperDevTool(),ScrapeWebsiteTool()],
            tools=[SerperDevTool()],
            llm="gemini/gemini-2.0-flash",
            role="Scientific Papers Researcher Specialist",
            goal="Accurately search the internet to find pertinent scientific paper",
            backstory=(
                "You are a meticulous researcher, and your primary function is "
                "to use your search tool to gather data."
            ),
            verbose=True,
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            llm=llm,
            # llm="gemini/gemini-2.0-flash",
            tool = [ScrapeWebsiteTool()],
            role="Expert Summarization Writer",
            goal="Write a short summarization of the scientific papers from the previous task",
            backstory="You are an experience writer in summarizing a text",
            verbose=True,
        )

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
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
            agent=self.researcher(),
        )

    @task
    def summarize_task(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_task"],  # type: ignore[index]
            agent=self.writer(),
        )

    @task
    def reviewer_task(self) -> Task:
        return Task(
            config=self.tasks_config["reviewer_task"],  # type: ignore[index]
            agent=self.reviewer(),
        )
    @task
    def aggregate_task(self) -> Task:
        return Task(
            config=self.tasks_config["aggregate_task"],  # type: ignore[index]
            agent=self.aggregator(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.researcher(), self.writer(), self.aggregator(),self.reviewer()],
            tasks=[self.research_task(), self.summarize_task(), self.aggregate_task(),self.reviewer_task()],
        )


if __name__ == "__main__":
    blog_crew = BlogCrew()
    blog_crew.crew().kickoff(inputs={"topic": "Summarization with llms"})
