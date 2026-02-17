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
# llm = LLM(model="ollama/gpt-oss:120b", base_url="http://localhost:11434")
llm = LLM(model="ollama/qwen2.5:72b", base_url="http://localhost:11434")
# llm = LLM(model="ollama/llama3.1:70b", base_url="http://localhost:11434")
@CrewBase
class SummarizationCrew:
    """research/writing crew"""

    # agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def writer(self) -> Agent:
        return Agent(
            llm=llm,
            # llm="gemini/gemini-2.0-flash",
            # tools=[DirectoryReadTool()],
            # role="Expert Summarization Writer",
            # goal="Write a short summarization of the scientific papers found from the previous task,"
            # "make sure that all the paper is taken in consideration by using the tools (if given)",
            # goal="Write a short summarization for each scientific papers found inside the folder 'knowledge'",
            # goal="Write a short summarization on this paper text given ",
            # backstory="You are an experience writer in summarizing a text, "
            # "speficially scientific papers, always capable to select the most relevant informations",
            role="Expert Scientific Summary Writer and Critic",
            goal=(
                "Produce a **400-500 word summary** of the given scientific paper. "
                "The summary must be written in the **formal, academic style** of a 'Related Work' section "
                "and **must include a dedicated section on the paper's limitations, shortcomings, and areas for future work.**"
            ),
            backstory=(
                "You are an experienced academic writer and critic, specializing in dissecting scientific papers. "
                "Your expertise lies in creating highly **detailed, structured, and critical** summaries that are "
                "suitable for inclusion in the 'Related Work' section of a new publication. "
                "You are exceptional at not only extracting the core contributions, methodology, and results, "
                "but also identifying and articulating the **novelty, assumptions, constraints, and open problems** (limitations) "
                "of the work being summarized. Your output is always between 400 and 500 words."
            ),
            verbose=True,
            max_rpm=10,
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
            agents=[ self.writer() ],
            tasks=[ self.summarize_task() ],
            process=Process.sequential,
        )