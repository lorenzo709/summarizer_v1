from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from src.MyTypes import Summary

load_dotenv()

# llm = LLM(model="ollama/deepseek-r1:8b", base_url="http://localhost:11434")
# llm = LLM(model="ollama/gpt-oss:120b", base_url="http://localhost:11434")
llm = LLM(model="ollama/qwen2.5:72b", base_url="http://localhost:11434")
# llm = LLM(model="ollama/llama3.1:70b", base_url="http://localhost:11434")

@CrewBase
class GapResearcherCrew:
    """Crew responsible for findings gaps in the current state of the art"""

    # agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def gap_researcher(self) -> Agent:
        return Agent(
            llm=llm,
            role="Chief Research Gap Analyst",
            goal=(
                "To **synthesize and critically analyze** the collective list of limitations and"
                "shortcomings from multiple scientific papers, and then **translate** them into a"
                "prioritized list of **3-5 distinct and high-impact research opportunities (or"
                "'holes')** that represent the current State-of-the-Art gaps in the specified"
                "field."
            ),
            backstory=(
                "You are a **seasoned academic and research strategist** with a profound"
                "understanding of the scientific method and literature review. Your expertise"
                "lies in looking beyond individual paper failures to **identify systemic gaps and"
                "unmet needs** within a scientific domain. You do not just list limitations; you"
                "treat them as **clues** that, when aggregated, point to the most critical,"
                "unexplored areas for future work. You are known for providing **clear,"
                "well-justified, and innovative** directions for research."
            ),
            verbose=True,
        )


    @task
    def research_gap_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_gap_task"],  # type: ignore[index]
            agent=self.gap_researcher(),
            output_pydantic= Summary
        )


    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[ self.gap_researcher() ],
            tasks=[ self.research_gap_task() ],
            process=Process.sequential,
        )