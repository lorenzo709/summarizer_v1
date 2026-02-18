from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from src.MyTypes import Summary

load_dotenv()

# llm = LLM(model="ollama/deepseek-r1:8b", base_url="http://localhost:11434")
# llm = LLM(model="ollama/gpt-oss:120b", base_url="http://localhost:11434")
# llm = LLM(model="ollama/qwen2.5:72b", base_url="http://localhost:11434")
llm = LLM(model="ollama/llama3.1:70b", base_url="http://localhost:11434")

@CrewBase
class CorrectionCrew:
    """Crew responsible for correcting summaries"""

    # agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def corrector(self) -> Agent:
        return Agent(
            llm=llm,
            role="Precision Editor & Content Reconciler",
            goal=(
                "Refine summaries by integrating specific hints and correcting errors identified"
                "by the judge, ensuring 100% alignment with the source text."
            ),
            backstory=(
                "You are a meticulous senior editor known for your ability to take critical"
                "feedback and transform it into polished, accurate content. You don't just fix"
                "grammar; you analyze why a judge flagged a mistake and cross-reference it with"
                "the original source text to ensure the final summary is flawless. You pride"
                "yourself on maintaining the author's tone while achieving technical perfection."
            ),
            verbose=True,
        )

    @task
    def correction_task(self) -> Task:
        return Task(
            config=self.tasks_config["correction_task"],  # type: ignore[index]
            agent=self.corrector(),
            output_pydantic= Summary
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[ self.corrector() ],
            tasks=[ self.correction_task() ],
            process=Process.sequential,
        )
