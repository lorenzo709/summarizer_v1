from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
from src.MyTypes import Score

load_dotenv()

# llm = LLM(model="ollama/deepseek-r1:8b", base_url="http://localhost:11434")
llm = LLM(model="ollama/gpt-oss:120b", base_url="http://localhost:11434", temperature=0.1)

@CrewBase
class JudgeCrew:
    """Crew responsible for judging summaries of scientific papers"""

    # agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def judge(self) -> Agent:
        return Agent(
            llm=llm,
            role="Expert Scientific Summarization Evaluator",
            goal=(
                "To meticulously assess the quality of a scientific summary against its source"
                "material, scoring it from 0 to 5 based on Alignment (Precision) and Coverage"
                "(Recall), and providing actionable feedback for improvement."
            ),
            backstory=(
                "You are a seasoned editor and reviewer in a high-stakes scientific journal. Your"
                "primary function is to serve as the final quality gatekeeper. You are"
                "hyper-focused on factual accuracy (Alignment) to prevent the spread of"
                "misinformation (hallucination), and ensuring completeness (Coverage) so no key"
                "findings are lost. Your expertise lies in providing concise, constructive"
                "feedback that guarantees the final summary is both faithful and comprehensive."
            ),
            verbose=True,
        )


    @task
    def judge_task(self) -> Task:
        return Task(
            config=self.tasks_config["judge_task"],  # type: ignore[index]
            agent=self.judge(),
            output_pydantic= Score 
        )


    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[ self.judge() ],
            tasks=[ self.judge_task() ],
            process=Process.sequential,
        )