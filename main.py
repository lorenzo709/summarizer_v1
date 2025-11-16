from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.flow.flow import Flow, listen, start
from crewai_tools import (
    ArxivPaperTool,
    DirectoryReadTool,
    FileReadTool,
    ScrapeWebsiteTool,
    SerperDevTool,
)
from dotenv import load_dotenv

import asyncio
from pydantic import BaseModel

from crews.AggregatorCrew.AggregatorCrew import AggregateCrew
from crews.ResearcherCrew.ResearcherCrew import ResearcherCrew
from crews.SummarizationCrew.SummarizationCrew import SummarizationCrew
from crews.ReviewerCrew import ReviewerCrew

from MyTypes import Paths_to_Papers, ParsedText, Summary, ProsCons
from typing import List

load_dotenv()

llm = LLM(model="ollama/deepseek-r1:8b", base_url="http://localhost:11434")

# @CrewBase 
# class ResearchCrew:
#     agents_config = "config/agents.yaml"
#     tasks_config = "config/tasks.yaml"

#     @agent
#     def researcher(self) -> Agent:
#         return Agent(
#             tools=[
#                 SerperDevTool(),
#                 ArxivPaperTool(
#                     download_pdfs=True,
#                     save_dir="./knowledge",
#                     use_title_as_filename=True,
#                 ),
#             ],
#             # llm="gemini/gemini-2.0-flash",
#             llm=llm,
#             role="Scientific Papers Researcher Specialist",
#             goal="Accurately search the internet to find pertinent scientific"
#             "papers on a given topic and use tools such as ArxivPaperTool to download the PDF in a specific folder",
#             backstory=(
#                 "You are a meticulous researcher, and your primary function is "
#                 "to use your search tool to gather data, and your "
#                 "pdf download tool to save it."
#             ),
#             verbose=True,
#         )
    

#     @task
#     def research_task(self) -> Task:
#         return Task(
#             config=self.tasks_config["research_task"],  # type: ignore[index]
#             agent=self.researcher(),
#         )
    
#     @crew
#     def crew(self) -> Crew:
#         return Crew(
#             agents=[self.researcher()],
#             tasks=[self.research_task()],
#             process=Process.sequential,
#         )
    


# @CrewBase
# class ResearcherCrew:
#     """research/writing crew"""

#     agents_config = "config/agents.yaml"
#     tasks_config = "config/tasks.yaml"

#     # @agent
#     # def researcher(self) -> Agent:
#     #     return Agent(
#     #         tools=[
#     #             SerperDevTool(),
#     #             ArxivPaperTool(
#     #                 download_pdfs=True,
#     #                 save_dir="./knowledge",
#     #                 use_title_as_filename=True,
#     #             ),
#     #         ],
#     #         # llm="gemini/gemini-2.0-flash",
#     #         llm=llm,
#     #         role="Scientific Papers Researcher Specialist",
#     #         goal="Accurately search the internet to find pertinent scientific"
#     #         "papers on a given topic and use tools such as ArxivPaperTool to download the PDF in a specific folder",
#     #         backstory=(
#     #             "You are a meticulous researcher, and your primary function is "
#     #             "to use your search tool to gather data, and your "
#     #             "pdf download tool to save it."
#     #         ),
#     #         verbose=True,
#     #     )

#     @agent
#     def writer(self) -> Agent:
#         return Agent(
#             llm=llm,
#             # llm="gemini/gemini-2.0-flash",
#             tools=[PDFParserTool(), DirectoryReadTool()],
#             role="Expert Summarization Writer",
#             # goal="Write a short summarization of the scientific papers found from the previous task,"
#             # "make sure that all the paper is taken in consideration by using the tools (if given)",
#             # goal="Write a short summarization for each scientific papers found inside the folder 'knowledge'",
#             goal="Write a short summarization on the scientific paper found inside the folder 'knowledge'",
#             backstory="You are an experience writer in summarizing a text, "
#             "speficially scientific papers, always capable to select the most relevant informations",
#             verbose=True,
#         )

#     @agent
#     def reviewer(self) -> Agent:
#         return Agent(
#             llm=llm,
#             # llm="gemini/gemini-2.0-flash",
#             role="Academic Reviewer Specialist",
#             goal="""To meticulously analyze a given academic paper, identifying
#             its core contributions, strengths (pros), and areas for improvement
#             or shortcomings (limitations), and structure these findings into two
#             distinct, well-reasoned lists""",
#             backstory="""You spent a decade as a peer reviewer for top-tier
#             academic journals in your field. You are known for your objective,
#             balanced, and sharp analytical mind. You believe that all research,
#             no matter how groundbreaking, has inherent boundaries, and your sole
#             purpose is to provide an honest, unbiased, and constructive
#             critique. Your reviews are concise, insightful, and strictly focus
#             on the scientific merit and applicability of the work""",
#             verbose=True,
#         )

#     # @agent
#     # def aggregator(self) -> Agent:
#     #     return Agent(
#     #         # config=self.agents_config["paper_writer_agent"],  # type: ignore[index]
#     #         llm=llm,
#     #         # llm="gemini/gemini-2.0-flash",
#     #         role="Expert Summarization WriterAcademic Synthesizer and Narrative Editor",
#     #         goal="""To generate a single, continuous, and easy-to-read document
#     #         (400–600 words) that seamlessly integrates the content of 5–6
#     #         individual paper summaries, ensuring a logical progression of ideas
#     #         from start to finish.
#     #         """,
#     #         backstory="""you are the Lead Synthesis Editor for a top-tier
#     #         science communication firm. Your sole job is to take fragmented,
#     #         specialized research summaries and fuse them into a single,
#     #         cohesive, and compelling narrative for executive and public
#     #         audiences""",
#     #         verbose=True,
#     #     )

#     # @task
#     # def research_task(self) -> Task:
#     #     return Task(
#     #         config=self.tasks_config["research_task"],  # type: ignore[index]
#     #         agent=self.researcher(),
#     #     )

#     @task
#     def scrape_task(self) -> Task:
#         return Task(
#             config=self.tasks_config["scraper_task"],  # type: ignore[index]
#             agent=self.writer(),
#         )

#     @task
#     def summarize_task(self) -> Task:
#         return Task(
#             config=self.tasks_config["summarize_task"],  # type: ignore[index]
#             agent=self.writer(),
#         )

#     @task
#     def reviewer_task(self) -> Task:
#         return Task(
#             config=self.tasks_config["reviewer_task"],  # type: ignore[index]
#             agent=self.reviewer(),
#         )

#     # @task
#     # def aggregate_task(self) -> Task:
#     #     return Task(
#     #         config=self.tasks_config["aggregate_task"],  # type: ignore[index]
#     #         agent=self.aggregator(),
#     #     )

#     @crew
#     def crew(self) -> Crew:
#         return Crew(
#             agents=[
#                 # self.researcher(),
#                 self.writer(),
#                 # self.aggregator(),
#                 self.reviewer(),
#             ],
#             tasks=[
#                 # self.research_task(),
#                 self.scrape_task(),
#                 self.summarize_task(),
#                 self.reviewer_task(),
#                 # self.aggregate_task(),
#             ],
#             process=Process.sequential,
#         )

# @CrewBase
# class AggregateCrew:

#     agents_config = "config/agents.yaml"
#     tasks_config = "config/tasks.yaml"
#     @agent
#     def aggregator(self) -> Agent:
#         return Agent(
#             # config=self.agents_config["paper_writer_agent"],  # type: ignore[index]
#             llm=llm,
#             # llm="gemini/gemini-2.0-flash",
#             role="Expert Summarization WriterAcademic Synthesizer and Narrative Editor",
#             goal="""To generate a single, continuous, and easy-to-read document
#             (400–600 words) that seamlessly integrates the content of 5–6
#             individual paper summaries, ensuring a logical progression of ideas
#             from start to finish.
#             """,
#             backstory="""you are the Lead Synthesis Editor for a top-tier
#             science communication firm. Your sole job is to take fragmented,
#             specialized research summaries and fuse them into a single,
#             cohesive, and compelling narrative for executive and public
#             audiences""",
#             verbose=True,
#         )

#     @task
#     def aggregate_task(self) -> Task:
#         return Task(
#             config=self.tasks_config["aggregate_task"],  # type: ignore[index]
#             agent=self.aggregator(),
#         )

#     @crew
#     def crew(self) -> Crew:
#         return Crew(
#             agents=[
#                 self.aggregator(),
#             ],
#             tasks=[
#                 self.aggregate_task(),
#             ],
#             process=Process.sequential,
#         )

class ResearcherState(BaseModel):
    parsed_papers: List[ParsedText] = []
    summaries: List[Summary] = []
    # pros_and_cons: List[ProsCons]

class ResearcherFlow(Flow[ResearcherState]):
    
    @start()
    def research_interesting_papers(self):
        print("Starting to look for interesting papers on topic")

        output = (
            ResearcherCrew()
            .crew()
            .kickoff(inputs={"topic": "llm for summarization"})
        )

        self.state.parsed_papers = output["parsed_papers"]

    @listen(research_interesting_papers)
    async def summarize_papers(self):
        print("starting summarazing the content")
        tasks = []

        async def write_single_summary(parsed_text):
            output = (
                SummarizationCrew()
                .crew()
                .kickoff( inputs={ "paper": parsed_text.parsed_text } )
            )
            print("IM HERE!!!!!")
            summ = output["summary"]
            summary = Summary(summary=summ)
            return summary
        
        for raw_paper in self.state.parsed_papers:
            task = asyncio.create_task(write_single_summary(raw_paper))
            tasks.append(task)

        summaries = await asyncio.gather(*tasks)
        print("finished writing all the summaries")
        self.state.summaries.extend(summaries)
        
    @listen(summarize_papers)
    async def aggregate_results(self):
        print("Aggregating all the summarises in a single block")
        all_summaries_string = [summary.summary for summary in self.state.summaries]
        all_summaries = " ".join(all_summaries_string)
        print(all_summaries)
        output = (
            AggregateCrew()
            .crew()
            .kickoff(
                inputs={"summaries": all_summaries}
            )
        )

        print(output)

def kickoff():
    researcher_flow= ResearcherFlow()
    researcher_flow.kickoff()

def plot():
    researcher_flow= ResearcherFlow()
    researcher_flow.plot()

if __name__ == "__main__":

    kickoff()



