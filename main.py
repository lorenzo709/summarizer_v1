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

from src.crews.AggregatorCrew.AggregatorCrew import AggregateCrew
from src.crews.ResearcherCrew.ResearcherCrew import ResearcherCrew
from src.crews.SummarizationCrew.SummarizationCrew import SummarizationCrew
from src.crews.ReviewerCrew import ReviewerCrew

from src.MyTypes import ParsedText, Summary, ProsCons
from typing import List
from tools.pdf_parser_no_tool_version import parser

load_dotenv()

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
        print("CREW 1 FINISHED")
        parsed_papers = []
        papers_to_parse = output["papers"]
        for paper in papers_to_parse:
            parsed_text = parser(paper.pdf_path)
            pdf_name = paper.pdf_name
            print(paper.pdf_path)
            print(paper.pdf_name)
            final_paper = ParsedText(pdf_name=pdf_name,parsed_text=parsed_text)
            parsed_papers.append(final_paper)
        print(parsed_papers[0].parsed_text[:100])
        self.state.parsed_papers = parsed_papers
        # self.state.parsed_papers = output["parsed_papers"]

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
            summ = output["summary"]
            print(summ)
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
        all_summaries = "\n\n".join(all_summaries_string)
        print(all_summaries)
        output = (
            AggregateCrew()
            .crew()
            .kickoff(
                inputs={"summaries": all_summaries}
            )
        )
        final_result = output["summary"]
        print(final_result)

def kickoff():
    researcher_flow= ResearcherFlow()
    researcher_flow.kickoff()

def plot():
    researcher_flow= ResearcherFlow()
    researcher_flow.plot()

if __name__ == "__main__":

    kickoff()
    plot()



