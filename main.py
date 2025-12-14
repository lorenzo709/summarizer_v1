from crewai import LLM, Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.flow.flow import Flow, listen, start, and_
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
from src.crews.ReviewerCrew.ReviewerCrew import ReviewerCrew
from src.crews.GapResearcherCrew.GapResearcherCrew import GapResearcherCrew
from src.crews.JudgeCrew.JudgeCrew import JudgeCrew

from src.MyTypes import ParsedText, Summary, ProsCons, PaperFound, Score
from typing import List
from tools.pdf_parser_no_tool_version import parser

load_dotenv()

class ResearcherState(BaseModel):
    topic: str = ""
    parsed_papers: List[ParsedText] = []
    summaries: List[Summary] = []
    pros_and_cons: List[ProsCons] = []
    gaps_in_SOTA: str = ""

class ResearcherFlow(Flow[ResearcherState]):
    
    @start()
    def research_interesting_papers(self):
        print("Starting to look for interesting papers on topic")
        # self.state.topic = "Vision Transformers (ViT)"
        # output = (
        #     ResearcherCrew()
        #     .crew()
        #     .kickoff(inputs={"topic":self.state.topic})
        # )
        # print("CREW 1 FINISHED")
        parsed_papers = []
        # papers_to_parse = output["papers"]
        papers_to_parse = [
        PaperFound(
        pdf_name="Evo-ViT_ Slow-Fast Token Evolution for Dynamic Vision Transformer.pdf",
        pdf_path="knowledge/Evo-ViT_ Slow-Fast Token Evolution for Dynamic Vision Transformer.pdf"
        ),
        PaperFound(
        pdf_name="PatchRot_ A Self-Supervised Technique for Training Vision Transformers.pdf",
        pdf_path="knowledge/PatchRot_ A Self-Supervised Technique for Training Vision Transformers.pdf"
        ),
        PaperFound(
        pdf_name="Vicinity Vision Transformer.pdf",
        pdf_path="knowledge/Vicinity Vision Transformer.pdf"
        ),
        PaperFound(
        pdf_name="Vision Transformer with Quadrangle Attention.pdf",
        pdf_path="knowledge/Vision Transformer with Quadrangle Attention.pdf"
        ),
        ]
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
                .kickoff( inputs={ "paper": parsed_text.parsed_text } ) # ADDED ASYNC
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

    @listen(research_interesting_papers)
    async def review_papers(self):
        print("starting reviewing the papers")
        tasks = []

        async def write_single_review(parsed_text):
            output = ( 
                ReviewerCrew()
                .crew()
                .kickoff( inputs={ "paper": parsed_text.parsed_text } ) # ADDED ASYNC
            )
            pro_con = output["summary"]
            pdf_title = parsed_text.pdf_name
            pro_and_con = ProsCons(paper_name=pdf_title,pros_and_cons=pro_con)
            return pro_and_con
        
        for raw_paper in self.state.parsed_papers:
            task = asyncio.create_task(write_single_review(raw_paper))
            tasks.append(task)

        pros_and_cons = await asyncio.gather(*tasks)
        print("finished writing all the reviews")
        self.state.pros_and_cons.extend(pros_and_cons)
        
    @listen(review_papers)
    async def find_gaps_in_SOTA(self):
        print("starting finding gaps in SOTA")
        formatted_items = [
            f"{pro_and_con.paper_name}: \n{pro_and_con.pros_and_cons}"
            for pro_and_con in self.state.pros_and_cons
        ]
        pro_limitation_points_input = "\n".join(formatted_items)
        output = (
            GapResearcherCrew()
            .crew()
            .kickoff( inputs={ "topic": self.state.topic, "pro_limitation_points_input": pro_limitation_points_input } )
        )
        final_result = output["summary"]
        self.state.gaps_in_SOTA = final_result

    @listen(and_(summarize_papers,review_papers,find_gaps_in_SOTA))
    async def aggregate_results(self):
        print("Aggregating all the summarises in a single block")
        all_summaries_string = [summary.summary for summary in self.state.summaries]
        all_summaries = "\n\n".join(all_summaries_string)
        print(all_summaries)
        output = (
            AggregateCrew()
            .crew()
            .kickoff(
                inputs={"summaries": all_summaries, "hints":""}
            )
        )
        final_summary = output["summary"]
        print("INITIAL FINAL SUMMARY:\n")
        print(final_summary)
        output_judge = (
            JudgeCrew()
            .crew()
            .kickoff(
                inputs={"source_summaries": all_summaries, "final_summary": final_summary}
            )
        )
        score = output_judge["score"]
        hints = output_judge["hints"]

        times_final_summary_judged = 0
        while score < 5 and times_final_summary_judged < 2:
            output = (
                AggregateCrew()
                .crew()
                .kickoff(
                    inputs={"summaries": all_summaries, "hints":hints}
                )
            )
            final_summary = output["summary"]
            output_judge = (
                JudgeCrew()
                .crew()
                .kickoff(
                    inputs={"source_summaries": all_summaries, "final_summary": final_summary}
                )
            )
            score = output_judge["score"]
            hints = output_judge["hints"]
            times_final_summary_judged += 1
        
        print("--- INDIVIDUAL PAPER ANALYSIS ---")
        print("-" * 35)
        # Assuming the lists are ordered consistently by paper.
        # Since ProsCons and Summary are separate lists in your state, 
        # you'll need to iterate using an index if you assume they match one-to-one.
        # We'll use the shorter list's length to be safe.
        num_papers = min(len(self.state.summaries), len(self.state.pros_and_cons))

        for i in range(num_papers):
            # Get the data for the current paper
            paper_summary = self.state.summaries[i]
            paper_pros_cons = self.state.pros_and_cons[i]

            # 1. Print Paper Name
            # We use paper_pros_cons.paper_name here, assuming it holds the identifier.
            print(f"Paper Name {i + 1}: \n{paper_pros_cons.paper_name}")
            
            # 2. Print Paper Summary
            # Assuming Summary model has a 'summary' field.
            print(f"Paper Summary {i + 1}: \n{paper_summary.summary}")
            
            # 3. Print Pros/Cons
            # Assuming ProsCons model has a 'pros_and_cons' field.
            print(f"Pros/Cons Paper {i + 1}: \n{paper_pros_cons.pros_and_cons}")
            
            # Separator for the next paper
            print("-" * 35)
        
        # --- 2. Final Synthesis ---
        print("\n--- FINAL SYNTHESIS ---")
        print("-" * 35)

        # 4. Print Final Summary (from the Agent's output)
        # This typically represents the overall synthesis or conclusion.
        print(f"Final Summary: {final_summary}")

        # 5. Print Gaps in SOTA
        print(f"Gaps in SOTA: {self.state.gaps_in_SOTA}")

        print("-" * 35)

def kickoff():
    researcher_flow= ResearcherFlow()
    researcher_flow.kickoff()

def plot():
    researcher_flow= ResearcherFlow()
    researcher_flow.plot()

if __name__ == "__main__":

    kickoff()
    plot()



