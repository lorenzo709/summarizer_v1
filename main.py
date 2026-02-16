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
from src.crews.CorrectionCrew.CorrectionCrew import CorrectionCrew

from src.MyTypes import ParsedText, Summary, ProsCons, PaperFound, Score, Times, PaperInfos
from typing import List
from tools.pdf_parser_no_tool_version import parser
from tools.research_topic import search_and_save_pdf

from pathlib import Path
import time as tm

from codecarbon import EmissionsTracker

load_dotenv()

class ResearcherState(BaseModel):
    topic: str = ""
    parsed_papers: List[ParsedText] = []
    summaries: List[Summary] = []
    pros_and_cons: List[ProsCons] = []
    gaps_in_SOTA: str = ""
    papers_infos : List[PaperInfos] = []
    times: List[Times] = []

class ResearcherFlow(Flow[ResearcherState]):
    
    @start()
    def research_interesting_papers(self):
        print("Starting to look for interesting papers on topic")
        # self.state.topic = "Vision Transformers (ViT)"
        # self.state.topic = "catalytic water splitting on platinum"
        # self.state.topic = "Zero-Shot Robot Manipulation via CLIP-based Spatial Reasoning."
        self.state.topic = "Retrieval-Augmented Generation for Legacy Code Refactoring."
        start = tm.perf_counter()
        output = (
            ResearcherCrew()
            .crew()
            .kickoff(inputs={"topic":self.state.topic})
        )
        # print("CREW 1 FINISHED")
        parsed_papers = []
        papers_to_parse = []

        # this part is for debugging, return the list of downloaded papers

        # Adding "manual research"
        # topic = "Vision Transformers (ViT) 2025"
        # topic = "catalytic water splitting on platinum"
        # papers_info = search_and_save_pdf(topic,"./knowledge" )
        # self.state.papers_infos = papers_info
        folder_path = Path("./knowledge")

        for pdf_file in folder_path.glob("*.pdf"):
            pdf_name = pdf_file.name
            pdf_path = str(pdf_file)
            paper_found = PaperFound(pdf_name=pdf_name,pdf_path=pdf_path)
            papers_to_parse.append(paper_found)

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
        end = tm.perf_counter()
        final_time = (end - start) / 60
        time =Times(section="Search",total_time=final_time,avg_time=final_time)
        self.state.times.append(time)
        print(f"Search finisced in: {final_time:.2f}s")

    @listen(research_interesting_papers)
    async def summarize_papers(self):
        print("starting summarazing the content")
        tasks = []
        times = []

        async def write_single_summary(parsed_text):
            start = tm.perf_counter()
            output = ( 
                SummarizationCrew()
                .crew()
                .kickoff( inputs={ "paper": parsed_text.parsed_text } ) # ADDED ASYNC
            )
            summ = output["summary"]
            # TESTING IF JUDGE IS WORTH FOR SINGLE SUMMARY (ONLY ONCE, ALWAYS)
            output_judge = (
                JudgeCrew()
                .crew()
                .kickoff(
                    inputs={"source_summaries": parsed_text.parsed_text, "final_summary": summ}
                )
            )
            hints = output_judge["hints"]
            output = (
                CorrectionCrew()
                .crew()
                .kickoff(
                    inputs={"original_text": parsed_text.parsed_text, "current_summary": summ, "judge_hints":hints}
                )
            )
            # print(summ)
            # summary = Summary(summary=summ)
            summary = Summary(summary=output["summary"])
            end = tm.perf_counter()
            times.append(end-start)
            return summary
        
        for raw_paper in self.state.parsed_papers:
            task = asyncio.create_task(write_single_summary(raw_paper))
            tasks.append(task)

        summaries = await asyncio.gather(*tasks)
        print("finished writing all the summaries")
        total_time = sum(times) / 60
        avg_time = (sum(times)/len(times))/60 if times else 0
        print(f"total time:{total_time} || average call time:{avg_time}")
        time =Times(section="Summarization",total_time=total_time,avg_time=avg_time)
        self.state.times.append(time)
        self.state.summaries.extend(summaries)

    @listen(research_interesting_papers)
    async def review_papers(self):
        print("starting reviewing the papers")
        tasks = []
        times = []

        async def write_single_review(parsed_text):
            start = tm.perf_counter()
            output = ( 
                ReviewerCrew()
                .crew()
                .kickoff( inputs={ "paper": parsed_text.parsed_text } ) # ADDED ASYNC
            )
            pro_con = output["summary"]
            end = tm.perf_counter()
            times.append(end-start)
            pdf_title = parsed_text.pdf_name
            pro_and_con = ProsCons(paper_name=pdf_title,pros_and_cons=pro_con)
            return pro_and_con
        
        for raw_paper in self.state.parsed_papers:
            task = asyncio.create_task(write_single_review(raw_paper))
            tasks.append(task)

        pros_and_cons = await asyncio.gather(*tasks)
        print("finished writing all the reviews")
        total_time = sum(times) / 60
        avg_time = ( sum(times)/len(times) )/ 60 if times else 0
        print(f"total time:{total_time} || average call time:{avg_time}")
        time =Times(section="Review",total_time=total_time,avg_time=avg_time)
        self.state.times.append(time)
        self.state.pros_and_cons.extend(pros_and_cons)
        
    @listen(review_papers)
    async def find_gaps_in_SOTA(self):
        print("starting finding gaps in SOTA")
        start = tm.perf_counter()
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
        end = tm.perf_counter()
        final_result = output["summary"]
        final_time = (end - start) / 60
        time =Times(section="Gaps",total_time=final_time,avg_time=final_time)
        self.state.times.append(time)
        print(f"total time:{final_time}")
        self.state.gaps_in_SOTA = final_result

    @listen(and_(summarize_papers,review_papers,find_gaps_in_SOTA))
    async def aggregate_results(self):
        print("Aggregating all the summarises in a single block")
        all_summaries_string = [summary.summary for summary in self.state.summaries]
        all_summaries = "\n\n".join(all_summaries_string)
        print(all_summaries)
        start = tm.perf_counter()
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
        end = tm.perf_counter()
        time_final_summary = (end - start) / 60
        print(f"total time for final summary:{time_final_summary}")
        time_aggregate = Times(section="Aggregate",total_time=time_final_summary,avg_time=time_final_summary)
        self.state.times.append(time_aggregate)
        start = tm.perf_counter()
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
                CorrectionCrew()
                .crew()
                .kickoff(
                    inputs={"original_text": all_summaries, "current_summary": final_summary, "judge_hints":hints}
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
        
        end = tm.perf_counter()
        time_judge = (end - start) / 60
        print(f"total time for correction:{time_judge}")
        time = Times(section="Judge",total_time=time_judge,avg_time=time_judge)
        self.state.times.append(time)

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

        # 6. Print papers infos
        # print("Bibliography\n")
        # for paper in self.state.papers_infos:
        #     print(f"{paper.publicationInfo}, {paper.title}, {paper.year}, {paper.pdfUrl}\n")
        # print("-" * 35)

        # 7. Print Times
        print("Times:")
        for time in self.state.times:
            print(f"Section: {time.section}, Total Time: {time.total_time}, Average Time: {time.avg_time}")
        


def kickoff():
    researcher_flow= ResearcherFlow()
    tracker = EmissionsTracker(
        project_name="My Project",
        measure_power_secs= 60,
        save_to_file=True,
        output_dir="./emissions",
    )
    try:
        researcher_flow.kickoff()
    finally:
        tracker.stop()

def plot():
    researcher_flow= ResearcherFlow()
    researcher_flow.plot()

if __name__ == "__main__":

    kickoff()
    plot()



