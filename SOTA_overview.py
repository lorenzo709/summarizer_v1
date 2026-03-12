import asyncio
from pydantic import BaseModel
import json

from pathlib import Path
import time as tm

from dotenv import load_dotenv

from src.MyTypes import ParsedText, Summary, ProsCons, PaperFound, Score, Times, PaperInfos, SummaryProConsSinglePaper, ResultPipeLine
from typing import List
from tools.pdf_parser_no_tool_version import parser

from src.crews.JudgeCrew.JudgeCrew import JudgeCrew
from src.crews.CorrectionCrew.CorrectionCrew import CorrectionCrew
from src.crews.AggregatorCrew.AggregatorCrew import AggregateCrew

from src.crews.ReviewerCrew.ReviewerCrew import ReviewerCrew
from src.crews.GapResearcherCrew.GapResearcherCrew import GapResearcherCrew


TOPIC = "Liquid Neural Networks for Continuous-time Signal Processing."
MODEL = "llama4:scout"

def setup(rp: ResultPipeLine):
    parsed_papers = []
    papers_to_parse = []

    start = tm.perf_counter()

    folder_path = Path("./knowledge")
    for pdf_file in folder_path.glob("*.pdf"):
        pdf_name = pdf_file.name
        pdf_path = str(pdf_file)
        paper_found = PaperFound(pdf_name=pdf_name,pdf_path=pdf_path)
        papers_to_parse.append(paper_found)

    for paper in papers_to_parse:
        parsed_text = parser(paper.pdf_path)
        pdf_name = paper.pdf_name
        final_paper = ParsedText(pdf_name=pdf_name,parsed_text=parsed_text)
        parsed_papers.append(final_paper)
    end = tm.perf_counter()
    final_time = (end - start) / 60
    time =Times(section="Search",total_time=final_time,avg_time=final_time)
    rp.times.append(time)
    print(f"Search finisced in: {final_time:.2f}s")
    return parsed_papers

async def review_papers(parsed_papers: List[ParsedText], rp: ResultPipeLine): 
    print("starting reviewing the papers")
    tasks = []
    times = []

    start_total = tm.perf_counter()
    THREAD_LIMITER = asyncio.Semaphore(3)

    async def write_single_review(rp,parsed_text):
        async with THREAD_LIMITER:
            start = tm.perf_counter()
            output = await ( 
                ReviewerCrew()
                .crew()
                .kickoff_async( inputs={ "paper": parsed_text.parsed_text } ) # ADDED ASYNC
            )
            pro_con = Summary(summary=output["summary"])
            end = tm.perf_counter()
            times.append(end-start)
            existing_paper = next(
                    (p for p in rp.processed_papers if p.paper_name == parsed_text.pdf_name), 
                    None
                )
            if existing_paper:
                # print(f"Paper '{new_paper_data.paper_name}' already exists. Skipping or updating...")
                existing_paper.pros_and_cons = pro_con.summary
            else:
                processed_paper = SummaryProConsSinglePaper(
                    paper_name = parsed_text.pdf_name,
                    summary = "",
                    pros_and_cons = pro_con.summary
                )
                rp.processed_papers.append(processed_paper)
                # print(f"Added new paper: {new_paper_data.paper_name}")
            return pro_con

    for raw_paper in parsed_papers:
        task = asyncio.create_task(write_single_review(rp,raw_paper))
        tasks.append(task)

    summaries = await asyncio.gather(*tasks)
    print("finished writing all the reviews")

    end_total = tm.perf_counter()
    # total_time = sum(times) / 60
    total_time = (end_total - start_total) / 60
    avg_time = (sum(times)/len(times))/60 if times else 0
    print(f"total time:{total_time} || average call time:{avg_time}")
    time =Times(section="Review",total_time=total_time,avg_time=avg_time)
    rp.times.append(time)
        

def find_gaps_in_SOTA(rp: ResultPipeLine):
    print("starting finding gaps in SOTA")

    start = tm.perf_counter()

    formatted_items = [
        f"{paper.paper_name}: \n{paper.pros_and_cons}"
        for paper in rp.processed_papers
    ]
    pro_limitation_points_input = "\n".join(formatted_items)
    output = (
        GapResearcherCrew()
        .crew()
        .kickoff( inputs={ "topic": rp.topic, "pro_limitation_points_input": pro_limitation_points_input } )
    )
    gaps_in_SOTA = output["summary"]
    print("STATE OF THE ART\n")
    print(gaps_in_SOTA)
    end = tm.perf_counter()
    time_final_summary = (end - start) / 60
    print(f"total time for SOTA:{time_final_summary}")
    time= Times(section="Aggregate",total_time=time_final_summary,avg_time=time_final_summary)
    rp.times.append(time)
    rp.gaps_in_SOTA = gaps_in_SOTA


def main():
    with open('result.json', 'r') as file:
        data = json.load(file)

    # 2. Parse the dictionary into the Pydantic model
    result_pipeline = ResultPipeLine.model_validate(data)    

    papers = setup(result_pipeline)
    asyncio.run(review_papers(papers, result_pipeline))
    find_gaps_in_SOTA(result_pipeline)
    print(result_pipeline.model_dump_json())

    with open ("result.json","w") as f:
        f.write(result_pipeline.model_dump_json())

if __name__ == "__main__":
    main()