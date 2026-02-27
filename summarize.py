import asyncio
from pydantic import BaseModel

from pathlib import Path
import time as tm

from dotenv import load_dotenv

from src.MyTypes import ParsedText, Summary, ProsCons, PaperFound, Score, Times, PaperInfos, SummaryProConsSinglePaper, ResultPipeLine
from typing import List
from tools.pdf_parser_no_tool_version import parser

from src.crews.SummarizationCrew.SummarizationCrew import SummarizationCrew
from src.crews.JudgeCrew.JudgeCrew import JudgeCrew
from src.crews.CorrectionCrew.CorrectionCrew import CorrectionCrew
from src.crews.AggregatorCrew.AggregatorCrew import AggregateCrew


def setup(rp: ResultPipeLine):
    parsed_papers = []
    papers_to_parse = []

    topic = "Liquid Neural Networks for Continuous-time Signal Processing."
    rp.topic = topic
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

async def sum_papers(parsed_papers: List[ParsedText], rp: ResultPipeLine): 
    print("starting summarazing the content")
    tasks = []
    times = []

    THREAD_LIMITER = asyncio.Semaphore(2)

    async def write_single_summary(rp,parsed_text):
        async with THREAD_LIMITER:
            start = tm.perf_counter()
            output = ( 
                await SummarizationCrew()
                .crew()
                .akickoff( inputs={ "paper": parsed_text.parsed_text } ) # ADDED ASYNC
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
            processed_paper = SummaryProConsSinglePaper(
                paper_name = parsed_text.pdf_name,
                summary = summary.summary,
                pros_and_cons = ""
            )
            rp.processed_papers.append(processed_paper)
            return summary

    for raw_paper in parsed_papers:
        task = asyncio.create_task(write_single_summary(rp,raw_paper))
        tasks.append(task)

    summaries = await asyncio.gather(*tasks)
    print("finished writing all the summaries")
    total_time = sum(times) / 60
    avg_time = (sum(times)/len(times))/60 if times else 0
    print(f"total time:{total_time} || average call time:{avg_time}")
    time =Times(section="Summarization",total_time=total_time,avg_time=avg_time)
    rp.times.append(time)
        

def aggregate_summaries(rp: ResultPipeLine):
    print("Aggregating all the summarises in a single block")
    all_summaries_string = [processed_paper.summary for processed_paper in rp.processed_papers]
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
    rp.times.append(time)
    rp.final_summary = final_summary


def main():
    result_pipeline = ResultPipeLine(
        topic = "",
        processed_papers=[],
        final_summary="",
        gaps_in_SOTA="",
        times=[],
        notes="",
    )
    papers = setup(result_pipeline)
    asyncio.run(sum_papers(papers, result_pipeline))
    aggregate_summaries(result_pipeline)
    print(result_pipeline.model_dump_json())

if __name__ == "__main__":
    main()