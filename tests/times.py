from pathlib import Path
import sys
import os

import glob
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "..", "src")
tool_path = os.path.join(current_dir, "..", "tools")
sys.path.append(os.path.abspath(src_path))
sys.path.append(os.path.abspath(tool_path))

input_folder = "./output_results_judge_on_single_summ"
output_folder = "./times"

os.makedirs(output_folder, exist_ok=True)
json_files = glob.glob(os.path.join(input_folder, "*.json"))


all_times = []

for file_path in json_files:
    file_name = os.path.basename(file_path)

    print(f"Processing: {file_name}") 
    with open(file_path, "r") as f:
        file_content = f.read()
    raw_data = json.loads(file_content)

    topic = raw_data.get("topic","")
    model = raw_data.get("model","")

    time_review = 0
    time_summ = 0 
    time_aggr_judge = 0

    for entry in raw_data.get("times",[]):

        section = entry["section"]
        total_time = entry["total_time"]
        if section == "Review":
            time_review += total_time
        elif section == "Summarization": 
            time_summ += total_time
        elif section in ["Judge","Aggregate"]: 
            time_aggr_judge += total_time
        
    max_time = max(time_review, time_summ)
    final_time = max_time + time_aggr_judge
    time = {
        "model": model,
        "topic": topic, 
        "final_time": final_time
    }
    all_times.append(time)
    print(f"{topic}:{model} = {final_time}")
    print("-" * 30)

folder = os.path.basename(input_folder)
filename = f"times_{folder}.json"
output_path = os.path.join(output_folder, filename)
with open(output_path, "w") as f:
    json.dump(all_times, f, indent=4)

        


    