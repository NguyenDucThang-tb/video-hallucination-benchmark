import json
import os
import argparse


def extract_pred(video_llm_output):
    video_llm_output = video_llm_output.lower()
    if video_llm_output.startswith("yes"):
        return "Yes."
    elif video_llm_output.startswith("no"):
        return "No."
    else:
        return None

def main(predictions):
    total_questions = 0
    total_correct = 0
    for split, pred_dict in predictions.items():
        question_cnt = 0
        correct = 0
        
        for video_key, video_info_with_qa in pred_dict.items():
            for qa in video_info_with_qa['qa']:
                question_cnt += 1
                gt_answer = qa['answer']
                pred = extract_pred(qa['prediction'])
                
                if gt_answer == pred:
                    correct += 1
        total_questions += question_cnt
        total_correct += correct    
        print (f"{split}: ques: {question_cnt}, correct: {correct}, acc: {correct / question_cnt}")
    print (f"overall: ques: {total_questions}, correct: {total_correct}, acc: {total_correct / total_questions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True)
    args = parser.parse_args()

    input_file = args.input_file
    with open(input_file, 'r') as f:
        predictions = json.load(f)
        
    main(predictions)