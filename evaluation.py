import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def evaluate_retrieval(output_file, ground_truth_file):
    output_data = load_json(output_file)
    ground_truth_data = load_json(ground_truth_file)

    output_dict = {item['qid']: item['retrieve'] for item in output_data['answers']}
    ground_truth_dict = {item['qid']: item['retrieve'] for item in ground_truth_data['ground_truths']}

    total = len(ground_truth_dict)
    matches = 0

    for qid, retrieve in ground_truth_dict.items():
        if qid in output_dict and output_dict[qid] == retrieve:
            matches += 1

    return matches / total if total > 0 else 0

if __name__ == "__main__":
    output_file = 'output.json'
    ground_truth_file = 'dataset/preliminary/ground_truths_example.json'
    accuracy = evaluate_retrieval(output_file, ground_truth_file)
    print(f"Retrieval accuracy: {accuracy * 100:.2f}%")