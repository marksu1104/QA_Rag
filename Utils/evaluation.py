import json

class Evaluation:
    def __init__(self, 
                 output_file = 'output.json', 
                 ground_truth_file = './dataset/preliminary/ground_truths_example.json', 
                 question_file = './dataset/preliminary/questions_example.json'):
        
        self.output_file = output_file
        self.ground_truth_file = ground_truth_file

        self.output_data = self.load_json(self.output_file)
        self.ground_truth_data = self.load_json(self.ground_truth_file)


    @staticmethod
    def load_json(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def evaluate_retrieval(self):
        output_dict = {item['qid']: item['retrieve'] for item in self.output_data['answers']}
        ground_truth_dict = {item['qid']: item['retrieve'] for item in self.ground_truth_data['ground_truths']}

        total = len(ground_truth_dict)
        matches = 0

        for qid, retrieve in ground_truth_dict.items():
            if qid in output_dict and output_dict[qid] == retrieve:
                matches += 1

        return matches / total if total > 0 else 0

    def evaluate_retrieval_by_category(self):
        output_dict = {item['qid']: item for item in self.output_data['answers']}
        ground_truth_dict = {item['qid']: item for item in self.ground_truth_data['ground_truths']}

        category_correct = {}
        category_total = {}

        for qid, ground_truth_item in ground_truth_dict.items():
            category = ground_truth_item['category']
            if category not in category_total:
                category_total[category] = 0
                category_correct[category] = 0

            category_total[category] += 1

            if qid in output_dict and output_dict[qid]['retrieve'] == ground_truth_item['retrieve']:
                category_correct[category] += 1

        category_accuracy = {category: (category_correct[category] / category_total[category]) * 100
                             for category in category_total}

        return category_accuracy
    
    def output_evaluation(self):
        # Calculate overall accuracy
        accuracy = self.evaluate_retrieval()
        print("< Evaluation by Ground Truths > ")
        print(f"  - Retrieval accuracy: {accuracy * 100:.2f}%")

        # Calculate accuracy by category
        category_accuracy = self.evaluate_retrieval_by_category()
        for category, accuracy in category_accuracy.items():
            print(f"     - Category: [{category}], Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    output_file = 'output.json'
    ground_truth_file = './dataset/preliminary/ground_truths_example.json'
    question_file = './dataset/preliminary/questions_example.json'
    
    evaluator = Evaluation(output_file, ground_truth_file, question_file)
    
    evaluator.output_evaluation()
