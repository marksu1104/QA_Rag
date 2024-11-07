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
        self.question_data = self.load_json(question_file)


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

    def get_incorrect_answers(self):
        output_dict = {item['qid']: item for item in self.output_data['answers']}
        ground_truth_dict = {item['qid']: item for item in self.ground_truth_data['ground_truths']}
        question_dict = {item['qid']: item for item in self.question_data['questions']}

        incorrect_answers = []

        for qid, ground_truth_item in ground_truth_dict.items():
            if qid not in output_dict or output_dict[qid]['retrieve'] != ground_truth_item['retrieve']:
                incorrect_answers.append({
                    'qid': qid,
                    'query': question_dict[qid]['query'],
                    'source': question_dict[qid]['source'],
                    'expected': ground_truth_item['retrieve'],
                    'actual': output_dict[qid]['retrieve'] if qid in output_dict else None,
                    'category': ground_truth_item['category']
                })

        return incorrect_answers

    def output_incorrect_answers(self, output_path='incorrect_answers.json'):
        incorrect_answers = self.get_incorrect_answers()
        # with open(output_path, 'w', encoding='utf-8') as file:
        #     json.dump(incorrect_answers, file, ensure_ascii=False, indent=4)
        # print(f"Incorrect answers saved to {output_path}")

        # Print incorrect answers by category
        incorrect_by_category = {}
        for answer in incorrect_answers:
            category = answer['category']
            if category not in incorrect_by_category:
                incorrect_by_category[category] = []
            incorrect_by_category[category].append(answer)

        for category, answers in incorrect_by_category.items():
            print(f"\nCategory: {category}")
            for answer in answers:
                print(f"  - QID: {answer['qid']}, Expected: {answer['expected']}, Actual: {answer['actual']}, Source: {answer['source']}, Query: {answer['query']}")
                


if __name__ == "__main__":
    output_file = 'output.json'
    ground_truth_file = './dataset/preliminary/ground_truths_example.json'
    question_file = './dataset/preliminary/questions_example.json'
    
    evaluator = Evaluation(output_file, ground_truth_file, question_file)
    
    evaluator.output_evaluation()
