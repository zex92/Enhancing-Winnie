import random
import pandas as pd
from query_data import query_rag
from langchain_community.llms.ollama import Ollama
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def query_and_validate(question: str, expected_response: str, model_name: str):
    mistral_response, llama_response = query_rag(question)

    if model_name == "mistral":
        actual_response = mistral_response
    elif model_name == "llama3.1":
        actual_response = llama_response
    else:
        raise ValueError("Invalid model name. Choose 'mistral' or 'llama'.")

    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=actual_response
    )

    model = Ollama(model=model_name)
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(f"Evaluating with {model_name} model")
    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return True, actual_response
    elif "false" in evaluation_results_str_cleaned:
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        return False, actual_response
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

def calculate_token_f1_score(reference, prediction):
    ref_tokens = reference.split()
    pred_tokens = prediction.split()

    common = Counter(ref_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_models(queries, reference_answers):
    selected_indices = random.sample(range(len(queries)), 20)
    mistral_predictions = []
    llama_predictions = []
    correct_mistral = 0
    correct_llama = 0

    for idx in selected_indices:
        question = queries[idx]
        expected_response = reference_answers[idx]
        print(f"Index: {idx}, Question: {' '.join(question.split()[:5])}..., Expected Answer: {' '.join(expected_response.split()[:5])}...")

        print("Mistral Model Evaluation:")
        is_correct, mistral_response = query_and_validate(question, expected_response, model_name="mistral")
        mistral_predictions.append(mistral_response)
        if is_correct:
            correct_mistral += 1

        print("\nLLaMA Model Evaluation:")
        is_correct, llama_response = query_and_validate(question, expected_response, model_name="llama3.1")
        llama_predictions.append(llama_response)
        if is_correct:
            correct_llama += 1

    em_mistral = correct_mistral / len(selected_indices)
    em_llama = correct_llama / len(selected_indices)

    f1_mistral_scores = [calculate_token_f1_score(ref, pred) for ref, pred in zip(reference_answers[:len(mistral_predictions)], mistral_predictions)]
    f1_llama_scores = [calculate_token_f1_score(ref, pred) for ref, pred in zip(reference_answers[:len(llama_predictions)], llama_predictions)]

    f1_mistral = sum(f1_mistral_scores) / len(f1_mistral_scores)
    f1_llama = sum(f1_llama_scores) / len(f1_llama_scores)

    print(f"\nMistral Model - Exact Match Score: {em_mistral}")
    print(f"Mistral Model - F1 Score: {f1_mistral}")
    print(f"\nLLaMA Model - Exact Match Score: {em_llama}")
    print(f"LLaMA Model - F1 Score: {f1_llama}")

if __name__ == "__main__":
    # Load the data from the provided Excel file
    file_path = 'Validation test.xlsx'
    df = pd.read_excel(file_path)

    queries = df['Question'].tolist()
    reference_answers = df['Reference Answer'].tolist()

    evaluate_models(queries, reference_answers)
