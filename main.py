import os
import requests
import json
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables with defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-ai/DeepSeek-V3")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.hyperbolic.xyz/v1/chat/completions")

if not OPENAI_API_KEY:
    raise Exception("Please set your OPENAI_API_KEY in the .env file or as an environment variable")


def call_openai_api(prompt, temperature, max_tokens=1500):
    """Make a request to OpenAI's API with the given parameters."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }

    response = requests.post(OPENAI_API_URL, json=data, headers=headers, stream=True)
    if response.status_code != 200:
        raise Exception(f"OpenAI request failed with status code: {response.status_code}, {response.text}")

    full_response = ""
    for line in response.iter_lines():
        if line:
            # Remove 'data: ' prefix and skip empty lines
            line = line.decode('utf-8')
            if line.startswith('data: '):
                line = line[6:]  # Remove 'data: ' prefix
                if line == '[DONE]':
                    break
                try:
                    json_object = json.loads(line)
                    if len(json_object['choices']) > 0:
                        delta = json_object['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content = delta['content']
                            print(content, end='', flush=True)
                            full_response += content
                except json.JSONDecodeError:
                    continue

    print()  # Add a newline at the end
    return full_response.strip()


def babble_agent(prompt):
    """Generate creative ideas (Babble agent) using high temperature."""
    babble_prompt = f"You are Babble, a creative agent. Your purpose is to come up with innovative and imaginative ideas. Prompt: {prompt}"
    return call_openai_api(babble_prompt, temperature=0.9)


def prune_agent(babble_response, original_prompt):
    """Evaluate Babble's ideas for feasibility, uniqueness, and prompt adherence (Prune agent) using low temperature."""
    eval_prompt = (
        f"You are Prune, an evaluator whose purpose is to assess creative ideas.\n"
        f"Original Prompt: '{original_prompt}'\n"
        f"Babble's Ideas: '{babble_response}'\n"
        "Evaluate the idea on three criteria (feasibility, uniqueness, and prompt adherence) on a scale of 1-10.\n"
        "Provide your response in the following JSON format:\n"
        "{\n"
        '    "feasibility_score": <1-10>,\n'
        '    "uniqueness_score": <1-10>,\n'
        '    "adherence_score": <1-10>,\n'
        '    "overall_score": <average of the three scores>,\n'
        '    "explanation": "<detailed evaluation explanation>"\n'
        "}"
    )
    response = call_openai_api(eval_prompt, temperature=0.2)

    # Clean up the response by removing markdown code block formatting if present
    response = response.strip()
    if response.startswith("```"):
        # Remove the first line if it contains ```json or just ```
        response = response.split('\n', 1)[1]
    if response.endswith("```"):
        # Remove the last line containing ```
        response = response.rsplit('\n', 1)[0]
    response = response.strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Fallback in case the response isn't valid JSON
        return {
            "feasibility_score": 0,
            "uniqueness_score": 0,
            "adherence_score": 0,
            "overall_score": 0,
            "explanation": "Failed to parse evaluation. Raw response: " + response
        }


def main():
    parser = argparse.ArgumentParser(description='Generate and evaluate creative ideas using Babble and Prune agents.')
    parser.add_argument('prompt', nargs='?', help='The creative prompt to generate ideas for')
    args = parser.parse_args()

    original_prompt = args.prompt if args.prompt else input("Enter your prompt: ")
    print("Original Prompt:", original_prompt)

    target_score = 7.5  # Minimum acceptable overall score
    max_attempts = 5    # Maximum number of attempts to find a good idea
    best_response = None
    best_evaluation = None
    best_score = 0

    for attempt in range(max_attempts):
        print(f"\nAttempt {attempt + 1}/{max_attempts}")
        print("\nBabble Agent generating ideas...\n")
        babble_response = babble_agent(original_prompt)
        print("Babble Agent Response:")
        print(babble_response)

        print("\nPrune Agent evaluating Babble's ideas...\n")
        evaluation = prune_agent(babble_response, original_prompt)
        print("Prune Agent Evaluation:")
        print(f"Feasibility Score: {evaluation['feasibility_score']}/10")
        print(f"Uniqueness Score: {evaluation['uniqueness_score']}/10")
        print(f"Adherence Score: {evaluation['adherence_score']}/10")
        print(f"Overall Score: {evaluation['overall_score']:.1f}/10")
        print("\nDetailed Evaluation:")
        print(evaluation['explanation'])

        if evaluation['overall_score'] > best_score:
            best_score = evaluation['overall_score']
            best_response = babble_response
            best_evaluation = evaluation

        if evaluation['overall_score'] >= target_score:
            print(f"\nSuccess! Found a good idea with score {evaluation['overall_score']:.1f}")
            break
        else:
            print(f"\nScore {evaluation['overall_score']:.1f} is below target {target_score}. Trying again...")

    print("\n=== Final Results ===")
    if best_response and best_evaluation:
        print(f"Best idea found (score: {best_score:.1f}/10):")
        print(best_response)
        print("\nFinal Evaluation:")
        print(best_evaluation['explanation'])
    else:
        print("No valid ideas were generated. Please try again with a different prompt.")


if __name__ == "__main__":
    main()
