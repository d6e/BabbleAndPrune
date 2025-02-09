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
    """Evaluate Babble's ideas with comprehensive criteria using low temperature."""
    eval_prompt = (
        f"You are Prune, an evaluator whose purpose is to assess creative ideas.\n"
        f"Original Prompt: '{original_prompt}'\n"
        f"Babble's Ideas: '{babble_response}'\n\n"
        "First, provide a detailed evaluation explanation covering all of the following criteria:\n"
        "1. Feasibility - Is it practically implementable?\n"
        "2. Character Identity - Does it fit the character's color pie and thematic elements?\n"
        "3. Design Elegance - Does it clearly and elegantly communicate its design goals?\n"
        "4. Power Level - Is it balanced (neither over/underpowered)?\n"
        "5. Novelty/Creativity - Does it present new ideas rather than redundant effects?\n"
        "6. Purpose - Does it fulfill a clear role or present interesting options?\n"
        "7. Uniqueness - How distinct is it from existing designs?\n"
        "8. Prompt Adherence - How well does it address the original prompt?\n"
        "9. Consistency - Does the language, wording, naming, and text flow match Slay the Spire's style?\n\n"
        "After your explanation, provide numerical scores in the following JSON format:\n"
        "{\n"
        '    "feasibility_score": <1-10>,\n'
        '    "character_identity_score": <1-10>,\n'
        '    "design_elegance_score": <1-10>,\n'
        '    "power_level_score": <1-10>,\n'
        '    "novelty_score": <1-10>,\n'
        '    "purpose_score": <1-10>,\n'
        '    "uniqueness_score": <1-10>,\n'
        '    "adherence_score": <1-10>,\n'
        '    "consistency_score": <1-10>,\n'
        '    "overall_score": <average of all scores>\n'
        "}"
    )
    response = call_openai_api(eval_prompt, temperature=0.2, max_tokens=3000)

    # Clean up any markdown formatting
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    # Split the response into explanation and JSON parts
    parts = response.split('{\n')
    if len(parts) != 2:
        return create_error_response("Failed to parse evaluation. Raw response: " + response)

    explanation = parts[0].strip()
    json_str = '{' + parts[1].strip()

    try:
        scores = json.loads(json_str)
        # Validate scores
        for key in scores:
            if key != 'overall_score' and key.endswith('_score'):
                if not isinstance(scores[key], (int, float)) or scores[key] < 0 or scores[key] > 10:
                    return create_error_response(f"Invalid score for {key}: {scores[key]}")
        
        scores['explanation'] = explanation
        return scores
    except json.JSONDecodeError:
        return create_error_response("Failed to parse evaluation JSON. Raw response: " + response)


def create_error_response(error_msg):
    """Helper function to create error response with zero scores"""
    return {
        "feasibility_score": 0,
        "character_identity_score": 0,
        "design_elegance_score": 0,
        "power_level_score": 0,
        "novelty_score": 0,
        "purpose_score": 0,
        "uniqueness_score": 0,
        "adherence_score": 0,
        "consistency_score": 0,
        "overall_score": 0,
        "explanation": error_msg
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
        
        print("Detailed Evaluation:")
        print(evaluation['explanation'])
        
        print("\nScores:")
        print(f"Feasibility Score: {evaluation['feasibility_score']}/10")
        print(f"Character Identity Score: {evaluation['character_identity_score']}/10")
        print(f"Design Elegance Score: {evaluation['design_elegance_score']}/10")
        print(f"Power Level Score: {evaluation['power_level_score']}/10")
        print(f"Novelty Score: {evaluation['novelty_score']}/10")
        print(f"Purpose Score: {evaluation['purpose_score']}/10")
        print(f"Uniqueness Score: {evaluation['uniqueness_score']}/10")
        print(f"Adherence Score: {evaluation['adherence_score']}/10")
        print(f"Consistency Score: {evaluation['consistency_score']}/10")
        print(f"Overall Score: {evaluation['overall_score']:.1f}/10")

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
