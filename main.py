import os
import requests

# Retrieve the API key from the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("Please set your OPENAI_API_KEY as an environment variable")


def call_openai_api(prompt, temperature, max_tokens=150):
    """Make a request to OpenAI's API with the given parameters."""
    url = "https://api.openai.com/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    data = {
        "model": "text-davinci-003",
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        raise Exception(f"OpenAI request failed with status code: {response.status_code}, {response.text}")
    result = response.json()
    return result["choices"][0]["text"].strip()


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
        "Please provide a detailed evaluation focusing on the following aspects: feasibility, uniqueness, and adherence to the original prompt."
    )
    return call_openai_api(eval_prompt, temperature=0.2)


def main():
    original_prompt = "Suggest innovative solutions for a sustainable future."
    print("Original Prompt:", original_prompt)

    print("\nBabble Agent generating ideas...\n")
    babble_response = babble_agent(original_prompt)
    print("Babble Agent Response:")
    print(babble_response)

    print("\nPrune Agent evaluating Babble's ideas...\n")
    prune_response = prune_agent(babble_response, original_prompt)
    print("Prune Agent Evaluation:")
    print(prune_response)


if __name__ == "__main__":
    main()
