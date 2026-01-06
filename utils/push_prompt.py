from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
import yaml
import argparse
import os


def push_prompt(
    client: Client,
    prompt_name: str,
    prompt_path: str,
    tags: list[str],
) -> str:
    # check if prompt_path exists and is a yaml file
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file {prompt_path} not found")
    if not prompt_path.endswith(".yaml"):
        raise ValueError(f"Prompt file {prompt_path} is not a yaml file")

    with open(prompt_path, "r") as file:
        prompt = yaml.safe_load(file)
    prompt_template = ChatPromptTemplate.from_template(prompt["prompt"])
    url = client.push_prompt(prompt_name, object=prompt_template, tags=tags)
    if url:
        print(f"Prompt {prompt_name} pushed to LangSmith: {url}")
    else:
        raise ValueError(f"Failed to push prompt {prompt_name} to LangSmith")

    return url


def main():
    parser = argparse.ArgumentParser(description="Push a prompt to LangSmith")
    parser.add_argument("--prompt_name", type=str, help="The name of the prompt")
    parser.add_argument("--prompt_path", type=str, help="The path to the prompt file")
    parser.add_argument("--tags", nargs="*", default=[], help="The tags to add to the prompt")
    args = parser.parse_args()

    if isinstance(args.tags, str):
        args.tags = [args.tags]

    client = Client()
    url = push_prompt(client, args.prompt_name, args.prompt_path, args.tags)
    print(f"Prompt {args.prompt_name} pushed to LangSmith:\n {url}")


if __name__ == "__main__":
    main()
