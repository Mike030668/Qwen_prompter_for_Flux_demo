from ranger_generation.prompter.qwen_prompter import generate_structured_prompt
import json, sys

if __name__ == "__main__":
    user_text = " ".join(sys.argv[1:]) or input("Your prompt → ")
    result = generate_structured_prompt(user_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
