'''
This script is used to update the agent rules in:
.CLAUDE.md
.openhands/microagents/repo.md

It will read the agent rules from the AGENTS.md file and update the claude and microagents agent rules in the .md files.
'''

AGENTS_MD_PATH = "AGENTS.md"
CLAUDE_MD_PATH = "CLAUDE.md"
MICROAGENTS_MD_PATH = ".openhands/microagents/repo.md"

def read_agents_md(agents_md_path: str) -> str:
    with open(agents_md_path, "r") as file:
        return file.read()

def update_md_with_agents_md(agents_md_path: str, md_paths: list[str]) -> None:
    agents_md_text = read_agents_md(agents_md_path)
    for md_path in md_paths:
        with open(md_path, "r") as file:
            md = file.read()
        # Remove existing agents content if present
        if agents_md_text in md:
            md = md.replace(agents_md_text, "")
        # Append the agents content
        md = md + "\n" + agents_md_text
        with open(md_path, "w") as file:
            file.write(md)

update_md_with_agents_md(AGENTS_MD_PATH, [CLAUDE_MD_PATH, MICROAGENTS_MD_PATH])