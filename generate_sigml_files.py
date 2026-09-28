import re
from pathlib import Path
#27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "sign_4.txt"
OUTPUT_FOLDER = BASE_DIR / "static" / "classSign"

OUTPUT_FOLDER.mkdir(exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

name_pattern = r'"name"\s*:\s*"([^"]+)"'
name_matches = list(re.finditer(name_pattern, text))

used_names = {}
count = 0


def clean_sigml_block(block):
    block = block.strip()
    block = block.replace('\\"', '"')
    block = block.replace("><", ">\n<")
    return block


# 🔥 Bangla-only name extractor
def get_bangla_name(full_name):
    name = full_name.strip()

    # 1. remove bracket part
    name = name.split("(")[0]

    # 2. remove underscore part
    name = name.split("_")[0]

    # 3. keep only Bangla + space
    name = re.sub(r'[^\u0980-\u09FF\s]', '', name)

    # 4. normalize space → underscore
    name = re.sub(r'\s+', '_', name.strip())

    return name


for i, match in enumerate(name_matches):
    full_name = match.group(1).strip()

    bangla_name = get_bangla_name(full_name)

    start = match.end()
    end = name_matches[i + 1].start() if i + 1 < len(name_matches) else len(text)
    entry_text = text[start:end]

    sign_blocks = re.findall(
        r'<hns_sign\b.*?</hns_sign>',
        entry_text,
        flags=re.DOTALL
    )

    if not sign_blocks:
        print(f"Skipped (no sign found): {full_name}")
        continue

    # duplicate handling
    if bangla_name in used_names:
        used_names[bangla_name] += 1
        final_name = f"{bangla_name}_{used_names[bangla_name]}"
    else:
        used_names[bangla_name] = 1
        final_name = bangla_name

    cleaned_blocks = [clean_sigml_block(block) for block in sign_blocks]

    sigml_content = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<sigml>\n'
        + "\n".join(cleaned_blocks) +
        '\n</sigml>\n'
    )

    output_file = OUTPUT_FOLDER / f"{final_name}.sigml"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(sigml_content)

    count += 1
    print(f"Created: {output_file.name}")

print(f"\nDone! Total {count} .sigml files created.")
print(f"Output folder: {OUTPUT_FOLDER}")