def generate_blog_components(topic):
    content = ""
    comeco = False
    #Create a cache directory if it doesn't exist
    os.makedirs(".cache/posts", exist_ok=True)
    cache_key = hashlib.sha256(topic.encode()).hexdigest()
    cache_path = f".cache/posts/{cache_key}.yaml"
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = f.read()
        if "IMAGE_PROMPT:" not in cached:
            print("CACHE MISS - Re-generating Image")
        else:
            print(f"💾 Cached blog retrieved for: {topic}")
            return cached

    # Generate metadata front matter via LLM
    meta_prompt = f"""
You are a witty and conversational tech blogger crafting a tutorial on "{topic}." Using up to 300 tokens, output only valid YAML front-matter fenced with triple dashes. Fill in each field thoughtfully—no placeholders. Also:
- Suggest a `hero_image_prompt` for the article's header.
- Include a list field `inline_image_prompts` for images placed within sections.
---
title: ""
meta_title: ""
meta_desc: ""
slug: ""
keyphrase: ""
synonyms: []
categories: []
tags: []
hero_image_prompt: ""
inline_image_prompts: []
alt_text: ""
---
"""
    with requests.Session() as session:
        meta_resp = session.post(
            f"http://{OLLAMA_SERVER}/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": meta_prompt,
                "temperature": 0.5,
                "max_tokens": 300,
                "stream": False,
                "device": "cuda",
            },
            timeout=300
        )
        meta_resp.raise_for_status()
        meta_data = meta_resp.json().get("response", "").strip()
    print("🔍 [DEBUG] meta_data from LLM (first 300 chars):")
    print(meta_data[:300].replace("\n", "\\n"))
    # Start full_raw with metadata front matter
    full_raw = meta_data + "\n"
    # Initialize chaining context
    context_accum = full_raw

    # Generate outline directly from YAML sections in metadata if available
    try:
        # Parse the metadata YAML
        import yaml
        meta_parsed = yaml.safe_load(meta_data.split('---')[1])
        if meta_parsed and 'sections' in meta_parsed and isinstance(meta_parsed['sections'], list):
            outline = meta_parsed['sections']
            print(f"✅ Using sections from metadata: {outline}")
        else:
            # Generate outline using the dedicated function
            outline = generate_blog_outline(topic)
    except Exception as e:
        print(f"⚠️ Failed to parse sections from metadata: {e}")
        # Generate outline using the dedicated function
        outline = generate_blog_outline(topic)
        
    # Only keep the first 4-5 sections for deeper exploration
    outline = outline[:5]
    # For each section heading, generate its content
    for section in outline:

        section_prompt = f"""
Write the next section titled "{section}" in a friendly, engaging style—imagine you're explaining to a curious friend. 
Use smooth transitions, a bit of humor, and emphasize clarity.

Your output should include:
- At least one **specific comparison, benchmark, stat, or quantified insight** (real or plausible) relevant to the topic.
- A **real-world use case or anecdote** that illustrates the core point or claim.
- Avoid vague or generic claims—ground the section in reality with a concrete example, data point, or mini-case study.
- It's okay to be witty or over-the-top, but never at the expense of clarity or informativeness.

When it fits naturally(don't over use it), insert image placeholders like [IMAGE: description of scene]. Only output the section content.
"""     
        # Minimal feedback for section generation
        print(f"🔨 Generating section content: {section}")
        sec_resp = requests.post(
            f"http://{OLLAMA_SERVER}/api/generate",
            json={
                "model": "llama3:8b",
                "prompt": section_prompt,
                "temperature": 0.7,
                "max_tokens": 1200,
                "stream": False,
                "device": "cuda",
            },
            timeout=600
        )
        sec_data = sec_resp.json()
        section_text = sec_data.get("response", "").strip()
        # Sanitize markdown headings or bold lines that could break YAML
        section_text = re.sub(r'^\s*\*\*(.*?)\*\*', r'\1', section_text, flags=re.MULTILINE)
        section_text = re.sub(r'^#+\s*(.*)', r'\1', section_text, flags=re.MULTILINE)
        # Strip any nested YAML front-matter
        section_text = _strip_frontmatter(section_text)
        # Remove a repeated section heading if present as first line
        lines = section_text.splitlines()
        if lines and lines[0].strip().startswith(section):
            section_text = "\n".join(lines[1:]).strip()
        # Update chaining context with this section
        context_accum += section_text + "\n"
        # Append each section to full_raw (metadata remains at top)
        full_raw += f"{section_text}\n"
    # Write full_raw to cache and use as the response_text
    with open(cache_path, "w") as f:
        f.write(full_raw)
    response_text = full_raw

    # Return the full raw content (with [IMAGE: ...] placeholders intact)
    return full_raw
