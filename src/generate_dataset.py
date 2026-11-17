"""
Synthetic dataset generator.

Idea: build a prompt from a CORE task sentence plus optional COMPONENTS
(context, audience, constraints, output_format, purpose). Because we know which
components we added, the labels are free:

    missing_<component> = 1 if we did NOT add that component.

Noise (typos, lowercase, filler words, shuffled order) keeps the model from
memorising exact templates.

Run:  python -m src.generate_dataset
"""
import random
import pandas as pd

from .config import DATA_DIR, GAPS, GAP_COLS, SEED, TASKS

# --------------------------------------------------------------------------
# Vocabulary pools
# --------------------------------------------------------------------------
LANGS = ["python", "java", "javascript", "c++", "go", "rust", "typescript", "c#"]
CODE_TASKS = [
    "reverse a linked list", "parse a csv file", "sort a list of dictionaries by key",
    "validate an email address", "read a json file and print its keys",
    "find duplicate numbers in an array", "connect to a sqlite database",
    "scrape titles from a web page", "calculate fibonacci numbers",
    "merge two sorted arrays", "count word frequency in a text file",
    "check if a string is a palindrome", "convert celsius to fahrenheit",
    "implement a stack", "send a get request to an api", "generate random passwords",
    "do a binary search", "remove stopwords from text", "rename files in a folder",
    "detect cycles in a graph", "check if a number is prime", "build a todo list app",
    "upload a file to a server", "calculate the average of a list",
]
WRITE_TOPICS = [
    "remote work", "climate change", "my first internship", "the future of ai",
    "healthy eating", "time management", "cybersecurity basics", "space exploration",
    "learning to code", "electric vehicles", "mental health awareness", "startup culture",
    "social media addiction", "open source software", "a rainy day in mumbai",
    "friendship", "online privacy", "city life",
]
RECIPIENTS = ["my professor", "my manager", "a client", "the hr team", "my landlord",
              "a recruiter", "my team", "a customer"]
JOBS = ["data analyst", "software engineer", "marketing", "product manager",
        "machine learning engineer", "graphic designer", "sales", "research"]
PRODUCTS = ["a smart water bottle", "a budgeting app", "wireless earbuds",
            "an online course", "a fitness tracker", "handmade candles"]
DOCS = ["article", "research paper", "meeting transcript", "report", "email thread",
        "chapter", "news story", "blog post", "legal document", "lecture notes"]
SUM_TOPICS = ["climate policy", "quarterly sales", "machine learning", "a product launch",
              "the economy", "a clinical trial", "new regulations", "a football match"]
CONCEPTS = [
    "backpropagation", "neural networks", "recursion", "blockchain", "overfitting",
    "gradient descent", "the transformer architecture", "big o notation",
    "docker containers", "apis", "hash tables", "reinforcement learning", "inflation",
    "photosynthesis", "machine learning", "sql joins", "git branching",
    "convolutional networks", "the cap theorem", "object oriented programming",
    "lstm networks", "attention mechanisms", "decision trees", "normalization",
]
CONCEPT_PAIRS = [("lstm", "gru"), ("sql", "nosql"), ("stack", "queue"), ("tcp", "udp"),
                 ("precision", "recall"), ("cnn", "rnn"), ("bagging", "boosting"),
                 ("git merge", "git rebase")]
ANALYSIS_DATA = ["customer churn data", "website traffic", "sales numbers", "survey responses",
                 "stock prices", "student grades", "social media engagement",
                 "employee attrition data", "weather patterns", "app usage logs"]
COMPARE_PAIRS = [("python", "java"), ("react", "vue"), ("tensorflow", "pytorch"),
                 ("renting", "buying a house"), ("online classes", "offline classes"),
                 ("aws", "azure"), ("flask", "django"), ("mysql", "postgres")]
PROS_CONS = ["remote work", "electric cars", "nuclear energy", "cloud computing",
             "microservices", "online learning", "open source", "social media"]
REVIEW_THINGS = ["my business plan", "this marketing strategy", "my resume", "my essay",
                 "this research proposal", "my project design", "my pitch", "my portfolio"]
BRAIN_TOPICS = ["a hackathon", "a machine learning project", "a youtube channel",
                "a mobile app", "a college fest", "a science exhibition", "a startup",
                "a data science portfolio", "a blog", "a healthcare project", "a podcast",
                "a game", "a weekend trip"]
BRAIN_THINGS = ["my coffee shop", "a pet shop", "a tech startup", "my podcast",
                "a fitness app", "a bakery", "a travel blog", "my band"]

# --------------------------------------------------------------------------
# CORE sentences (the actual task). {x} slots get filled randomly.
# Avoid words that belong to components (short, simple, beginner, bullet...).
# --------------------------------------------------------------------------
def core_coding(r):
    c, l = r.choice(CODE_TASKS), r.choice(LANGS)
    return r.choice([
        f"write a function to {c}", f"implement a program to {c} in {l}",
        f"create a script that can {c}", f"fix the bug in my {l} code to {c}",
        f"how do i {c} in {l}", f"write {l} code to {c}",
        f"debug this {l} program that tries to {c}", f"build a {l} program to {c}",
        f"help me code something to {c}", f"i need a {l} snippet to {c}",
    ])

def core_writing(r):
    t = r.choice(WRITE_TOPICS)
    return r.choice([
        f"write a blog post about {t}", f"draft an email to {r.choice(RECIPIENTS)} about {t}",
        f"write a story about {t}", f"compose a linkedin post on {t}",
        f"write a cover letter for a {r.choice(JOBS)} position",
        f"write a speech on {t}", f"create a product description for {r.choice(PRODUCTS)}",
        f"write an essay on {t}", f"write a poem about {t}",
        f"draft a message to {r.choice(RECIPIENTS)} regarding {t}",
    ])

def core_summarization(r):
    d, t = r.choice(DOCS), r.choice(SUM_TOPICS)
    return r.choice([
        f"summarize this {d} about {t}", f"give me a summary of the {d}",
        f"tldr of this {d}", f"condense the following {d}",
        f"shorten this {d} on {t}", f"write a summary of this {d}",
        f"summarise the {d} on {t}", f"what are the main points of this {d}",
    ])

def core_explanation(r):
    c = r.choice(CONCEPTS)
    a, b = r.choice(CONCEPT_PAIRS)
    return r.choice([
        f"explain {c}", f"what is {c}", f"how does {c} work", f"teach me {c}",
        f"what is the difference between {a} and {b}", f"why is {c} important",
        f"help me understand {c}", f"tell me about {c}", f"how do {a} and {b} differ",
    ])

def core_analysis(r):
    x, (a, b) = r.choice(ANALYSIS_DATA), r.choice(COMPARE_PAIRS)
    return r.choice([
        f"analyze {x}", f"compare {a} and {b}", f"give me pros and cons of {r.choice(PROS_CONS)}",
        f"evaluate {r.choice(REVIEW_THINGS)}", f"review {r.choice(REVIEW_THINGS)}",
        f"what are the trends in {x}", f"find insights in {x}",
        f"critique {r.choice(REVIEW_THINGS)}", f"{a} vs {b} which is better",
    ])

def core_brainstorming(r):
    t, th = r.choice(BRAIN_TOPICS), r.choice(BRAIN_THINGS)
    return r.choice([
        f"give me ideas for {t}", f"suggest names for {th}", f"brainstorm {t}",
        f"i need project ideas about {t}", f"what can i do for {t}",
        f"suggest topics for {t}", f"come up with ideas for {t}",
        f"need creative ideas for {th}",
    ])

CORE = {
    "coding": core_coding, "writing": core_writing, "summarization": core_summarization,
    "explanation": core_explanation, "analysis": core_analysis,
    "brainstorming": core_brainstorming,
}

# --------------------------------------------------------------------------
# COMPONENT sentences
# --------------------------------------------------------------------------
PROJECTS = ["college mini project", "startup landing page", "machine learning course project",
            "data science internship", "client website", "hackathon prototype",
            "research paper", "company newsletter", "mobile app", "portfolio website",
            "final year project", "team presentation"]
ROLES = ["student", "developer", "startup founder", "teacher", "freelancer", "researcher",
         "marketing intern", "data analyst", "second year engineering student"]
AUDIENCES = ["beginners", "college students", "non-technical managers", "a 10 year old",
             "senior engineers", "busy executives", "high school students",
             "complete newcomers", "recruiters", "first year undergraduates",
             "small business owners", "my grandmother"]
TONES = ["formal", "friendly", "professional", "casual", "persuasive", "neutral", "humorous"]
FORMATS = ["bullet points", "a numbered list", "a table", "json", "markdown",
           "a single paragraph", "step by step instructions", "three paragraphs",
           "plain text", "an outline with headings", "a code block with comments"]
PURPOSES = ["impress my interviewer", "submit it for my assignment", "present it to my class",
            "publish it on my blog", "convince my manager", "learn it before my exam",
            "put it in my portfolio", "share it with my team", "get more users to sign up",
            "save time on a repetitive task", "prepare for a viva"]

def comp_context(r, task):
    p, role = r.choice(PROJECTS), r.choice(ROLES)
    return r.choice([
        f"I'm a {role} working on a {p}.", f"This is for my {p}.",
        f"Context: it is part of a {p}.", f"I am currently doing a {p} and need this.",
        f"background: we are building a {p}.", f"My team is working on a {p}.",
        f"I'm a {role} and this is for a {p}.",
    ])

def comp_audience(r, task):
    a = r.choice(AUDIENCES)
    return r.choice([
        f"The audience is {a}.", f"Write it for {a}.", f"It's meant for {a}.",
        f"Target readers are {a}.", f"Explain it as if i were {a}.",
        f"Assume the reader is {a}.", f"This is aimed at {a}.", f"Pitch it to {a}.",
    ])

def comp_constraints(r, task):
    n = r.choice([50, 100, 150, 200, 250, 300])
    pool = [
        f"Keep it under {n} words.", f"Maximum {n} words.", "Do not use jargon.",
        "Avoid technical terms.", f"Use a {r.choice(TONES)} tone.",
        f"Keep the tone {r.choice(TONES)}.", f"No more than {r.choice([3, 5, 6])} sentences.",
        "Make it concise.", "Use simple words only.", "Don't make anything up.",
        f"Limit it to {r.choice([3, 5, 7])} points.",
    ]
    if task == "coding":
        pool += ["Don't use any external libraries.", "It must handle edge cases and errors.",
                 "Do not use recursion.", "It should run in under 2 seconds.",
                 "Follow clean code conventions."]
    return r.choice(pool)

def comp_format(r, task):
    f = r.choice(FORMATS)
    return r.choice([
        f"Give the answer as {f}.", f"Format the output as {f}.", f"Return it in {f}.",
        f"Present it as {f}.", f"Output should be {f}.", f"Respond with {f}.",
        f"Structure the response as {f}.",
    ])

def comp_purpose(r, task):
    p = r.choice(PURPOSES)
    return r.choice([
        f"I need this so that I can {p}.", f"The goal is to {p}.",
        f"I want to use this to {p}.", f"My aim is to {p}.", f"Purpose: {p}.",
        f"I'm doing this to {p}.",
    ])

COMPONENTS = {
    "context": comp_context, "audience": comp_audience, "constraints": comp_constraints,
    "output_format": comp_format, "purpose": comp_purpose,
}


# Inline variants: "explain X to beginners in under 100 words" style. Real users often
# write this way instead of separate sentences.
def inline_audience(r):
    return r.choice([f" for {r.choice(AUDIENCES)}", f" to {r.choice(AUDIENCES)}",
                     f" aimed at {r.choice(AUDIENCES)}"])

def inline_constraints(r):
    n = r.choice([50, 100, 150, 200, 250, 300])
    return r.choice([f" in under {n} words", f" in {r.choice([3, 5, 6])} sentences",
                     f" in a {r.choice(TONES)} tone", f" within {n} words", " without jargon",
                     " and keep it brief"])

def inline_format(r):
    f = r.choice(FORMATS)
    return r.choice([f" as {f}", f" in {f}", f" using {f}"])

INLINE = {"audience": inline_audience, "constraints": inline_constraints,
          "output_format": inline_format}

# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------
FILLERS = ["hey ", "please ", "pls ", "can you ", "i want you to ", "hi, ", "could you ",
           "i need you to ", "yo ", "quick question: "]

def typo(word, r):
    if len(word) > 4:
        i = r.randrange(1, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word

def add_noise(text, r):
    if r.random() < 0.15:  # a couple of typos
        words = text.split()
        for _ in range(r.choice([1, 2])):
            j = r.randrange(len(words))
            words[j] = typo(words[j], r)
        text = " ".join(words)
    if r.random() < 0.5:
        text = text.lower()
    if r.random() < 0.3:
        text = text.rstrip(".")
    return text

# --------------------------------------------------------------------------
def make_example(r):
    task = r.choice(TASKS)
    core = CORE[task](r)
    if r.random() < 0.25:
        core = r.choice(FILLERS) + core
    richness = r.uniform(0.1, 0.7)
    present, parts = {}, []
    for g in GAPS:
        present[g] = r.random() < richness
        if not present[g]:
            continue
        if g in INLINE and r.random() < 0.4:   # attach to the core sentence
            core += INLINE[g](r)
        else:
            parts.append(COMPONENTS[g](r, task))
    r.shuffle(parts)
    core = core[0].upper() + core[1:] if r.random() < 0.6 else core
    if not core.endswith((".", "?")) and parts:
        core += "."
    if parts and r.random() < 0.15:           # sometimes component first
        sentences = [parts[0]] + [core] + parts[1:]
    else:
        sentences = [core] + parts
    text = add_noise(" ".join(sentences), r)
    row = {"text": text, "task": task}
    for g in GAPS:
        row[f"missing_{g}"] = 0 if present[g] else 1
    return row

def main(n=12000):
    r = random.Random(SEED)
    rows, seen = [], set()
    while len(rows) < n:
        ex = make_example(r)
        if ex["text"] in seen:
            continue
        seen.add(ex["text"])
        rows.append(ex)
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_train, n_val = int(0.7 * n), int(0.15 * n)
    splits = {"train": df[:n_train], "val": df[n_train:n_train + n_val],
              "test": df[n_train + n_val:]}
    DATA_DIR.mkdir(exist_ok=True)
    for name, part in splits.items():
        part.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows")
    print("\nMissing-rate per part (train):")
    print(splits["train"][GAP_COLS].mean().round(2).to_string())
    print("\nExamples:")
    for _, row in df.head(5).iterrows():
        print(" -", row["text"], "|", row["task"], "|", [int(row[c]) for c in GAP_COLS])

if __name__ == "__main__":
    main()
