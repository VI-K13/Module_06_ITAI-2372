"""AI Product Description Generator: LLM writing with a basic claim-screening safeguard."""

import re
import threading

import torch
from flask import Flask, render_template, request

app = Flask(__name__)

# These are writing instructions. A tone must never change the actual facts.
TONE_GUIDES = {
    "friendly": (
        "Use a warm, welcoming, conversational voice. Use simple, helpful language. "
        "Avoid exaggeration and keep the writing approachable."
    ),
    "professional": (
        "Use a factual, concise, polished business voice. Focus on clear product details. "
        "Avoid exclamation marks, hype, or personal stories."
    ),
    "fun": (
        "Use an upbeat, playful voice with a little personality. "
        "A light exclamation is fine, but do not invent product benefits."
    ),
    "luxury": (
        "Use an elegant, understated editorial voice, as in a modern design catalogue. "
        "Use graceful phrasing and a calm rhythm, not empty luxury buzzwords. "
        "Do not say premium, crafted, exclusive, finest, exceptional quality, "
        "or superior unless these specific details are supplied. No exclamation marks."
    ),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = (
    "Qwen/Qwen2.5-1.5B-Instruct" if DEVICE == "cuda"
    else "Qwen/Qwen2.5-0.5B-Instruct"
)
_generator = None
_generator_lock = threading.Lock()


def get_generator():
    """Load the Hugging Face text-generation pipeline once when needed."""
    global _generator
    with _generator_lock:
        if _generator is None:
            # Option B uses the pipeline class requested in the assignment.
            # A local Colab model does not need a paid API key.
            from transformers import pipeline
            print(f"Loading {MODEL_NAME} with the Transformers pipeline on {DEVICE}...")
            _generator = pipeline(
                task="text-generation",
                model=MODEL_NAME,
                device=0 if DEVICE == "cuda" else -1,
                dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            )
    return _generator


def parse_keywords(raw):
    """Split, trim and de-duplicate comma-separated user-supplied facts."""
    features = []
    for item in raw.split(","):
        value = " ".join(item.strip().split())
        if value and value.casefold() not in [f.casefold() for f in features]:
            features.append(value)
    return features


def make_messages(product, features, tone, feedback=""):
    """Keep the LLM prompt separate from the user interface and claim checks."""
    rules = (
        "You are an e-commerce copywriter. Only the product name and feature list "
        "are supplied product facts. Write an engaging 2-3 sentence product description "
        "for an online store. Describe the provided features accurately. "
        "Never add unlisted materials, certifications, specifications, durations, "
        "performance results, warranties, quality rankings, or extra use cases. "
        "Do not write 'premium materials', 'perfect fit', 'unmatched sound quality', "
        "or 'advanced technology' without evidence. "
        "If details are sparse, keep the description short. "
        "Output only the description, no title, markdown or explanations."
    )
    user_prompt = (
        f"Product: {product}\nUser-supplied features: {', '.join(features)}\n"
        f"Tone instruction: {TONE_GUIDES[tone]}\n"
        "Keep all factual statements grounded in the exact features above."
    )
    if feedback:
        user_prompt += (
            "\nThe last draft contained these potentially unsupported claims: "
            + feedback + ". Remove them. Do not replace them with new claims."
        )
    return [{"role": "system", "content": rules},
            {"role": "user", "content": user_prompt}]


def generate_draft(product, features, tone, feedback=""):
    """Generate a Qwen product description with Hugging Face's pipeline."""
    generator = get_generator()
    messages = make_messages(product, features, tone, feedback)

    # The text-generation pipeline formats chat messages with Qwen's chat template.
    # return_full_text=False keeps the prompt out of the displayed description.
    # The lock protects a shared generator from simultaneous Flask requests.
    with _generator_lock:
        with torch.inference_mode():
            result = generator(
                messages,
                max_new_tokens=125,
                do_sample=False,
                repetition_penalty=1.12,
                return_full_text=False,
                pad_token_id=generator.tokenizer.eos_token_id,
            )

    draft = result[0]["generated_text"]
    # Handle chat outputs that include an assistant message dictionary.
    if isinstance(draft, list):
        draft = draft[-1]["content"]
    draft = draft.strip()
    if not draft:
        raise RuntimeError("The language model returned an empty description.")
    return draft


# COMMON claim patterns are a screening list, NOT a comprehensive fact-checker.
# A claim is allowed if that same wording appears in the user's feature list.
RISKY_CLAIMS = {
    "premium materials": r"\bpremium (?:quality )?materials?\b",
    "superior or unmatched quality": r"\b(?:unmatched|unparalleled|superior|best-in-class|top-of-the-line)\b",
    "unverified technology level": r"\b(?:advanced|cutting-edge|state-of-the-art) (?:technology|noise cancellation|features?)\b",
    "perfect fit": r"\bperfect fit\b|\bfits? perfectly\b",
    "sound-quality promise": r"\b(?:crystal.clear|studio.quality|unmatched|exceptional) sound\b|\bsound clarity\b",
    "battery or charging claim": r"\b(?:battery life|fast charging|quick charge|rechargeable battery)\b",
    "material claim": r"\b(?:stainless steel|genuine leather|organic cotton|BPA.free|aluminum alloy)\b",
    "water resistance": r"\b(?:waterproof|water.resistant|sweatproof)\b",
    "certification or warranty": r"\b(?:certified|FDA.approved|lifetime warranty|money.back guarantee)\b",
    "unsupported use case": r"\b(?:intense workouts?|long flights?|underwater use|extreme sports)\b",
    "absolute promise": r"\b(?:guaranteed|flawless|100%|zero distractions|complete silence)\b",
}


def screen_claims(description, product, features):
    """Flag some common invented claims and numbers not supplied by the user.

    This is deliberately conservative but incomplete; passing does NOT mean
    an AI-generated description is factually verified.
    """
    source_text = " ".join([product] + features).casefold()
    findings = []
    for label, pattern in RISKY_CLAIMS.items():
        for match in re.finditer(pattern, description, flags=re.I):
            if match.group(0).casefold() not in source_text:
                findings.append(label)
                break

    # Specific numbers (e.g., 24-hour battery life) must appear in the input.
    supplied_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", source_text))
    output_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", description))
    if output_numbers - supplied_numbers:
        findings.append("new numbers or measurements")

    if len(description) > 850:
        findings.append("description is too long")
    return list(dict.fromkeys(findings))


def join_features(features):
    """Join the exact supplied words without claiming new specifications."""
    if len(features) == 1:
        return features[0]
    if len(features) == 2:
        return " and ".join(features)
    return ", ".join(features[:-1]) + ", and " + features[-1]


def grounded_fallback(product, features, tone):
    """Create conservative copy only from the input when both LLM drafts are flagged.

    This template is a transparent fallback, not a claim that the LLM was verified.
    """
    feature_text = join_features(features)
    templates = {
        "friendly": (
            f"Meet {product}! Its listed features include {feature_text}. "
            "A simple way to see the details that matter to you."
        ),
        "professional": (
            f"{product} is listed with the following features: {feature_text}. "
            "This description focuses on the supplied specifications without added claims."
        ),
        "fun": (
            f"Say hello to {product}! On the feature list: {feature_text}. "
            "That's a good place to start exploring!"
        ),
        "luxury": (
            f"A considered introduction to {product}. Its defining listed details are "
            f"{feature_text}. An understated presentation of the essentials."
        ),
    }
    return templates[tone]


def generate_description(product, features, tone):
    """Generate, screen, retry once, then use a grounded fallback if needed.

    Returns a result dictionary so the page can explain what happened.
    """
    if not product.strip() or not features or tone not in TONE_GUIDES:
        raise ValueError("A product, features and valid tone are required.")

    all_findings = []
    feedback = ""
    for attempt in range(2):
        draft = generate_draft(product, features, tone, feedback=feedback)
        findings = screen_claims(draft, product, features)
        if not findings:
            return {
                "description": draft,
                "mode": "revised" if attempt else "generated",
                "findings": all_findings,
            }
        all_findings = list(dict.fromkeys(all_findings + findings))
        feedback = ", ".join(findings)

    return {
        "description": grounded_fallback(product, features, tone),
        "mode": "fallback",
        "findings": all_findings,
    }


@app.route("/", methods=["GET", "POST"])
def home():
    """Show the input form and generate the description after submission."""
    result = None
    error = None
    product = ""
    raw_keywords = ""
    tone = "friendly"

    if request.method == "POST":
        product = " ".join(request.form.get("product_name", "").split())
        raw_keywords = request.form.get("keywords", "").strip()
        tone = request.form.get("tone", "friendly")
        features = parse_keywords(raw_keywords)

        # Server-side validation is important even when HTML inputs are required.
        if not product or not features:
            error = "Enter a product name and at least one real product feature."
        elif len(product) > 100 or len(raw_keywords) > 250:
            error = "The product name or feature list is too long."
        elif len(features) > 10 or any(len(f) > 70 for f in features):
            error = "Use at most 10 short, comma-separated features."
        elif tone not in TONE_GUIDES:
            error = "Choose a valid writing tone."
        else:
            try:
                result = generate_description(product, features, tone)
            except Exception:
                app.logger.exception("Text generation failed")
                error = "The model could not generate text. Check Colab output and try again."

    return render_template(
        "index.html",
        result=result,
        error=error,
        product_name=product,
        keywords=raw_keywords,
        tone=tone,
        tone_options=TONE_GUIDES,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
