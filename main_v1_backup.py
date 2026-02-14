import re
import textwrap
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Content Repurposer API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

API_KEY = "demo-key-2024"

SUPPORTED_FORMATS = [
    {"id": "twitter", "name": "Twitter / X Thread", "description": "280-char tweets optimized for engagement"},
    {"id": "linkedin", "name": "LinkedIn Post", "description": "Professional post with emojis and hashtags"},
    {"id": "email", "name": "Email Newsletter", "description": "Newsletter-ready email format"},
    {"id": "summary", "name": "Executive Summary", "description": "Concise summary for decision makers"},
]


# --- Pydantic Models ---

class RepurposeRequest(BaseModel):
    content: str
    formats: List[str]


class ExtractPointsRequest(BaseModel):
    content: str


class HeadlineVariantsRequest(BaseModel):
    headline: str


# --- Helpers ---

def verify_api_key(x_api_key: Optional[str] = None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def extract_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_key_points(text: str) -> List[str]:
    """Extract key bullet points from content."""
    sentences = extract_sentences(text)
    if not sentences:
        return ["No content provided."]

    points = []
    # Take the first sentence as the thesis
    if len(sentences) >= 1:
        points.append(sentences[0])

    # Grab sentences that contain signal words
    signal_words = [
        "important", "key", "critical", "essential", "significant",
        "must", "should", "need", "result", "because", "therefore",
        "however", "conclusion", "ultimately", "first", "finally",
        "benefit", "advantage", "strategy", "tip", "step",
    ]
    for s in sentences[1:]:
        lower = s.lower()
        if any(w in lower for w in signal_words):
            if s not in points:
                points.append(s)
        if len(points) >= 7:
            break

    # If we still have fewer than 3 points, pad with remaining sentences
    for s in sentences:
        if s not in points:
            points.append(s)
        if len(points) >= 5:
            break

    return points


def repurpose_twitter(content: str) -> dict:
    """Create a Twitter/X thread from content."""
    sentences = extract_sentences(content)
    key_points = extract_key_points(content)

    tweets = []

    # Hook tweet
    hook = sentences[0] if sentences else content[:250]
    if len(hook) > 250:
        hook = hook[:247] + "..."
    tweets.append(f"{hook}\n\nA thread:")

    # Body tweets from key points
    for i, point in enumerate(key_points[:5], 1):
        tweet_text = point
        if len(tweet_text) > 270:
            tweet_text = tweet_text[:267] + "..."
        tweets.append(f"{i}/ {tweet_text}")

    # Closing tweet
    tweets.append(
        "If you found this valuable:\n"
        "- Repost to share with your audience\n"
        "- Follow for more content like this\n"
        "- Drop a comment with your thoughts"
    )

    return {
        "format": "twitter",
        "label": "Twitter / X Thread",
        "thread": tweets,
        "tweet_count": len(tweets),
        "output": "\n\n---\n\n".join(tweets),
    }


def repurpose_linkedin(content: str) -> dict:
    """Create a LinkedIn post from content."""
    sentences = extract_sentences(content)
    key_points = extract_key_points(content)

    # Hook line
    hook = sentences[0] if sentences else content[:200]
    if len(hook) > 200:
        hook = hook[:197] + "..."

    body_points = ""
    emojis = ["->", "->", "->", "->", "->"]
    for i, point in enumerate(key_points[:5]):
        emoji = emojis[i % len(emojis)]
        body_points += f"\n{emoji} {point}\n"

    # Build the post
    post = (
        f"{hook}\n\n"
        f"Here's what I learned:\n"
        f"{body_points}\n"
        f"---\n\n"
        f"What's your take on this? Share your thoughts below.\n\n"
        f"#ContentMarketing #GrowthHacking #MarketingTips #ContentStrategy #DigitalMarketing"
    )

    return {
        "format": "linkedin",
        "label": "LinkedIn Post",
        "output": post,
        "character_count": len(post),
    }


def repurpose_email(content: str) -> dict:
    """Create an email newsletter format."""
    sentences = extract_sentences(content)
    key_points = extract_key_points(content)

    # Subject line
    first_sentence = sentences[0] if sentences else "Important Update"
    if len(first_sentence) > 60:
        subject = first_sentence[:57] + "..."
    else:
        subject = first_sentence

    # Preview text
    preview = sentences[1] if len(sentences) > 1 else first_sentence

    # Body
    bullet_points = "\n".join([f"  * {p}" for p in key_points[:5]])

    # Build the full content snippet (first ~500 chars)
    body_content = " ".join(sentences[:6])
    if len(body_content) > 500:
        body_content = body_content[:497] + "..."

    email = (
        f"Subject: {subject}\n"
        f"Preview: {preview}\n\n"
        f"---\n\n"
        f"Hi there,\n\n"
        f"{body_content}\n\n"
        f"KEY TAKEAWAYS:\n\n"
        f"{bullet_points}\n\n"
        f"---\n\n"
        f"Want to learn more? Reply to this email -- I read every response.\n\n"
        f"Until next time,\n"
        f"Your Team"
    )

    return {
        "format": "email",
        "label": "Email Newsletter",
        "output": email,
        "subject_line": subject,
        "preview_text": preview,
    }


def repurpose_summary(content: str) -> dict:
    """Create an executive summary."""
    sentences = extract_sentences(content)
    key_points = extract_key_points(content)

    word_count = len(content.split())
    summary_sentences = sentences[:3]
    summary_text = " ".join(summary_sentences)

    bullet_points = "\n".join([f"  - {p}" for p in key_points[:5]])

    conclusion = sentences[-1] if len(sentences) > 3 else summary_sentences[-1] if summary_sentences else ""

    summary = (
        f"EXECUTIVE SUMMARY\n"
        f"{'=' * 40}\n\n"
        f"Overview:\n{summary_text}\n\n"
        f"Key Points:\n{bullet_points}\n\n"
        f"Bottom Line:\n{conclusion}\n\n"
        f"---\n"
        f"Original: {word_count} words | Summary: {len(summary_text.split())} words | "
        f"Compression: {round((1 - len(summary_text.split()) / max(word_count, 1)) * 100)}%"
    )

    return {
        "format": "summary",
        "label": "Executive Summary",
        "output": summary,
        "original_word_count": word_count,
        "summary_word_count": len(summary_text.split()),
    }


FORMAT_HANDLERS = {
    "twitter": repurpose_twitter,
    "linkedin": repurpose_linkedin,
    "email": repurpose_email,
    "summary": repurpose_summary,
}


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/formats")
async def get_formats(x_api_key: Optional[str] = Header(None)):
    verify_api_key(x_api_key)
    return {"formats": SUPPORTED_FORMATS}


@app.post("/api/repurpose")
async def repurpose_content(body: RepurposeRequest, x_api_key: Optional[str] = Header(None)):
    verify_api_key(x_api_key)

    if not body.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    valid_formats = [f["id"] for f in SUPPORTED_FORMATS]
    invalid = [f for f in body.formats if f not in valid_formats]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid formats: {invalid}. Supported: {valid_formats}")

    results = {}
    for fmt in body.formats:
        handler = FORMAT_HANDLERS.get(fmt)
        if handler:
            results[fmt] = handler(body.content)

    return {
        "success": True,
        "input_length": len(body.content),
        "formats_requested": body.formats,
        "results": results,
    }


@app.post("/api/extract-points")
async def extract_points(body: ExtractPointsRequest, x_api_key: Optional[str] = Header(None)):
    verify_api_key(x_api_key)

    if not body.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    points = extract_key_points(body.content)
    return {
        "success": True,
        "points": points,
        "count": len(points),
    }


@app.post("/api/headline-variants")
async def headline_variants(body: HeadlineVariantsRequest, x_api_key: Optional[str] = Header(None)):
    verify_api_key(x_api_key)

    headline = body.headline.strip()
    if not headline:
        raise HTTPException(status_code=400, detail="Headline cannot be empty")

    # Remove trailing punctuation for manipulation
    clean = re.sub(r'[.!?]+$', '', headline).strip()
    words = clean.split()

    # 1. Question form
    question = f"Are You Making This Mistake? {clean}?"

    # 2. How-to
    howto = f"How to {clean} (Step-by-Step Guide)"

    # 3. Listicle
    listicle = f"7 Ways to {clean} That Actually Work"

    # 4. Emotional
    emotional = f"The Surprising Truth About {clean} That Nobody Talks About"

    # 5. Data-driven
    data_driven = f"Study Shows: {clean} Can Increase Results by 312%"

    variants = [
        {"type": "Question", "headline": question},
        {"type": "How-To", "headline": howto},
        {"type": "Listicle", "headline": listicle},
        {"type": "Emotional", "headline": emotional},
        {"type": "Data-Driven", "headline": data_driven},
    ]

    return {
        "success": True,
        "original": headline,
        "variants": variants,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8771)
