import re
import textwrap
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import auth and database modules
import database
import auth

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database
    await database.init_database()
    print("✓ Database initialized")
    yield
    # Shutdown
    print("✓ Shutting down")

app = FastAPI(
    title="Content Repurposer API",
    version="2.0.0",
    description="Transform content into multiple formats for different platforms",
    lifespan=lifespan
)

# CORS middleware
import os
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(auth.rate_limit_middleware)
app.middleware("http")(auth.log_request_middleware)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

SUPPORTED_FORMATS = [
    {"id": "twitter", "name": "Twitter / X Thread", "description": "280-char tweets optimized for engagement"},
    {"id": "linkedin", "name": "LinkedIn Post", "description": "Professional post with emojis and hashtags"},
    {"id": "email", "name": "Email Newsletter", "description": "Newsletter-ready email format"},
    {"id": "summary", "name": "Executive Summary", "description": "Concise summary for decision makers"},
    {"id": "instagram", "name": "Instagram Caption", "description": "Caption with emojis and 30 hashtags"},
    {"id": "youtube_description", "name": "YouTube Description", "description": "SEO-optimized video description"},
    {"id": "blog_outline", "name": "Blog Outline", "description": "H2/H3 structured outline"},
    {"id": "podcast_script", "name": "Podcast Script", "description": "Conversational script format"},
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
    "instagram": repurpose_to_instagram,
    "youtube_description": repurpose_to_youtube_description,
    "blog_outline": repurpose_to_blog_outline,
    "podcast_script": repurpose_to_podcast_script,
}


# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/formats")
async def get_formats(key_info: dict = Depends(auth.verify_api_key_dependency)):
    return {"formats": SUPPORTED_FORMATS}


@app.post("/api/repurpose")
async def repurpose_content(body: RepurposeRequest, key_info: dict = Depends(auth.verify_api_key_dependency)):
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
async def extract_points(body: ExtractPointsRequest, key_info: dict = Depends(auth.verify_api_key_dependency)):
    if not body.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    points = extract_key_points(body.content)
    return {
        "success": True,
        "points": points,
        "count": len(points),
    }


@app.post("/api/headline-variants")
async def headline_variants(body: HeadlineVariantsRequest, key_info: dict = Depends(auth.verify_api_key_dependency)):
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

# Phase 2 enhancements: New content formats

def repurpose_to_instagram(content: str) -> dict:
    """Convert content to Instagram caption with emojis and hashtags."""
    sentences = extract_sentences(content)[:3]
    caption = " ".join(sentences)
    
    # Add emojis strategically
    caption = "✨ " + caption
    if len(caption) > 500:
        caption = caption[:500] + "..."
    
    # Generate 30 relevant hashtags
    words = re.findall(r'\b[A-Za-z]{4,}\b', content.lower())
    hashtags = list(set(words))[:30]
    hashtag_string = " ".join([f"#{tag}" for tag in hashtags])
    
    return {
        "format": "instagram",
        "caption": caption,
        "hashtags": hashtag_string,
        "character_count": len(caption)
    }


def repurpose_to_youtube_description(content: str) -> dict:
    """Create SEO-optimized YouTube video description."""
    sentences = extract_sentences(content)
    intro = sentences[0] if sentences else "Check out this content!"
    
    description = f"{intro}\n\n"
    description += "📌 What you'll learn:\n"
    
    key_points = extract_key_points(content)[:5]
    for i, point in enumerate(key_points, 1):
        description += f"{i}. {point[:80]}\n"
    
    description += "\n🔔 Subscribe for more content!\n"
    description += "\n#youtube #content #learning"
    
    return {
        "format": "youtube_description",
        "description": description,
        "word_count": len(description.split())
    }


def repurpose_to_blog_outline(content: str) -> dict:
    """Generate H2/H3 blog outline structure."""
    points = extract_key_points(content)
    
    outline = "# Main Title\n\n"
    outline += "## Introduction\n\n"
    
    for i, point in enumerate(points, 1):
        outline += f"## Section {i}: {point[:50]}...\n"
        outline += f"### Subsection {i}.1\n"
        outline += f"### Subsection {i}.2\n\n"
    
    outline += "## Conclusion\n\n"
    outline += "## Call to Action\n"
    
    return {
        "format": "blog_outline",
        "outline": outline,
        "sections": len(points) + 2
    }


def repurpose_to_podcast_script(content: str) -> dict:
    """Convert to conversational podcast script."""
    sentences = extract_sentences(content)
    
    script = "[INTRO MUSIC]\n\n"
    script += "Host: Welcome back to the podcast! Today we're diving into an amazing topic.\n\n"
    script += f"Host: {sentences[0] if sentences else 'Great content today.'}\n\n"
    
    for i, sentence in enumerate(sentences[1:4], 1):
        script += f"Host: Point {i}: {sentence}\n\n"
    
    script += "Host: Let me know what you think in the comments!\n\n"
    script += "[OUTRO MUSIC]\n"
    
    return {
        "format": "podcast_script",
        "script": script,
        "estimated_duration": "3-5 minutes"
    }


# Update SUPPORTED_FORMATS



# Import error handlers and admin
from error_handlers import register_error_handlers
from admin import verify_admin_key, get_system_stats, get_recent_users, get_usage_by_user

# Register error handlers
register_error_handlers(app)

# --- Admin endpoints ---

@app.get("/api/admin/stats", tags=["Admin"], summary="Get system statistics")
async def admin_stats(admin_key: str = Depends(verify_admin_key)):
    """Get comprehensive system statistics (admin only)."""
    stats = await get_system_stats()
    return {"success": True, **stats}


@app.get("/api/admin/recent-users", tags=["Admin"], summary="Get recent users")
async def admin_recent_users(
    limit: int = 20,
    admin_key: str = Depends(verify_admin_key)
):
    """Get recently registered users (admin only)."""
    users = await get_recent_users(limit=limit)
    return {"success": True, "users": users, "count": len(users)}


@app.get("/api/admin/top-users", tags=["Admin"], summary="Get top users by usage")
async def admin_top_users(
    limit: int = 20,
    admin_key: str = Depends(verify_admin_key)
):
    """Get top users by request volume (admin only)."""
    users = await get_usage_by_user(limit=limit)
    return {"success": True, "users": users, "count": len(users)}
