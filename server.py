from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import httpx
import os
import json
import re
import time
from collections import defaultdict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")

# ─── הגנות ─────────────────────────────────────────────────────
MAX_FILE_MB = 25
RATE_PER_HOUR = 10
RATE_PER_DAY = 30
MAX_TRANSCRIPT = 50000

rate_store = defaultdict(list)

def get_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    return fwd.split(",")[0].strip() if fwd else request.client.host

def check_rate(ip: str):
    now = time.time()
    rate_store[ip] = [t for t in rate_store[ip] if t > now - 86400]
    last_hour = [t for t in rate_store[ip] if t > now - 3600]
    if len(last_hour) >= RATE_PER_HOUR:
        mins = int((min(last_hour) + 3600 - now) / 60)
        raise HTTPException(429, f"יותר מדי בקשות. נסה שוב בעוד {mins} דקות.")
    if len(rate_store[ip]) >= RATE_PER_DAY:
        raise HTTPException(429, f"הגעת למכסה היומית ({RATE_PER_DAY} בקשות). נסה מחר.")
    rate_store[ip].append(now)

# ─── Static ─────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/static/sw.js")
async def sw():
    return FileResponse("static/sw.js", media_type="application/javascript")

@app.get("/static/manifest.json")
async def manifest():
    return FileResponse("static/manifest.json", media_type="application/json")

@app.get("/health")
async def health():
    return {"status": "ok", "time": time.time()}

@app.get("/usage")
async def usage(request: Request):
    ip = get_ip(request)
    now = time.time()
    reqs = [t for t in rate_store.get(ip, []) if t > now - 86400]
    return {
        "last_hour": len([t for t in reqs if t > now - 3600]),
        "last_day": len(reqs),
        "limit_hour": RATE_PER_HOUR,
        "limit_day": RATE_PER_DAY,
    }

# ─── Transcribe ─────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe(request: Request, file: UploadFile = File(...)):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key לא מוגדר")

    check_rate(get_ip(request))

    content = await file.read()

    if len(content) / 1024 / 1024 > MAX_FILE_MB:
        raise HTTPException(413, f"קובץ גדול מדי. מקסימום {MAX_FILE_MB}MB.")
    if len(content) < 500:
        raise HTTPException(400, "הקובץ ריק או קצר מדי.")

    ext = (file.filename or "").lower().split(".")[-1]
    ct = file.content_type or ""
    if not (ct.startswith("audio/") or ct.startswith("video/") or
            ext in ["mp3","mp4","wav","m4a","webm","ogg","flac","aac"]):
        raise HTTPException(400, f"סוג קובץ לא נתמך.")

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                files={"file": (file.filename or "audio.mp3", content, ct or "audio/mpeg")},
                data={"model": "whisper-1", "language": "he",
                      "response_format": "verbose_json",
                      "timestamp_granularities[]": "segment"}
            )
    except httpx.TimeoutException:
        raise HTTPException(504, "פסק זמן — נסה קובץ קצר יותר.")
    except Exception as e:
        raise HTTPException(502, f"שגיאת חיבור: {str(e)[:80]}")

    if r.status_code == 429:
        raise HTTPException(429, "יותר מדי בקשות. נסה שוב בעוד דקה.")
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"שגיאת תמלול: {r.text[:150]}")

    return r.json()

# ─── Summarize (3-stage AI pipeline) ────────────────────────────
ENRICH_SYSTEM = """אתה פרופסור בכיר עם ידע אנציקלופדי. קיבלת תמלול שיעור אקדמי.
עליך לנתח אותו ולהחזיר JSON בלבד עם המפתחות הבאים:

{
  "topics": ["רשימת נושאים מרכזיים"],
  "enriched_concepts": [
    {
      "concept": "שם המושג",
      "lecture_explanation": "מה המורה אמר",
      "better_explanation": "הסבר עמוק יותר שלך עם אינטואיציה",
      "best_analogy": "האנלוגיה הטובה ביותר מהעולם האמיתי",
      "concrete_example": "דוגמה מספרית או קונקרטית מלאה",
      "common_mistakes": ["טעות נפוצה 1", "טעות נפוצה 2"],
      "deeper_insight": "תובנה עמוקה שהמורה לא ציין"
    }
  ],
  "missed_by_lecturer": ["דבר חשוב שהמורה פספס או הסביר בצורה גרועה"],
  "corrections": ["תיקון אם המורה אמר משהו לא מדויק — כתוב בעדינות"],
  "connections_to_other_topics": ["קשר לנושא אחר שעוזר להבנה"],
  "exam_traps": ["מלכודות מבחן נפוצות בנושא זה"]
}"""

SUMMARIZE_SYSTEM = """אתה המורה הטוב ביותר בעולם. קיבלת:
1. תמלול שיעור אקדמי
2. ניתוח מועשר של פרופסור מומחה

צור סיכום שהוא פי 1000 טוב מכל מה שהמורה הסביר.
החזר JSON בלבד, ללא טקסט נוסף.

⚠️ כללי LaTeX — חובה:
• שדות "latex" ו-"formula": LaTeX טהור בלבד — אסור $$ או $ — לדוגמה: \\frac{d}{dx}f(x)
• לעולם אל תכתוב 0 במכנה — כתוב תמיד את הביטוי המלא
• שדות טקסטואליים (content/intuition/analogy/meaning/desc/answer/text וכו'): עברית פשוטה בלבד, אסור $ או LaTeX

מבנה JSON:
{
  "title": "כותרת השיעור",
  "topics": ["נושאים"],
  "enriched": true,
  "sections": [
    {
      "type": "big_picture",
      "content": "תמונה גדולה: מה הנושא, למה הוא קיים, למה חשוב ללמוד אותו, קשר לעולם האמיתי"
    },
    {
      "type": "concept",
      "title": "שם המושג",
      "intuition": "הסבר אינטואיטיבי ראשון — לפני כל הגדרה פורמלית, בשפה פשוטה",
      "analogy": "אנלוגיה מהחיים שכל אחד מבין",
      "formal_definition": "ההגדרה הפורמלית — אחרי שהבנו אינטואיטיבית",
      "concrete_example": "דוגמה מספרית מלאה עם כל שלב מוסבר",
      "common_mistakes": ["⚠️ טעות נפוצה"],
      "deeper_insight": "💡 תובנה שהמורה לא ציין",
      "ai_enrichment": "💎 ידע נוסף מעבר לשיעור"
    },
    {
      "type": "formula",
      "title": "שם הנוסחה",
      "latex": "נוסחה בלטקס מדויקת",
      "what_it_calculates": "מה הנוסחה מחשבת במילים פשוטות",
      "variables_explained": {"x": "מה זה x", "y": "מה זה y"},
      "numerical_example": "דוגמה עם מספרים ממשיים, כל שלב מוסבר",
      "intuition": "למה הנוסחה נראית ככה — ההיגיון מאחוריה"
    },
    {
      "type": "exercise",
      "title": "כותרת התרגיל",
      "problem": "שאלה מלאה ומדויקת",
      "strategy": "אסטרטגיית פתרון — מה נעשה ולמה לפני שמתחילים",
      "steps": [
        {
          "explanation": "מה אנחנו עושים בשלב זה ולמה — משפט מלא",
          "latex": "הנוסחה המדויקת בלטקס",
          "result_meaning": "מה התוצאה אומרת לנו"
        }
      ],
      "conclusion": "מסקנה מלאה עם הסבר למה הגענו לזה",
      "exam_tip": "🎯 איך לענות על שאלה כזו במבחן"
    },
    {
      "type": "mental_model",
      "content": "מודל מנטלי: 4-5 משפטים שהסטודנט יכול לחזור לחבר ברחוב"
    },
    {
      "type": "missed_by_lecturer",
      "items": ["💎 דבר חשוב שהמורה פספס — עם הסבר מלא"]
    },
    {
      "type": "exam_traps",
      "items": ["⚠️ מלכודת מבחן — עם הסבר מה לעשות במקום"]
    }
  ],
  "formulas_tab": [
    {
      "name": "שם הנוסחה",
      "latex": "נוסחה בלטקס",
      "one_line": "הסבר בשורה אחת"
    }
  ],
  "exercises_tab": [
    {
      "title": "שם התרגיל",
      "problem": "השאלה",
      "steps": [
        {"text": "הסבר השלב", "latex": "נוסחה מדויקת"}
      ],
      "answer": "תשובה סופית"
    }
  ],
  "key_points_tab": [
    "🎯 נקודה חשובה — משפט אחד ברור"
  ]
}"""

SUMMARIZE_FALLBACK_SYSTEM = """אתה מומחה לסיכום שיעורים. החזר JSON בלבד.
כללים: LaTeX טהור בשדות latex/formula בלבד (אסור $ או $$). עברית פשוטה בכל שאר השדות.
מבנה: {"title":"נושא","topics":["נושא"],"enriched":false,"sections":[
{"type":"explanation","title":"כותרת","content":"הסבר"},
{"type":"formula","title":"נוסחאות","formulas":[{"name":"שם","latex":"\\\\frac{a}{b}","meaning":"הסבר","source":"voice"}]},
{"type":"exercise","title":"תרגיל","problem":"שאלה","steps":[{"step":1,"desc":"הסבר","formula":""}],"answer":"תשובה"},
{"type":"highlight","title":"חשוב למבחן","points":["נקודה"]}
],"formulas_tab":[],"exercises_tab":[],"key_points_tab":[]}"""


async def _gpt(client: httpx.AsyncClient, system: str, user: str,
               model: str = "gpt-4o", max_tokens: int = 1500) -> str:
    r = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
        json={"model": model, "max_tokens": max_tokens,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": user}]}
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _parse_json(raw: str) -> dict:
    m = re.search(r'\{[\s\S]*\}', raw)
    return json.loads(m.group(0) if m else raw)


@app.post("/summarize")
async def summarize(req: Request):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key לא מוגדר")

    check_rate(get_ip(req))

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "JSON לא תקין")

    transcript = str(body.get("transcript", ""))[:MAX_TRANSCRIPT]
    screen_text = str(body.get("screen_text", ""))[:5000]
    segments    = body.get("segments", [])[:200]
    shots_meta  = body.get("shots_meta", [])[:50]

    if len(transcript) < 50:
        raise HTTPException(400, "התמלול קצר מדי לסיכום.")

    # Build rich context string
    ctx = ""
    if segments:
        ctx = "=== תמלול ===\n"
        for seg in segments:
            t = seg.get("start", 0)
            ctx += f"[{int(t//60)}:{int(t%60):02d}] {str(seg.get('text',''))[:500]}\n"
            near = [s for s in shots_meta if abs(s.get("timestamp", 0) - t) < 15]
            if near:
                ctx += f"  [📸 {near[0].get('reason','')}]\n"
    else:
        ctx = f"=== שיעור ===\n{transcript}"
    if screen_text:
        ctx += f"\n\n=== מסך ===\n{screen_text}"

    t_start = time.time()

    try:
        async with httpx.AsyncClient(timeout=60) as client:

            # ── STAGE 1: Knowledge enrichment ──────────────────────
            enrichment_json = ""
            enriched = False
            try:
                raw_enrich = await _gpt(client, ENRICH_SYSTEM,
                                        f"נתח את השיעור הבא:\n\n{ctx[:20000]}",
                                        model="gpt-4o", max_tokens=2000)
                enrich_data = _parse_json(raw_enrich)
                enrichment_json = json.dumps(enrich_data, ensure_ascii=False)
                topics = enrich_data.get("topics", [])
                print(f"[summarize] Stage 1 done. Topics: {topics}")
                enriched = True
            except Exception as e:
                print(f"[summarize] Stage 1 enrichment failed (continuing without it): {e}")

            # ── STAGE 2: Main summary generation ───────────────────
            if enriched:
                user_msg = (
                    f"תמלול השיעור:\n{ctx[:20000]}\n\n"
                    f"ניתוח הפרופסור המומחה:\n{enrichment_json[:8000]}"
                )
            else:
                user_msg = f"סכם את השיעור הבא:\n\n{ctx[:28000]}"

            raw_summary = await _gpt(client, SUMMARIZE_SYSTEM, user_msg,
                                     model="gpt-4o", max_tokens=4500)

        elapsed = round(time.time() - t_start, 1)

        # ── STAGE 3: Parse with retry on bad JSON ──────────────
        try:
            result = _parse_json(raw_summary)
        except Exception:
            print(f"[summarize] JSON parse failed on main response — retrying with fallback prompt")
            async with httpx.AsyncClient(timeout=60) as client2:
                raw_fallback = await _gpt(client2, SUMMARIZE_FALLBACK_SYSTEM,
                                          f"סכם:\n\n{ctx[:15000]}",
                                          model="gpt-4o", max_tokens=3000)
            try:
                result = _parse_json(raw_fallback)
            except Exception:
                raise HTTPException(500, "שגיאה בפענוח הסיכום. נסה שוב.")

        result["enriched"] = enriched
        print(f"[summarize] Done in {elapsed}s — enriched={enriched}, sections={len(result.get('sections', []))}")
        return result

    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(504, "הסיכום לקח יותר מדי זמן. נסה שוב.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise HTTPException(429, "יותר מדי בקשות. נסה שוב.")
        raise HTTPException(e.response.status_code, f"שגיאה: {e.response.text[:150]}")
    except Exception as e:
        raise HTTPException(502, f"שגיאת חיבור: {str(e)[:80]}")

# ─── Compress ───────────────────────────────────────────────────
@app.post("/compress")
async def compress(req: Request):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key לא מוגדר")

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "JSON לא תקין")

    chunks = [str(c)[:12000] for c in body.get("chunks", [])[:20] if isinstance(c, str) and c.strip()]
    if not chunks:
        raise HTTPException(400, "חסרים chunks")

    system = """אתה בוט דחיסת טקסט חכם. קרא את הטקסט הבא וצמצם אותו ל-30% מגודלו המקורי.
כללים:
- שמור על כל המידע החשוב: נוסחאות, הגדרות, דוגמאות, מספרים
- הסר: חזרות, מילות מעבר מיותרות, דיאלוג סתמי
- שמור על עברית תקינה וזרימה טבעית
- אל תסכם — רק דחוס ושמור את התוכן"""

    async def compress_one(chunk: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                    json={"model": "gpt-4o-mini", "max_tokens": 2000,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": chunk}]}
                )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
        return chunk  # fallback: return original

    results = await asyncio.gather(*[compress_one(c) for c in chunks])
    return {"compressed": list(results)}

# ─── OCR ────────────────────────────────────────────────────────
@app.post("/ocr")
async def ocr(req: Request):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key לא מוגדר")

    check_rate(get_ip(req))

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "JSON לא תקין")

    images = [img for img in body.get("images", [])[:6]
              if isinstance(img, str) and img.startswith("data:image/")]
    if not images:
        return {"text": ""}

    content = [{"type": "text", "text": "חלץ כל טקסט ונוסחאות (LaTeX) מהצילומים."}]
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": img, "detail": "high"}})

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                json={"model": "gpt-4o", "max_tokens": 2000,
                      "messages": [{"role": "user", "content": content}]}
            )
        if r.status_code == 200:
            return {"text": r.json()["choices"][0]["message"]["content"]}
    except Exception:
        pass

    return {"text": ""}

# ─── Chat (Ask the summary) ─────────────────────────────────────
@app.post("/chat")
async def chat(req: Request):
    if not OPENAI_KEY:
        raise HTTPException(500, "API Key לא מוגדר")

    check_rate(get_ip(req))

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "JSON לא תקין")

    messages = body.get("messages", [])
    summary = body.get("summary", {})
    transcript = str(body.get("transcript", ""))[:3000]

    if not messages or not isinstance(messages, list):
        raise HTTPException(400, "חסרות הודעות")

    clean_messages = []
    for m in messages[-20:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            clean_messages.append({"role": m["role"], "content": str(m["content"])[:2000]})

    if not clean_messages:
        raise HTTPException(400, "הודעות לא תקינות")

    system_prompt = f"""אתה עוזר חכם לסטודנטים. עונה רק על בסיס הסיכום והתמלול שניתן לך.
סיכום: {json.dumps(summary, ensure_ascii=False)[:4000]}
תמלול: {transcript}
כללים: ענה בעברית פשוטה. אם השאלה לא קשורה — אמור שאתה עונה רק על חומר השיעור. היה ידידותי. עד 150 מילה."""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"},
                json={"model": "gpt-4o", "max_tokens": 600,
                      "messages": [{"role": "system", "content": system_prompt}] + clean_messages}
            )
    except httpx.TimeoutException:
        raise HTTPException(504, "פסק זמן. נסה שוב.")
    except Exception as e:
        raise HTTPException(502, f"שגיאת חיבור: {str(e)[:80]}")

    if r.status_code == 429:
        raise HTTPException(429, "יותר מדי בקשות. נסה שוב.")
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"שגיאה: {r.text[:150]}")

    answer = r.json()["choices"][0]["message"]["content"]
    return {"answer": answer}
