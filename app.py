import os
import time
import threading
import json

import streamlit as st
import openai
from together import Together
from dotenv import load_dotenv
import pandas as pd

from evals import run_enhanced_evaluation_pipeline_optimized


# ============================================================================
# LOAD ENV VARS
# ============================================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")


# ============================================================================
# MODEL CONFIG
# ============================================================================

VANILLA_MODEL = "gpt-4o-mini-2024-07-18"  # base model
FINE_TUNED_MODEL = "ft:gpt-4o-mini-2024-07-18:dgf-prod-dev-account:3:CafMKPv8"
SYNTHESIS_MODEL = "google/gemma-3n-E4B-it"

RETRY_ATTEMPTS = 3
RETRY_DELAY = 2


# ============================================================================
# SIMPLE LANGUAGE DETECTION
# ============================================================================

def detect_language_name(text: str) -> str:
    """Very simple language heuristic: Hindi (Devanagari) vs English/other."""
    for ch in text:
        if "\u0900" <= ch <= "\u097F":
            return "Hindi"
    return "English"


# ============================================================================
# PROMPTS
# ============================================================================

# Vanilla model prompt template from your JSON definition
VANILLA_PROMPT_TEMPLATE = """You are Farmer.CHAT, a knowledgeable agricultural advisor helping farmers in {location}.

**USER CONTEXT:**
Location: {location}
Preferred Language: {preferred_language}
Current Date: {current_date}
Query: {rephrased_query}
Crop/Livestock: {asset_name}
Variety: {asset_variety}
Current Stage: {asset_stage}
Main Concern: {concern}
Recommended Activity: {likely_activity}
Chat History: {chat_history}

**YOUR ROLE:**
Experienced agricultural extension worker with deep local knowledge
Supportive mentor who understands smallholder farming challenges
Culturally appropriate communication style for the region

**RESPONSE GUIDELINES:**

1. **Content Quality:**
  - Address the specific concern directly and practically
  - Provide actionable, region-appropriate advice
  - Include timing considerations based on current stage and season
  - Use local examples, varieties, and practices when relevant

2. **Communication Style:**
  - Warm, professional, and encouraging tone
  - Use {preferred_language} with appropriate cultural context
  - Explain technical concepts in simple terms
  - Avoid overly formal or academic language

3. **Practical Advice:**
  - Focus on low-cost, accessible solutions
  - Consider resource constraints of smallholder farmers
  - Mention local availability of inputs/resources
  - Include preventive measures when relevant

4. **Safety & Credibility:**
  - For chemical inputs: mention general categories rather than specific brands
  - Include safety precautions for handling chemicals/equipment
  - Encourage consultation with local experts for complex issues
  - Reference local agricultural departments or extension services

5. **Conversation Flow:**
  - Build on previous advice from chat history when relevant
  - Don't repeat information already covered
  - Ask clarifying questions if critical information is missing
  - Offer to elaborate on specific aspects if helpful

**RESPONSE FORMAT:**
Provide your complete response as a natural conversation in {preferred_language}. Structure your advice logically but don't force artificial formatting.

**AVOID:**
Generic advice that could apply to any crop/region
Overly technical jargon without explanation
Repetitive closing statements or tips
Specific product recommendations without local context
Assumptions about farmer's resources or experience level
"""


NEW_SPECIFICITY_PROMPT = """
You are an agricultural fact classifier. Your task is to classify answers as either "Specific" or "Not Specific" based on their contextual anchors and actionability.

Use this framework internally:
- Specific: Contains sufficient contextual anchors AND actionable insight
- Not Specific: Lacks contextual anchors OR actionable insight; generic or vague

Checklist (for your own reasoning, do NOT list these back):
1. Entity specificity (crop/variety/soil/weather/organization named)
2. Location specificity (named place or bounded geography)
3. Time specificity (explicit time window or marker, including seasons like Rabi/Kharif/Zaid, 30 DAS, pre-sowing)
4. Quantity/measurement (numeric or measurable details)
5. Conditionality/comparison (if-then, comparative baselines)
6. Mechanistic/causal link (clear cause-effect enabling decision-making)
7. Actionability (directly informs a decision or step relevant to context)

Decision rule:
- Specific = at least 2 of flags 1–6 are TRUE AND flag 7 is TRUE
- Not Specific = otherwise

OUTPUT FORMAT (STRICT):
Return a SINGLE LINE in this exact pipe-delimited format, no extra text:
LABEL|flag1,flag2,...|brief justification

Where:
- LABEL is exactly "Specific" or "Not Specific"
- flags is a comma-separated list of lowercase snake_case flag names you consider true (e.g. entity_specificity,time_specificity,actionability). Use empty string if none.
- justification is a short explanation referencing anchors and actionability (no newlines; use ';' if needed).
"""


COMPARISON_PROMPT = """
You are an expert agricultural assistant and model evaluator. You will compare TWO answers given to the SAME farmer question:
- "gpt4o" = vanilla GPT‑4o‑mini answer
- "ft" = Fine-Tuned + Fact-Stitching answer

Your main goal is to highlight where the Fine-Tuned pipeline is **better** along practically useful dimensions, especially:
- Actionability (clear steps for farmers)
- Quantity / dosage precision
- Timing (when to do what)
- Local context (Bihar, smallholder-friendly)
- Safety (correct, non-harmful, cautious)

Your job:
1. Mentally break the content into aspects (1–3 sentences or a coherent bullet).
2. For each aspect, decide which answer serves the farmer better:
   - "gpt4o" (GPT‑4o‑mini better)
   - "ft" (Fine-Tuned better)  ← if both are good but ft is slightly better, pick "ft"
   - "both" (equally good and similar)
   - "neither" (both are poor or irrelevant)
3. Give a short reason for each aspect-level judgment, focusing on:
   - factual correctness and recall
   - specificity and actionability
   - clarity and safety for farmers in Bihar

Then also provide an overall comparison summary with a clear "Better response" choice.

STRICT OUTPUT FORMAT (NO JSON, NO EXTRA TEXT):
Return multiple lines in this format:

SEG|aspect_name|winner|short_reason
...
OVERALL|better_model|short_reason

Where:
- aspect_name is a short name like "actionability", "dosage precision", "timing", "local context", "safety", etc. (no pipes, no newlines)
- winner is exactly one of: gpt4o, ft, both, neither
- short_reason is a one-line explanation (no newlines, avoid '|' character; use ';' if needed)
- better_model is exactly one of: gpt4o, ft, both, neither
"""


FACT_EXTRACTION_PROMPT = """
You are an agricultural fact generator specialized in farming practices. Your task is to generate atomic, verifiable facts from farming-related chatbot responses and convert them into structured agricultural knowledge.

**GENERATION SCOPE:**
- Generate ONLY facts related to agriculture, farming, crops, livestock, or agricultural practices
- Ignore user greetings, conversational elements, follow-up questions, and response metadata
- Focus on actionable agricultural information that farmers can apply
- Generate quantifiable data, specific techniques, timing recommendations, and measurable outcomes

**BIHAR AGRICULTURAL CONTEXT:**
- Common Bihar districts: Patna, Darbhanga, Madhubani, Champaran, Gopalganj, Gaya, Aurangabad, Muzaffarpur, Begusarai, Bhagalpur
- Primary crops: rice, wheat, maize, sugarcane, potato, onion, arhar (pigeon pea), masur (lentil), gram (chickpea), jute, tobacco
- Key challenges: flooding, drought, pest management, soil salinity, waterlogging
- Agricultural seasons: Kharif (June-October), Rabi (November-April), Zaid (April-June)


**FACT ATOMICITY REQUIREMENTS:**
Each fact must contain exactly ONE verifiable claim. Break down complex statements:

❌ Complex: "Apply neem oil at 3ml per liter in early morning every 7 days for aphid control during flowering stage"
✅ Atomic facts:
- "Apply neem oil at 3ml per liter concentration for aphid control"
- "Apply neem oil in early morning for optimal effectiveness"
- "Repeat neem oil application every 7 days for persistent aphid management"
- "Apply neem oil during flowering stage for aphid control"

**OUTPUT FORMAT:**
Return a JSON object with a "facts" array where each fact includes:

{
  "facts": [
    {
      "fact": "The atomic factual statement (preserve original phrasing when possible)",
      "category": "One of: [crop_variety, pest_disease, soil_management, irrigation, seasonal_practice, input_management]",
      "location_dependency": "bihar_specific | universal | region_adaptable",
      "bihar_relevance": "high | medium | low",
      "confidence": 0.0-1.0
    }
  ]
}

**CONFIDENCE SCORING GUIDELINES:**
- 0.9-1.0: Well-established scientific facts, standardized practices
- 0.7-0.8: Commonly accepted practices with good evidence
- 0.5-0.6: Traditional practices with mixed evidence
- 0.3-0.4: Emerging practices or limited evidence
- 0.1-0.2: Anecdotal or highly uncertain information

**STRICT EXCLUSION CRITERIA:**
- Greetings and pleasantries: "Hello [Name]", "Hope this helps!", "Thank you for asking"
- Follow-up suggestions: "Would you like to know about...", "Here are related questions", "Feel free to ask"
- Meta-responses: "Based on the context provided", "Sorry, this seems out of context", "I don't have information about"
- Opinion statements: "I think", "It's best to", "You should consider", "In my opinion"
- Conversational fillers: "Well", "Actually", "By the way", "Also note that"
- Disclaimers: "Please consult an expert", "Results may vary", "This is general advice"
- Question repetitions or acknowledgments of user queries

**QUALITY CHECKS:**
- Each fact should be independently verifiable
- Preserve specific measurements, quantities, and technical terms
- Maintain agricultural terminology accuracy
- Ensure facts are actionable for farmers
- Verify that each fact addresses a single agricultural concept

"""


SYNTHESIS_PROMPT = """
You are Farmer.CHAT, a knowledgeable agricultural advisor helping farmers in Bihar.


**YOUR ROLE:**
- Experienced agricultural extension worker with deep local knowledge
- Supportive mentor who understands smallholder farming challenges
- Culturally appropriate communication style for the region


**RESPONSE GUIDELINES:**


1. **Content Quality:**
 - Address the specific concern directly and practically
 - Provide actionable, region-appropriate advice
 - Include timing considerations based on current stage and season
 - Use local examples, varieties, and practices when relevant


2. **Communication Style:**
 - Warm, professional, and encouraging tone
 - Use simple, conversational language with appropriate cultural context
 - Explain technical concepts in simple terms
 - Avoid overly formal or academic language
 - **CRITICAL: Respond in the SAME LANGUAGE as the user's query**


3. **Practical Advice:**
 - Focus on low-cost, accessible solutions
 - Consider resource constraints of smallholder farmers
 - Mention local availability of inputs/resources
 - Include preventive measures when relevant


4. **Safety & Credibility:**
 - For chemical inputs: mention general categories rather than specific brands
 - Include safety precautions for handling chemicals/equipment
 - Encourage consultation with local experts for complex issues
 - Reference local agricultural departments or extension services


5. **Conversation Flow:**
 - Build on previous advice from chat history when relevant
 - Don't repeat information already covered
 - Ask clarifying questions if critical information is missing
 - Offer to elaborate on specific aspects if helpful


**FORMATTING REQUIREMENTS:**
- Use bullet points (•) for lists of steps, recommendations, or multiple items
- Use numbered lists (1., 2., 3.) for sequential steps or procedures
- Keep paragraphs short (2-3 sentences max)
- Use line breaks between different topics or sections
- Bold key terms or important warnings when needed
- Structure complex advice with clear headings or separators


**RESPONSE FORMAT:**
Provide your complete response as a natural conversation with proper formatting. Structure your advice logically with bullet points and numbered lists where appropriate. Keep responses between 150-300 words.


**AVOID:**
- Generic advice that could apply to any crop/region
- Overly technical jargon without explanation
- Repetitive closing statements or tips
- Specific product recommendations without local context
- Assumptions about farmer's resources or experience level
- Long paragraphs without breaks or bullet points
"""


# ============================================================================
# CLIENT INIT
# ============================================================================

openai.api_key = OPENAI_API_KEY
together_client = Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None


def build_vanilla_system_prompt(
    location: str,
    preferred_language: str,
    current_date: str,
    rephrased_query: str,
    asset_name: str = "",
    asset_variety: str = "",
    asset_stage: str = "",
    concern: str = "",
    likely_activity: str = "",
    chat_history: str = "",
) -> str:
    return VANILLA_PROMPT_TEMPLATE.format(
        location=location,
        preferred_language=preferred_language,
        current_date=current_date,
        rephrased_query=rephrased_query,
        asset_name=asset_name,
        asset_variety=asset_variety,
        asset_stage=asset_stage,
        concern=concern,
        likely_activity=likely_activity,
        chat_history=chat_history,
    )


# ============================================================================
# PIPELINE HELPERS
# ============================================================================

def run_vanilla_model(question: str, location: str, preferred_language: str, current_date: str):
    system_prompt = build_vanilla_system_prompt(
        location=location,
        preferred_language=preferred_language,
        current_date=current_date,
        rephrased_query=question,
        asset_name="धान" if "धान" in question else "",
        concern="सामान्य सलाह" if preferred_language.lower().startswith("hi") else "General advice",
        likely_activity="सलाह" if preferred_language.lower().startswith("hi") else "advisory",
        chat_history="",
    )

    resp = openai.ChatCompletion.create(
        model=VANILLA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=800,
    )
    return resp["choices"][0]["message"]["content"].strip()


def run_ft_stitched_pipeline(question: str, retry_count: int = 0):
    try:
        user_lang = detect_language_name(question)

        fact_response = openai.ChatCompletion.create(
            model=FINE_TUNED_MODEL,
            messages=[
                {"role": "system", "content": FACT_EXTRACTION_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        facts_json_str = fact_response["choices"][0]["message"]["content"]

        synthesis_user_prompt = f"""
USER QUERY LANGUAGE: {user_lang}
Preferred Language: {user_lang}

**FARMER QUERY:** {question}

**STRUCTURED FACTS:**
{facts_json_str}

**TASK:** Synthesize these facts into a natural, conversational response that directly answers the farmer's question while maintaining all technical accuracy and actionable details.

**IMPORTANT:**
1. Respond STRICTLY in {user_lang}. Do NOT mix in other languages except technical names.
2. Use bullet points (•) for lists and recommendations
3. Use numbered lists (1., 2., 3.) for sequential steps
4. Keep formatting clear and easy to read
"""

        synthesis_response = together_client.chat.completions.create(
            model=SYNTHESIS_MODEL,
            messages=[
                {"role": "system", "content": SYNTHESIS_PROMPT},
                {"role": "user", "content": synthesis_user_prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        final_response = synthesis_response.choices[0].message.content
        return final_response, facts_json_str

    except Exception as e:
        msg = str(e)
        if (
            retry_count < RETRY_ATTEMPTS
            and ("rate_limit" in msg.lower() or "timeout" in msg.lower())
        ):
            time.sleep(RETRY_DELAY * (retry_count + 1))
            return run_ft_stitched_pipeline(question, retry_count + 1)
        raise


def classify_specificity(text: str):
    """Classify a response as Specific / Not Specific using NEW_SPECIFICITY_PROMPT.

    Returns a small dict with label, flags, justification.
    """
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": NEW_SPECIFICITY_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        content = resp["choices"][0]["message"]["content"].strip()
        # Remove code fences if present
        if content.startswith("```"):
            content = content.strip("`").strip()

        # Expected format: LABEL|flag1,flag2|justification
        parts = content.split("|", 2)
        if len(parts) != 3:
            raise ValueError(f"Unexpected specificity format: {content}")

        label = parts[0].strip()
        flags_str = parts[1].strip()
        justification = parts[2].strip()

        flags = [f.strip() for f in flags_str.split(",") if f.strip()] if flags_str else []
        return {
            "label": label,
            "flags": flags,
            "justification": justification,
        }
    except Exception as e:
        return {
            "label": "Error",
            "flags": [],
            "justification": f"Specificity evaluation failed: {e}",
        }


def compare_answers(question: str, gpt4o_text: str, ft_text: str):
    """Use GPT‑4o as a judge to compare answers segment-wise.

    Returns a dict with:
      - 'segments': list of {aspect, winner, reason}
      - 'overall': {better_model, reason}
      - or {'error': ...} on failure.
    """
    try:
        judge_prompt = f"""
FARMER QUESTION:
{question}

GPT‑4o‑mini ANSWER:
{gpt4o_text}

FINE-TUNED ANSWER:
{ft_text}

Follow the instructions and JSON schema exactly.
"""
        resp = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": COMPARISON_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
            max_tokens=800,
        )
        content = resp["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`").strip()

        segments = []
        overall = {"better_model": "neither", "reason": ""}

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("SEG|"):
                parts = line.split("|", 3)
                if len(parts) < 4:
                    continue
                _, aspect, winner, reason = parts
                segments.append(
                    {
                        "aspect": aspect.strip(),
                        "winner": winner.strip(),
                        "reason": reason.strip(),
                    }
                )
            elif line.startswith("OVERALL|"):
                parts = line.split("|", 2)
                if len(parts) < 3:
                    continue
                _, better_model, reason = parts
                overall = {
                    "better_model": better_model.strip(),
                    "reason": reason.strip(),
                }

        if not segments and not overall.get("reason"):
            raise ValueError(f"Unexpected judge format: {content}")

        return {"segments": segments, "overall": overall}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Farmer.CHAT – Model Comparison", layout="wide")

st.title("Farmer.CHAT – Side-by-Side Model Comparison")
st.write("Compare **GPT‑4o‑mini** vs **Fine-Tuned** pipelines for farmer queries.")

with st.sidebar:
    st.header("Settings")
    st.write("API keys are read from `.env`. Override here if needed.")

    openai_key_override = st.text_input(
        "OpenAI API Key",
        value=OPENAI_API_KEY,
        type="password",
    )
    together_key_override = st.text_input(
        "Together API Key",
        value=TOGETHER_API_KEY,
        type="password",
    )

    if openai_key_override:
        openai.api_key = openai_key_override
    if together_key_override:
        # Override Together client if a key is provided in the sidebar
        together_client = Together(api_key=together_key_override)

    st.markdown("---")
    location = st.text_input("Farmer location", value="Bihar")
    preferred_language = st.text_input("Preferred language", value="Hindi")
    current_date = st.text_input("Current date", value="2026-02-09")

    # Performance metrics from last run (if available)
    if "last_perf" in st.session_state:
        st.markdown("---")
        st.subheader("Performance (last run)")
        lp = st.session_state["last_perf"]
        if lp.get("vanilla_time") is not None:
            st.metric("GPT‑4o‑mini latency (s)", f"{lp['vanilla_time']:.2f}")
        if lp.get("ft_time") is not None:
            st.metric("Fine-Tuned latency (s)", f"{lp['ft_time']:.2f}")
        if lp.get("vanilla_spec_label"):
            st.metric("GPT‑4o‑mini specificity", lp["vanilla_spec_label"])
        if lp.get("ft_spec_label"):
            st.metric("Fine-Tuned specificity", lp["ft_spec_label"])

default_question = "धान की फसल में कीट नियंत्रण कैसे करें?"
user_question = st.text_area(
    "Farmer query (same sent to both models)",
    value=default_question,
    height=140,
)

col1, col2 = st.columns(2)

if st.button("Run comparison"):
    if not openai.api_key:
        st.error("OpenAI API key is missing. Set it in `.env` or the sidebar.")
    elif not together_client:
        st.error("Together API key is missing. Set it in `.env` or the sidebar.")
    elif not user_question.strip():
        st.error("Please enter a farmer query.")
    else:
        vanilla_result = {"text": None, "error": None}
        ft_result = {"text": None, "facts": None, "error": None}
        perf = {"vanilla_time": None, "ft_time": None}

        def run_vanilla_thread():
            try:
                t0 = time.time()
                text = run_vanilla_model(
                    user_question, location, preferred_language, current_date
                )
                perf["vanilla_time"] = time.time() - t0
                vanilla_result["text"] = text
            except Exception as e:
                vanilla_result["error"] = str(e)

        def run_ft_thread():
            try:
                t0 = time.time()
                resp, facts = run_ft_stitched_pipeline(user_question)
                perf["ft_time"] = time.time() - t0
                ft_result["text"] = resp
                ft_result["facts"] = facts
            except Exception as e:
                ft_result["error"] = str(e)

        with st.spinner("Running both models (generation only)..."):
            t1 = threading.Thread(target=run_vanilla_thread)
            t2 = threading.Thread(target=run_ft_thread)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        # Store performance metrics for sidebar display
        if perf["vanilla_time"] is not None or perf["ft_time"] is not None:
            st.session_state["last_perf"] = perf

        # Show raw answers immediately
        with col1:
            st.subheader("GPT‑4o‑mini")
            if vanilla_result["error"]:
                st.error(f"Error: {vanilla_result['error']}")
            else:
                st.markdown(vanilla_result["text"])

        with col2:
            st.subheader("Fine-Tuned")
            if ft_result["error"]:
                st.error(f"Error: {ft_result['error']}")
            else:
                st.markdown(ft_result["text"])
                if ft_result["facts"]:
                    with st.expander("Show extracted facts JSON"):
                        st.code(ft_result["facts"], language="json")

        # Run comparison + specificity after showing answers
        if not vanilla_result["error"] and not ft_result["error"]:
            with st.spinner("Running comparison and specificity evals..."):
                judge_result = compare_answers(user_question, vanilla_result["text"], ft_result["text"])
                gpt4o_spec = classify_specificity(vanilla_result["text"])
                ft_spec = classify_specificity(ft_result["text"])

            st.markdown("---")
            st.subheader("Model Comparison (Judge View)")

            if "error" in judge_result:
                st.error(f"Judge error: {judge_result['error']}")
            else:
                segs = judge_result.get("segments", [])
                overall = judge_result.get("overall", {})

                # Simple colored list of segment judgments
                for seg in segs:
                    winner = seg.get("winner", "neither")
                    color = "#eeeeee"
                    if winner == "ft":
                        color = "#c8e6c9"  # green for Fine-Tuned (preferred)
                    elif winner == "gpt4o":
                        color = "#ffcdd2"  # red for vanilla

                    st.markdown(
                        f"<div style='background-color:{color};padding:8px;border-radius:4px;margin-bottom:4px;'>"
                        f"<b>Aspect:</b> {seg.get('aspect','')}<br>"
                        f"<b>Winner:</b> {winner}<br>"
                        f"<b>Reason:</b> {seg.get('reason','')}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Tabular-style explanation like your example
                st.markdown("**Overall judgment:**")
                better = overall.get("better_model", "neither")
                reason = overall.get("reason", "")
                better_label = "Fine-Tuned + Fact-Stitching" if better == "ft" else (
                    "GPT‑4o‑mini" if better == "gpt4o" else "Both / Neither"
                )
                st.markdown(f"👉 **Better response:** {better_label}")
                if reason:
                    st.markdown(f"**Why?** {reason}")

            st.markdown("---")
            st.subheader("Specificity (per answer)")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("**GPT‑4o‑mini**")
                st.write(f"Label: {gpt4o_spec['label']}")
                st.write(f"Flags: {', '.join(gpt4o_spec['flags']) or 'None'}")
                st.write(f"Justification: {gpt4o_spec['justification']}")

            with col_s2:
                st.markdown("**Fine-Tuned**")
                st.write(f"Label: {ft_spec['label']}")
                st.write(f"Flags: {', '.join(ft_spec['flags']) or 'None'}")
                st.write(f"Justification: {ft_spec['justification']}")

