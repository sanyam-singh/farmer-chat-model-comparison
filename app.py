import os
import time
import threading

import streamlit as st
import openai
from together import Together
from dotenv import load_dotenv


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
**FARMER QUERY:** {question}

**STRUCTURED FACTS:**
{facts_json_str}

**TASK:** Synthesize these facts into a natural, conversational response that directly answers the farmer's question while maintaining all technical accuracy and actionable details.

**IMPORTANT:**
1. Respond in the SAME LANGUAGE as the farmer's query above
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


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="Farmer.CHAT – Model Comparison", layout="wide")

st.title("Farmer.CHAT – Side-by-Side Model Comparison")
st.write("Compare **vanilla GPT‑4o‑mini** vs **Fine‑tuned + Fact‑Stitching** for farmer queries.")

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

        def run_vanilla_thread():
            try:
                vanilla_result["text"] = run_vanilla_model(
                    user_question, location, preferred_language, current_date
                )
            except Exception as e:
                vanilla_result["error"] = str(e)

        def run_ft_thread():
            try:
                resp, facts = run_ft_stitched_pipeline(user_question)
                ft_result["text"] = resp
                ft_result["facts"] = facts
            except Exception as e:
                ft_result["error"] = str(e)

        with st.spinner("Running both models..."):
            t1 = threading.Thread(target=run_vanilla_thread)
            t2 = threading.Thread(target=run_ft_thread)
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        with col1:
            st.subheader("Vanilla GPT‑4o‑mini")
            if vanilla_result["error"]:
                st.error(f"Error: {vanilla_result['error']}")
            else:
                st.markdown(vanilla_result["text"])

        with col2:
            st.subheader("Fine‑tuned + Fact‑Stitching")
            if ft_result["error"]:
                st.error(f"Error: {ft_result['error']}")
            else:
                st.markdown(ft_result["text"])
                if ft_result["facts"]:
                    with st.expander("Show extracted facts JSON"):
                        st.code(ft_result["facts"], language="json")

