def _is_continuation_question(current_question: str, chat_history: list) -> bool:
    """Lightweight check for likely follow-ups.

    We intentionally avoid hard-coded service lists and KB IDs so retrieval is not
    biased toward stale topics. This only flags obvious short follow-ups; callers
    should prefer an LLM-based rewriter if available.
    """
    if not chat_history:
        return False

    q_lower = current_question.lower()
    words = current_question.split()
    last_reference = next(
        (msg.get("reference") for msg in reversed(chat_history[:-1]) if msg.get("role") == "assistant" and msg.get("reference")),
        ""
    )

    continuation_markers = [
        'if that fails', 'if it fails', 'if not',
        'and if', 'what if', 'what then',
        'still', 'still not',
        'what else', 'anything else',
        'next step', 'then what', 'after that'
    ]

    topic_intro_markers = [
        'how to ', 'how do i', 'how can i', 'tell me about', 'issue with', 'problem with', 'trouble with'
    ]

    if any(marker in q_lower for marker in continuation_markers):
        return True

    # If we already have a KB reference in play and the new question is short and
    # not a clear topic-intro, treat it as a follow-up to keep retrieval anchored.
    if last_reference and len(words) <= 10 and not any(marker in q_lower for marker in topic_intro_markers):
        return True

    # Very short questions are often follow-ups, but we keep this conservative
    return len(words) < 5


def enhance_question_with_context(current_question: str, chat_history: list) -> str:
    """
    🎯 CHAT CONTINUITY: Enhance current question with context from chat history.
    
    SMART ENHANCEMENT: Only applies to follow-up/continuation questions, NOT to new topics.
    Topic detection is DATA-DRIVEN (not hardcoded keyword lists), so it works with future KBs.
    
    🎯 SMARTER CONTEXT EXTRACTION: Tracks KB topic from LAST ASSISTANT RESPONSE, not just
    previous user questions (which often lack topic keywords in follow-ups).
    
    Continuation example:
    - Chat: ["User cannot login to Citrix", assistant response from Citrix KB]
    - Question: "And if that fails?"
    - Enhanced: "citrix KB0015259 And if that fails?"
    
    New topic example (NO enhancement):
    - Chat: ["How to reset password?"]
    - Question: "How to fix Outlook issues"
    - Returns: "How to fix Outlook issues" (unchanged - new topic)
    
    🔄 FUTURE-PROOF: Uses linguistic patterns, not hardcoded topics.
    New KBs don't require code changes.
    
    Args:
        current_question: The user's current question
        chat_history: List of chat dicts with 'role' and 'content' (and optional 'reference')
    
    Returns:
        Enhanced question string if continuation, otherwise original question
    """
    # Only enhance when clearly a follow-up; otherwise keep the raw question to avoid
    # biasing retrieval toward unrelated KBs.
    if not _is_continuation_question(current_question, chat_history):
        return current_question

    # If the last assistant turn carried a KB reference, prepend it to keep retrieval
    # anchored to the same topic for follow-ups (e.g., avoid drifting to a different KB
    # when the user asks "where to escalate this").
    last_reference = next(
        (msg.get("reference") for msg in reversed(chat_history[:-1]) if msg.get("role") == "assistant" and msg.get("reference")),
        ""
    )
    if last_reference:
        return f"{last_reference} {current_question}".strip()

    # Otherwise, lightly stitch the last user utterance for minimal context.
    prior_user = next((msg.get("content", "") for msg in reversed(chat_history[:-1]) if msg.get("role") == "user"), "")
    if prior_user:
        return f"{prior_user} {current_question}".strip()

    return current_question


def _extract_question_intent(question: str) -> dict:
    """
    🎯 INTENT EXTRACTION: Determine EXACTLY what the user is asking for.
    
    This enables FOCUSED answers instead of dumping entire KB articles.
    
    Returns:
        {
            "scope": "specific" | "comprehensive" | "quickstart",
            "focus": "first_step" | "next_steps" | "all_steps" | "overview" | "troubleshooting",
            "depth": "brief" | "moderate" | "detailed"
        }
    
    Examples:
    - "What should L1 check first?" → scope:specific, focus:first_step, depth:brief
    - "And if that fails?" → scope:specific, focus:next_steps, depth:moderate
    - "Complete guide on X" → scope:comprehensive, focus:all_steps, depth:detailed
    - "How to do X?" → scope:specific, focus:all_steps, depth:moderate
    """
    q_lower = question.lower()
    
    intent = {
        "scope": "specific",        # Default to specific (not dumping everything)
        "focus": "all_steps",       # Default to all relevant steps for the question
        "depth": "moderate"         # Default to moderate detail
    }
    
    # 🎯 SCOPE: Is user asking for comprehensive overview or specific answer?
    if any(word in q_lower for word in ['complete', 'all', 'everything', 'entire', 'full guide', 'full checklist', 'step by step guide']):
        intent["scope"] = "comprehensive"
    elif any(word in q_lower for word in ['first', 'initial', 'start', 'beginning', 'before']):
        intent["scope"] = "specific"
    else:
        intent["scope"] = "specific"
    
    # 🎯 FOCUS: What specific aspect are they asking about?
    if any(word in q_lower for word in ['first', 'initially', 'start', 'before', 'begin']):
        intent["focus"] = "first_step"
    elif any(word in q_lower for word in ['then', 'next', 'after that', 'if that fails', 'if it fails', 'if not', 'what then', 'and then']):
        intent["focus"] = "next_steps"
    elif any(word in q_lower for word in ['what is', 'what are', 'tell me', 'explain', 'overview', 'summary']):
        intent["focus"] = "overview"
    elif any(word in q_lower for word in ['issue', 'problem', 'fix', 'troubleshoot', 'error', 'fail', 'not working']):
        intent["focus"] = "troubleshooting"
    else:
        intent["focus"] = "all_steps"
    
    # 🎯 DEPTH: How detailed should the answer be?
    if any(word in q_lower for word in ['quick', 'brief', 'short', 'summary', 'tldr', 'just tell me']):
        intent["depth"] = "brief"
    elif any(word in q_lower for word in ['detailed', 'detailed steps', 'step by step', 'complete', 'full', 'everything']):
        intent["depth"] = "detailed"
    else:
        intent["depth"] = "moderate"
    
    return intent


def _build_focused_instructions(intent: dict) -> str:
    """
    Build LLM instructions based on user's question intent.
    
    This replaces the one-size-fits-all "COMPREHENSIVE" instruction.
    """
    instructions = ""
    
    if intent["scope"] == "specific":
        instructions += "- Answer ONLY what is asked. Do NOT include unnecessary background or full procedures if not requested.\n"
    
    if intent["focus"] == "first_step":
        instructions += "- Focus ONLY on the FIRST step or initial check. Do not list all steps unless specifically asked.\n"
        instructions += "- Start with: 'First step: ...' or 'Initial check:...'\n"
    elif intent["focus"] == "next_steps":
        instructions += "- Provide ONLY the next steps/actions after the current context. Avoid repeating earlier troubleshooting.\n"
        instructions += "- Be concise and action-oriented.\n"
    elif intent["focus"] == "overview":
        instructions += "- Provide a brief overview/summary of the main points.\n"
        instructions += "- Use bullet points for clarity.\n"
    elif intent["focus"] == "troubleshooting":
        instructions += "- Focus on FIXING the problem, not explaining background.\n"
        instructions += "- List specific troubleshooting steps in order.\n"
    
    if intent["depth"] == "brief":
        instructions += "- Keep answer concise. Max 2-3 key points.\n"
        instructions += "- Avoid lengthy explanations.\n"
    elif intent["depth"] == "detailed":
        instructions += "- Provide comprehensive details and explanations.\n"
        instructions += "- Include context and reasons behind steps.\n"
    
    return instructions


def build_prompt(context, question, search_results=None):
    """
    Build a prompt for the LLM with context and question.
    
    🎯 FOCUSED ANSWERS: Instructions are tailored to the specific question intent,
    not just "answer comprehensively about everything."
    
    Args:
        context: Text context from search results
        question: User question
        search_results: Optional list of search result dicts with metadata
    
    Returns:
        Formatted prompt string
    """
    # Include KB metadata at the beginning of context if available
    kb_context = ""
    if search_results:
        kb_info = []
        for i, result in enumerate(search_results, 1):
            metadata = result.get("metadata", {})
            kb_number = metadata.get("kb_number", "")
            kb_title = metadata.get("kb_title", metadata.get("document_title", ""))
            if kb_number and kb_title:
                kb_info.append(f"- {kb_number}: {kb_title}")
            elif kb_number:
                kb_info.append(f"- {kb_number}")
            elif kb_title:
                kb_info.append(f"- {kb_title}")
        
        if kb_info:
            kb_context = "\n[Knowledge Base Sources]\n" + "\n".join(kb_info) + "\n\n"
    
    # 🎯 INTENT-BASED INSTRUCTIONS: Extract what the user really wants
    intent = _extract_question_intent(question)
    focused_instructions = _build_focused_instructions(intent)
    
    # 🎯 Dynamic header based on scope
    if intent["scope"] == "comprehensive":
        answer_guidance = "Provide a comprehensive and detailed answer covering all relevant aspects."
    else:
        answer_guidance = "Answer ONLY the specific question asked. Avoid unnecessary background or full procedures unless requested."
    
    return f"""
You are an Intelligent Agent Assist for GSD agents.

Answer ONLY using the provided knowledge base context.

ANSWER GUIDANCE:
{answer_guidance}
{focused_instructions}

Use bullet points or numbered lists when appropriate.

**CRITICAL: Always extract and clearly highlight escalation information:**
- If the context mentions which team to escalate to, ALWAYS include it
- If there are specific escalation conditions or triggers, ALWAYS mention them
- If escalation contacts or procedures exist, ALWAYS provide them
- Format escalation info prominently (use "Escalation:" or "Route to:" labels)
- Never omit escalation details even if they seem secondary

After the answer, ALWAYS add a "Reference" section.
List the KB ID and title from the sources at the top of this prompt.
The references must be visually subtle and secondary to the answer.
Format strictly as:

<your focused answer>

_Reference:_
_<KB_ID> – <KB_TITLE>_

If multiple KBs are used, list all uniquely.
Do NOT invent KB numbers.

CRITICAL RULES:
1. You MUST answer ONLY using the provided context.
2. You MUST NOT use outside knowledge.
3. You MUST answer EXACTLY what is asked - no more, no less.
4. **You MUST include escalation/team information if present in context.**
5. You MUST include a Reference section.

{kb_context}Context:
--------------------
{context}
--------------------

Question:
{question}

"""
