"""
Research Assistant Prompt Templates

This module contains specialized prompt templates for creating research assistant agents
that can help with academic research, fact-checking, and information synthesis.
"""

from agentframe import PromptTemplate, AgentPrompts, AgentConfig

# Research Assistant System Prompt
research_system_prompt = PromptTemplate(
    name="research_assistant_system",
    template="""You are an expert research assistant with advanced skills in information gathering, analysis, and synthesis.

Your core competencies:
- {personality_traits}

Research methodology:
- {communication_style}

Quality standards:
- Always cite sources when providing information
- Distinguish between facts, opinions, and speculation
- Acknowledge limitations and uncertainties
- Provide multiple perspectives on controversial topics
- Use {response_style} communication

Specialized knowledge areas:
- Academic research methods
- Source evaluation and credibility assessment
- Information synthesis and summarization
- Fact-checking and verification

Custom research guidelines:
{custom_guidelines}

Remember: Accuracy and intellectual honesty are paramount in all research activities.""",
    required_variables=["personality_traits", "communication_style", "response_style"],
    optional_variables={"custom_guidelines": "Prioritize peer-reviewed sources and primary research."},
    description="System prompt for research assistant agents",
    examples=[
        "Conduct literature reviews on specific topics",
        "Fact-check claims and statements",
        "Synthesize information from multiple sources",
        "Evaluate source credibility and bias"
    ]
)

# Research Planning Prompt
research_planning_prompt = PromptTemplate(
    name="research_planning",
    template="""Plan a comprehensive research approach for: {user_input}

Available research tools: {available_tools}

Research planning considerations:
1. What type of research is needed? (exploratory, descriptive, explanatory, evaluative)
2. What are the key research questions to address?
3. What sources should be consulted? (primary, secondary, tertiary)
4. What search strategies will be most effective?
5. How should the information be organized and presented?
6. What potential biases or limitations should be considered?

Create a structured research plan that ensures comprehensive and accurate results.

Research methodology steps:
- Define scope and objectives
- Identify key concepts and terms
- Plan search strategy
- Evaluate and select sources
- Synthesize findings
- Present conclusions with appropriate caveats""",
    required_variables=["user_input", "available_tools"],
    description="Planning prompt for research tasks and information gathering"
)

# Source Evaluation Prompt
source_evaluation_prompt = PromptTemplate(
    name="source_evaluation",
    template="""Evaluate the credibility and reliability of sources for research on: {topic}

Source evaluation criteria:
1. Authority - Who is the author? What are their credentials?
2. Accuracy - Is the information factual and well-supported?
3. Objectivity - Is there bias or agenda? Multiple perspectives?
4. Currency - How recent is the information? Is it still relevant?
5. Coverage - Is the topic covered comprehensively?

For each source, assess:
- Publication type (peer-reviewed, news, blog, government, etc.)
- Author expertise and institutional affiliation
- Evidence quality (citations, data, methodology)
- Potential conflicts of interest
- Corroboration from other sources

Provide a credibility rating and justification for each source.""",
    required_variables=["topic"],
    description="Template for evaluating source credibility and reliability"
)

# Literature Review Prompt
literature_review_prompt = PromptTemplate(
    name="literature_review",
    template="""Conduct a literature review on: {research_topic}

Literature review structure:
1. Introduction and scope definition
2. Search methodology and criteria
3. Key themes and findings
4. Analysis of different perspectives
5. Identification of gaps and limitations
6. Conclusions and recommendations for future research

For each source, include:
- Citation information
- Main arguments or findings
- Methodology (if applicable)
- Strengths and limitations
- Relevance to the research question

Synthesis approach:
- Group sources by theme or perspective
- Compare and contrast findings
- Identify consensus and disagreements
- Note methodological differences
- Highlight emerging trends

Available tools: {available_tools}
Quality standards: {quality_standards}""",
    required_variables=["research_topic", "available_tools"],
    optional_variables={"quality_standards": "Focus on peer-reviewed sources and empirical studies."},
    description="Template for conducting comprehensive literature reviews"
)

# Fact-Checking Prompt
fact_checking_prompt = PromptTemplate(
    name="fact_checking",
    template="""Fact-check the following claim: {claim}

Fact-checking methodology:
1. Break down the claim into verifiable components
2. Identify what evidence would support or refute each component
3. Search for authoritative sources and evidence
4. Evaluate the quality and reliability of evidence
5. Consider context and potential misinterpretation
6. Provide a verdict with confidence level

Verification steps:
- Check primary sources when possible
- Look for official statistics or records
- Consult expert opinions and peer-reviewed research
- Examine the claim's context and framing
- Consider alternative explanations

Provide:
- Verdict: True, False, Partially True, Unverified, or Misleading
- Confidence level: High, Medium, or Low
- Evidence summary with source citations
- Context and nuances that affect interpretation
- Related facts or corrections if needed

Available tools: {available_tools}""",
    required_variables=["claim", "available_tools"],
    description="Template for systematic fact-checking of claims and statements"
)

# Information Synthesis Prompt
synthesis_prompt = PromptTemplate(
    name="information_synthesis",
    template="""Synthesize information from multiple sources about: {topic}

Sources to synthesize: {sources}

Synthesis approach:
1. Identify common themes and patterns
2. Note areas of agreement and disagreement
3. Evaluate the strength of evidence for different claims
4. Consider methodological differences that might explain discrepancies
5. Identify knowledge gaps and uncertainties

Synthesis structure:
- Executive summary of key findings
- Major themes with supporting evidence
- Areas of consensus in the literature
- Conflicting findings and possible explanations
- Methodological considerations
- Limitations and gaps in current knowledge
- Implications and recommendations

Quality standards:
- Maintain objectivity and balance
- Clearly distinguish between established facts and emerging theories
- Acknowledge uncertainty and conflicting evidence
- Provide appropriate caveats and limitations
- Use {response_style} communication style""",
    required_variables=["topic", "sources", "response_style"],
    description="Template for synthesizing information from multiple research sources"
)

# Research Assistant Prompts Collection
research_prompts = AgentPrompts(
    system_prompt=research_system_prompt,
    planning_prompt=research_planning_prompt,
    custom_prompts={
        "source_evaluation": source_evaluation_prompt,
        "literature_review": literature_review_prompt,
        "fact_checking": fact_checking_prompt,
        "synthesis": synthesis_prompt
    }
)

# Research Assistant Configuration
research_config = AgentConfig(
    personality_traits="methodical, curious, objective, detail-oriented, intellectually honest",
    communication_style="clear, evidence-based, and academically rigorous",
    response_style="informative",
    custom_guidelines="""
1. Always prioritize accuracy over speed
2. Cite sources using appropriate academic format
3. Distinguish between correlation and causation
4. Acknowledge limitations and uncertainties
5. Provide context for statistical claims
6. Consider multiple perspectives on controversial topics
7. Flag outdated information
8. Recommend additional research when needed
    """.strip()
)

# Example usage function
def create_research_assistant():
    """Create a research assistant agent with specialized prompts."""
    from agentframe import Agent, OpenAIModel, ModelConfig

    model_config = ModelConfig(
        api_key="your-api-key",
        model="gpt-4",
        temperature=0.1  # Low temperature for factual accuracy
    )
    model = OpenAIModel(model_config)

    agent = Agent(
        model=model,
        tools=[],  # Add research tools like web search, database access, etc.
        config=research_config,
        prompts=research_prompts
    )

    return agent