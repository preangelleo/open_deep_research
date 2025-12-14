"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import os
import aiohttp
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
)
from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    anthropic_websearch_called,
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    openai_websearch_called,
    remove_up_to_last_ai_message,
    think_tool,
    get_model_token_limit,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
)
from open_deep_research.token_utils import get_token_usage

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and fallback logic
    safety_net_config = {
        "model": configurable.safety_net_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.safety_net_model, config),
        "tags": ["langsmith:nostream"]
    }

    primary_model = configurable_model.with_config(model_config).with_structured_output(ClarifyWithUser, include_raw=True)
    safety_net_model = configurable_model.with_config(safety_net_config).with_structured_output(ClarifyWithUser, include_raw=True)

    clarification_model = (
        primary_model
        .with_fallbacks([safety_net_model])
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    parsed_response = response["parsed"]
    raw_response = response["raw"]
    
    # Step 4: Route based on clarification analysis
    if parsed_response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={
                "messages": [AIMessage(content=parsed_response.question)],
                **get_token_usage(raw_response)  # Track tokens
            }
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={
                "messages": [AIMessage(content=parsed_response.verification)],
                **get_token_usage(raw_response)  # Track tokens
            }
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation with fallback
    safety_net_config = {
        "model": configurable.safety_net_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.safety_net_model, config),
        "tags": ["langsmith:nostream"]
    }

    primary_model = configurable_model.with_config(research_model_config).with_structured_output(ResearchQuestion, include_raw=True)
    safety_net_model = configurable_model.with_config(safety_net_config).with_structured_output(ResearchQuestion, include_raw=True)

    research_model = (
        primary_model
        .with_fallbacks([safety_net_model])
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    parsed_response = response["parsed"]
    raw_response = response["raw"]
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": parsed_response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=parsed_response.research_brief)
                ]
            },
            **get_token_usage(raw_response),  # Track tokens using raw response
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, fallback and model settings
    safety_net_config = {
        "model": configurable.safety_net_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.safety_net_model, config),
        "tags": ["langsmith:nostream"]
    }

    primary_bound = configurable_model.with_config(research_model_config).bind_tools(lead_researcher_tools)
    safety_net_bound = configurable_model.with_config(safety_net_config).bind_tools(lead_researcher_tools)

    research_model = (
        primary_bound
        .with_fallbacks([safety_net_bound])
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
            **get_token_usage(response),  # Track tokens
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Aggregate raw notes from all research results
            raw_notes_concat = "\n".join([
                "\n".join(observation.get("raw_notes", [])) 
                for observation in tool_results
            ])
            
            # Aggregate token usage from researcher sub-tasks
            subtask_input_tokens = sum(obs.get("total_input_tokens", 0) for obs in tool_results)
            subtask_output_tokens = sum(obs.get("total_output_tokens", 0) for obs in tool_results)
            
            update_payload["total_input_tokens"] = subtask_input_tokens
            update_payload["total_output_tokens"] = subtask_output_tokens
            
            if raw_notes_concat:
                update_payload["raw_notes"] = [raw_notes_concat]
                
        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, fallback and settings
    safety_net_config = {
        "model": configurable.safety_net_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.safety_net_model, config),
        "tags": ["langsmith:nostream"]
    }

    primary_bound = configurable_model.with_config(research_model_config).bind_tools(tools)
    safety_net_bound = configurable_model.with_config(safety_net_config).bind_tools(tools)

    research_model = (
        primary_bound
        .with_fallbacks([safety_net_bound])
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
            **get_token_usage(response),  # Track tokens
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    has_native_search = (
        openai_websearch_called(most_recent_message) or 
        anthropic_websearch_called(most_recent_message)
    )
    
    if not has_tool_calls and not has_native_search:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    primary_config = {
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    safety_net_config = {
        "model": configurable.safety_net_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.safety_net_model, config),
        "tags": ["langsmith:nostream"]
    }

    primary = configurable_model.with_config(primary_config)
    safety_net = configurable_model.with_config(safety_net_config)

    synthesizer_model = primary.with_fallbacks([safety_net])
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))
    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content],
                **get_token_usage(response),  # Track tokens
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    # Configure the 3-tier model priority queue
    # Priority 1: Default Writer (gemini-3-pro)
    p1_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Priority 2: Fallback Writer (gemini-2.5-pro)
    p2_config = {
        "model": configurable.writer_fallback_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.writer_fallback_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Priority 3: Safety Net (gemini-2.5-flash-lite)
    p3_config = {
        "model": configurable.safety_net_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.safety_net_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    p1_model = configurable_model.with_config(p1_config)
    p2_model = configurable_model.with_config(p2_config)
    p3_model = configurable_model.with_config(p3_config)
    
    # Writer Fallback Chain: P1 -> P2 -> P3
    writer_model = p1_model.with_fallbacks([p2_model, p3_model])
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            final_report = await writer_model.ainvoke([
                HumanMessage(content=final_report_prompt)
            ])
            
            # Return successful report generation
            return {
                "final_report": final_report.content, 
                "messages": [final_report],
                **cleared_state,
                **get_token_usage(final_report),  # Track tokens
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

async def notifier_node(state: AgentState, config: RunnableConfig):
    """Notifier node that sends the final report via Webhook (Fire-and-Forget).
    
    This node is purely deterministic code (no LLM). It constructs a detailed
    JSON payload following the enhanced schema and POSTs it to the configured WEBHOOK_URL.
    """
    final_report = state.get("final_report", "")
    # Robust handling: final_report might be a list (from reducers) or a string
    if isinstance(final_report, list):
        # Assuming the last item is the most recent report if it's a list
        if final_report:
            final_report = final_report[-1]
        else:
            final_report = ""
    # Ensure it's a string (it might be an AIMessage object if raw)
    if hasattr(final_report, "content"):
        final_report = str(final_report.content)
    else:
        final_report = str(final_report)
        
    # Restore missing metadata extraction
    run_id = config.get("configurable", {}).get("run_id") or config.get("run_id")
    thread_id = config.get("configurable", {}).get("thread_id") or config.get("thread_id")
    messages = state.get("messages", [])
    
    original_prompt = "Unknown prompt"
    for msg in messages:
        if isinstance(msg, HumanMessage):
            original_prompt = str(msg.content)
            break
            
    # Title extraction
    title = "Research Report"
    if final_report:
        try:
            first_line = final_report.strip().split('\n')[0]
            title = first_line.replace('#', '').strip()
        except Exception:
            pass
            
    word_count = len(final_report.split()) if final_report else 0

    # 4. Token Usage & Cost Calculation
    
    # Pricing Registry (USD per 1M tokens)
    # Support for flat rates and tiered rates based on context length (<= 200k vs > 200k).
    PRICING_REGISTRY = {
        "gemini-2.5-flash": {
            "type": "flat",
            "input": 0.30, 
            "output": 2.50
        },
        "gemini-2.5-pro": {
            "type": "tiered",
            "cutoff": 200000,
            "input": {"standard": 1.25, "long": 2.50},
            "output": {"standard": 10.00, "long": 15.00}
        },
        "gemini-3.0-pro": {
            "type": "tiered",
            "cutoff": 200000,
            "input": {"standard": 2.00, "long": 4.00},
            "output": {"standard": 12.00, "long": 18.00}
        },
        "gemini-3-pro-preview": { # Alias
            "type": "tiered",
            "cutoff": 200000,
            "input": {"standard": 2.00, "long": 4.00},
            "output": {"standard": 12.00, "long": 18.00}
        }
    }

    # Determine Model Name from Config
    configurable = config.get("configurable", {})
    # Check keys in order of likelihood for user API input
    model_name = (
        configurable.get("model_name") 
        or configurable.get("model") 
        or configurable.get("final_report_model") 
        or "gemini-2.5-flash" # Default fallback
    )
    
    # Determine Webhook URL from Config (API input overrides env var)
    webhook_url = (
        configurable.get("webhook_endpoint") 
        or os.environ.get("WEBHOOK_URL")
    )
    
    # Determine Pricing Config
    pricing_config = PRICING_REGISTRY.get("gemini-2.5-flash") # Default
    model_key = model_name.lower()
    
    # Simple substring match attempt if exact match fails
    if model_key not in PRICING_REGISTRY:
        for key, conf in PRICING_REGISTRY.items():
            if key in model_key:
                pricing_config = conf
    else:
        pricing_config = PRICING_REGISTRY[model_key]

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for msg in messages:
        usage = None
        if hasattr(msg, 'response_metadata'):
            usage = msg.response_metadata.get('usage_metadata') or msg.response_metadata.get('token_usage')
        elif hasattr(msg, 'additional_kwargs'):
             usage = msg.additional_kwargs.get('usage_metadata')
        
        if usage:
            inp = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
            out = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
            
            total_input_tokens += inp
            total_output_tokens += out
            
            # Calculate cost for this specific turn
            inp_rate = 0.0
            out_rate = 0.0
            
            if pricing_config["type"] == "flat":
                inp_rate = pricing_config["input"]
                out_rate = pricing_config["output"]
            elif pricing_config["type"] == "tiered":
                cutoff = pricing_config["cutoff"]
                if inp <= cutoff:
                    inp_rate = pricing_config["input"]["standard"]
                    out_rate = pricing_config["output"]["standard"]
                else:
                    inp_rate = pricing_config["input"]["long"]
                    out_rate = pricing_config["output"]["long"]
            
            turn_cost = (inp / 1_000_000 * inp_rate) + (out / 1_000_000 * out_rate)
            total_cost += turn_cost
            
    # Add cumulative tokens tracked in state
    state_input = state.get("total_input_tokens", 0)
    state_output = state.get("total_output_tokens", 0)
    
    # Calculate cost for accumulated tokens (using same pricing config as model, simple approximation)
    # Ideally should track model per usage but simplified for now
    if pricing_config["type"] == "flat":
        state_cost = (state_input / 1_000_000 * pricing_config["input"]) + (state_output / 1_000_000 * pricing_config["output"])
    else: # Tiered
        # Simplified tiered calculation (assuming all standard for accumulated to avoid complexity)
        state_cost = (state_input / 1_000_000 * pricing_config["input"]["standard"]) + (state_output / 1_000_000 * pricing_config["output"]["standard"])

    total_cost += state_cost
    total_input_tokens += state_input
    total_output_tokens += state_output

    # Debug token usage
    print(f"DEBUG: Model: {model_name}. Tokens: {total_input_tokens}/{total_output_tokens}. Total Cost: ${total_cost:.6f}")

    if not webhook_url:
        return {"messages": [AIMessage(content="WEBHOOK_URL not configured. Report not sent.")]}

    # Construct Enhanced Flat Payload
    payload = {
        "research_result": final_report,
        "title": title,
        "original_prompt": original_prompt,
        "run_id": run_id,
        "task_id": run_id, # Alias for compatibility
        "thread_id": thread_id,
        "status": "success",
        "timestamp": get_today_str(),
        "report_format": "markdown",
        "word_count": word_count,
        "cost_in_usd": round(total_cost, 6),
        "model_name": model_name
    }
    
    print(f"DEBUG: notifier_node started. WEBHOOK_URL={webhook_url}")
    print(f"DEBUG: Payload keys: {list(payload.keys())}")
    print(f"DEBUG: Payload content sample: {str(payload)[:500]}...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload, timeout=30) as response:
                if response.status >= 200 and response.status < 300:
                    print(f"DEBUG: Webhook success. Status: {response.status}")
                    return {"messages": [AIMessage(content=f"Report successfully sent to webhook: {webhook_url}")]}
                else:
                    response_text = await response.text()
                    print(f"DEBUG: Webhook failed. Status: {response.status}. Response: {response_text}")
                    return {"messages": [AIMessage(content=f"Failed to send report to webhook. Status: {response.status}")]}
    except Exception as e:
        return {"messages": [AIMessage(content=f"Error sending report to webhook: {e}")]}

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase
deep_researcher_builder.add_node("notifier_node", notifier_node)                      # Webhook delivery phase

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", "notifier_node")       # Report to webhook
deep_researcher_builder.add_edge("notifier_node", END)                             # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()