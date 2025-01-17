import logging
import os
import dotenv

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq

from src.tools.utilities import ( handle_open_world_question, 
                                  handle_date_question,
                                  collect_personal_data)
from src.openai_prompt_template.prompt_templates import ( initialize_primary_assistant_prompt_template )
from src.graph.graph_state import State
from src.graph.assistant import Assistant
from src.graph.utilities import create_tool_node_with_fallback
from typing import Literal


dotenv.load_dotenv()
CHECKPOINT = os.getenv("CHECKPOINT_PATH")

def build_graph():
    try:
        builder = StateGraph(State)
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
        #llm = ChatOpenAI(model_name=OPENAI_GPT_MODEL, temperature=0)
        
        # Primary Assistant
        primary_assistant_prompt = initialize_primary_assistant_prompt_template()
        primary_assistant_tools = [
            handle_open_world_question, 
            handle_date_question, collect_personal_data
            ] 

        assistant_runnable = primary_assistant_prompt | llm.bind_tools(
            primary_assistant_tools)
        builder.add_node("primary_assistant", Assistant(assistant_runnable))
        builder.add_node("primary_assistant_tools",
                        create_tool_node_with_fallback(primary_assistant_tools))

        def route_primary_assistant(
            state: State,
        ) -> Literal[
            "primary_assistant_tools",
            "__end__",
        ]:
            route = tools_condition(state)
            if route == END:
                return END
            tool_calls = state["messages"][-1].tool_calls
            if tool_calls:
                return "primary_assistant_tools"
            raise ValueError("Invalid route")

        builder.add_conditional_edges(
            "primary_assistant",
            route_primary_assistant,
            {
                "primary_assistant_tools": "primary_assistant_tools",
                END: END,
            },
        )
        builder.add_edge("primary_assistant_tools", "primary_assistant")
        builder.set_entry_point("primary_assistant")

        memory = SqliteSaver.from_conn_string(CHECKPOINT)
        return builder.compile(checkpointer=memory)
    except Exception as e:
        logging.error(f"Error building graph: {e}")
        return None