import operator
import os
from typing import Annotated, Sequence, TypedDict

import dotenv
import requests
import streamlit as st
from langchain.agents import tool
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import Field
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

dotenv.load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Kaybe Eval Llama 3"


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


@tool
def call_external_api(search_term: str) -> dict:
    """Fetches resources data from the Catalyst API, categorized by GOODS, FOOD, HOUSING, TRANSIT, HEALTH, MONEY, WORK, or LEGAL.

    Args:
        search_term (str): The user's search query, specifying the desired resource category (e.g., "food", "housing", "legal").

    Returns:
        A dictionary containing the API response data or an error message on failure.
    """
    subcat_id = 0
    cat_id = 0
    maintopic_id = 0
    user_zip = 0
    url = f"http://catalystws.celluphone.com/api/ResourcesLandings/{search_term}/{subcat_id}/{cat_id}/{maintopic_id}/{user_zip}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data[:1]
    else:
        return None


@tool
def call_get_outlet_products_api() -> list:
    """
    Fetches and processes product details for a specific company and outlet.

    Parameters:
    None

    Returns:
    list: A list of dictionaries containing filtered product details or an error message.
    """
    try:
        company_id = 1

        if not company_id:
            raise ValueError("Company ID is not set in StateManager.")
        
        pos_products_get_all_products = os.getenv("POS_PRODUCTS_GET_ALL_PRODUCTS")
        api_url = pos_products_get_all_products.format(company_id = company_id)
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        products = data.get("product", [])

        desired_fields = [
            "productID", "productName", "productDescr", "productTypeID", "isTrackInventory",
            "inventoryTrackingID", "unitID", "isReturnable", "createdUserID", "createdDt",
            "active", "imageIcon", "categoryTypeID", "categoryID", "brandID", "count",
            "inventory", "supplierID", "companyID", "costPrice", "sellingPrice",
            "availableStock", "stockReorderPoint", "outletsIds", "discount", "inventoryID"
        ]

        filtered_products = [
            {key: product.get(key) for key in desired_fields} for product in products
        ]

        return filtered_products

    except requests.exceptions.RequestException as e:
        error_message = f"Error fetching product details: {e}"
        return [{"error": "Failed to fetch product details", "details": error_message}]

def agent(state: dict) -> dict:
    """Invokes the agent model to generate a response based on the current state.

    Args:
        state (dict): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    prompt = PromptTemplate.from_template(
        """
        - For *sales performance questions*, highlight key metrics or trends and suggest improvements where applicable.
        - For *recent sales data queries*, include essential comparisons to previous periods and actionable insights.
        
        Question: {question}
                                            
        Just return the search term as final output
        search_term:
        
        """
    )
    messages = state["messages"]

    model_8b = "llama3-8b-8192"
    model_70b = "llama3-70b-8192"

    model = ChatGroq(temperature=0, model_name=model_70b)

    chain = prompt | model
    response = chain.invoke(messages)

    if response.content:
        search_term = response.content
        tool_response = call_get_outlet_products_api(search_term)

        response.tool_response = tool_response
        return {"messages": [response]}


def initialize_prompt_template() -> PromptTemplate:
    """Initialize the prompt template for generating responses."""
    template = PromptTemplate.from_template(
        """
        You are a helpful assistant named Kaybe who is very interative and provides suggestions whenever required.
        Generate a helpful and accurate response to the user's question, taking into consideration the provided context. 
        Use the question and context to craft a response that addresses the user's inquiry.
        In the final Answer do not mention about using the context to form the final response.
        If the context is empty or not present, please respond with "I do not have the information to answer this question.", 
        please do not use your Knowledege to answer in such cases where you do not have information in the context.
        In the response provide the site or contact details if available in the context.

        Question: {question}

        Context: {context} 

        Answer:
        """
    )
    return template



def generate(state: dict) -> dict:
    """Generate answer

    Args:
        state (dict): The current state

    Returns:
        dict: The updated state with re-phrased question
    """
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    question = messages[0].content
    docs = messages[1].tool_response

    prompt = initialize_prompt_template()

    model_8b = "llama3-8b-8192"
    model_70b = "llama3-70b-8192"

    llm = ChatGroq(temperature=0, model_name=model_70b)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
workflow.add_node("generate", generate)  # retrieval

# Call agent node to decide to retrieve or not
workflow.set_entry_point("agent")

workflow.add_edge("agent", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()


def main():
    st.title("Kaybe - Your AI Assistant")
    input_text = st.text_input("Enter your question or query:")
    if st.button("Ask Kaybe"):
        inputs = {
            "messages": [
                HumanMessage(
                    content=input_text
                )
            ]
        }
        final_output = None
        for output in app.stream(inputs):
            final_output = output
        st.write(final_output["generate"]["messages"][0])


if __name__ == "__main__":
    main()