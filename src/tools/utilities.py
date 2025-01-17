from datetime import datetime
import json
import logging
import os
from typing import Optional, Tuple
import dotenv
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain.prompts import PromptTemplate
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.agents import tool
from langchain_openai import ChatOpenAI
from config import RESTRICT_OPEN_ENDED_QUERIES
from src.openai_prompt_template.prompt_templates import get_date_prompt_template

dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_GPT_MODEL = os.getenv("OPENAI_GPT_MODEL")

MESSAGES_PLACEHOLDER = "{messages}"
RESTRICTED_STR = "Your response should always be: Sorry, I'm not permitted to answer this query."

UNRESTRICTED_STR = """The response should be short and concise and strictly follow the below rules: 
                1. The response should not exceed two hundred words. 
                2. The response should be written in easy-to-understand natural language. """


def initialize_open_ended_questions_prompt_template() -> PromptTemplate:

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{RESTRICTED_STR if RESTRICT_OPEN_ENDED_QUERIES else UNRESTRICTED_STR}"
                "Current user query: {user_query}. ",
            ),
            ("placeholder", MESSAGES_PLACEHOLDER),
        ]
    )
    return template


@tool
def handle_open_world_question(user_query: str) -> str:
    """
    Handles open-world questions and general knowledge questions asked by the user.

    Args:
        user_query (str): The question or query input by the user.

    Returns:
        str: The response to the user's query.

    Business Logic:
        This function is intended to process and respond to questions that fall under the category
        of open-world or general knowledge. The function will analyze the user query, determine 
        the type of question, and generate an appropriate response. 

    Examples of use:
        - user_query: "What is speed of light?" 
        - user_query: "Explain the theory of relativity."
        - user_query: "Write me an essay"
        - user_query: "How to cook ..."
    """
    try:
        prompt = initialize_open_ended_questions_prompt_template()

        model = ChatOpenAI(model_name=OPENAI_GPT_MODEL, temperature=0)
        chain = prompt | model
        response = chain.invoke(user_query)
        return response
    except Exception as e:
        logging.error(f"Failed to answer Open Ended user query: {e}")
        return "Failed to answer Open Ended user query"


@tool
def handle_date_question(user_query: str) -> Tuple[bool, Optional[datetime.date], Optional[datetime.date]]:
    """
        Handles date-related questions asked by the user. The function tries to extract and confirm dates mentioned 
        in the user query and returns relevant date information.

        Args:
            user_query (str): The question or query input by the user, which may contain date references.

        Examples of queries:
            - user_query: "What is today's date?" 
            - user_query: "What was last week's Friday date?"
            - user_query: "25-02-2024, what day was it?"
            - user_query: "What is the range date Last March month second week?"

        Returns:
            Tuple[bool, Optional[datetime.date], Optional[datetime.date]]:
                - A boolean indicating whether a date was detected in the user query.
                - The extracted start date (if applicable, else None).
                - The extracted end date (if applicable, else None).
    """
    try:
        today_date = datetime.now().strftime("%A, %Y-%m-%d")
        llm = ChatOpenAI(model_name=OPENAI_GPT_MODEL, temperature=0)

        prompt = PromptTemplate(
            input_variables=["today", "user_text"],
            template=get_date_prompt_template,
        )

        chain = RunnableSequence(prompt | llm)

        result = chain.invoke({"today": today_date, "user_text": user_query})
        result_content = result.content.strip('```json').strip('```').strip()
        date_dict = json.loads(result_content)
        logging.info(f"Parsed date dictionary: {date_dict}")

        # Function to parse a date string or return None
        def parse_date_or_none(date_string):
            if date_string is None:
                return None
            try:
                date_obj = datetime.strptime(date_string, "%Y-%m-%d").date()
                return date_obj.strftime("%m-%d-%Y")
            except ValueError as e:
                logging.error(f"Error parsing date '{date_string}': {e}")
                return None

        # Extract and parse both dates
        is_date_present = date_dict.get("date_reference_present")
        start_date = parse_date_or_none(date_dict.get("start_date"))
        end_date = parse_date_or_none(date_dict.get("end_date"))

        logging.info(
            f"Extracted dates - Is present: {is_date_present}, Start: {start_date}, End: {end_date}")
        return is_date_present, today_date, start_date, end_date

    except Exception as e:
        logging.error(f"Unexpected error in get_date function: {e}")
        return False, None, None
    
    
@tool
def collect_personal_data() -> dict:
    """
    Returns the detailed user profile data, including personal information, professional summary, experience,
    project details, technical skills, education, and participation.

    Parameters:
    None

    Returns:
    dict: A dictionary containing the user's detailed profile information.
    """
    joining_date = datetime(2022, 12, 26)  # 26-Dec-2022
    current_date = datetime.now()
    months_diff = (current_date.year - joining_date.year) * 12 + (current_date.month - joining_date.month)
    user_profile_data = {
        "name": "Saurabh Nale",
        "contact": {
            "phone": "+91 9511748787",
            "email": "saurabhnalepatil@gmail.com",
            "linkedin": {
                "url": "https://www.linkedin.com/in/saurabh-nale-967472216/",
                "display": "Linked-In"
            },
            "github": {
                "url": "https://github.com/saurabhnalepatil",
                "display": "GitHub"
            },
            "portfolio": {
                "url": "https://portfoliosaurabhnale.vercel.app/",
                "display": "Portfolio"
            }
        },
        "profile_summary": """
            Proficient Full-Stack Developer specializing in Python, Flask, and FastAPI for building robust and scalable web
            applications. Experienced in designing efficient backend systems, integrating with databases, and optimizing
            application performance. Passionate about solving complex problems and delivering seamless user experiences
            through clean and maintainable code.
        """,
        "experience": [
            { 
                "total_number_of_months_experince": months_diff 
            },
            {
                "title": "Software Developer",
                "company": "RBIS Technologies PVT. LTD.",
                "location": "Pune, Maharashtra, India",
                "duration": "July 2023 – Present",
                "responsibilities": [
                    "Collaborated closely with developers, designers, and stakeholders to successfully deliver 4+ projects.",
                    "Enhanced database performance and implemented optimized queries and scalable data management solutions.",
                    "Led backend development with Python and Flask, creating robust APIs and services.",
                    "Applied best practices in code structure, Git version control, and deployment."
                ]
            },
            {
                "title": "Software Developer Intern",
                "company": "RBIS Technologies PVT. LTD.",
                "location": "Pune, Maharashtra, India",
                "duration": "Dec 2022 – Jun 2023",
                "responsibilities": [
                    "Gained hands-on experience with various software development tools and platforms.",
                    "Developed interactive and responsive user interfaces using Angular and Bootstrap.",
                    "Contributed to the development of internal tools and documentation to improve project workflows."
                ]
            }
        ],
        "projects": [
            {
                "name": "RetailBuddy",
                "technologies": ["Python", "FastAPI", "LangChain", "Langsmith", "PostgreSQL", "OpenAI", "ChromaDB"],
                "duration": "Sept 2024 - Present",
                "description": [
                    "Developed a Retail Control System AI Bot leveraging advanced NLP models and voice input.",
                    "Integrated ChromaDB to store and manage vector embeddings for AI-driven query handling.",
                    "Utilized LangChain and LangGraph to build a versatile agent-based system."
                ]
            },
            {
                "name": "Kaybe",
                "technologies": ["Python", "Fast API", "Langchain", "Langsmith", "PostgreSQL", "SQL-Alchemy", "OpenAI"],
                "duration": "Mar 2024 - Aug 2024",
                "description": [
                    "Engineered an AI-powered Q&A chatbot leveraging advanced NLP frameworks.",
                    "Streamlined query handling with PostgreSQL for efficient storage and context-driven responses.",
                    "Designed a multilingual system with a custom language changer."
                ]
            },
            {
                "name": "Essay Grading System",
                "technologies": ["Python", "Azure Function", "Flask", "MS-SQL", "Pinecone", "Azure DevOps"],
                "duration": "Jun 2023 - Mar 2024",
                "description": [
                    "Developed an AI-based system for grading essays using Azure Functions and SQL Server.",
                    "Achieved 80% improvement in grading efficiency with AI-driven algorithms.",
                    "Optimized CI/CD pipelines using Azure DevOps for seamless updates and releases."
                ]
            },
            {
                "name": "HuntScaler",
                "technologies": ["Angular", "Flask", "Python", "Typescript", "MS-SQL", "Bootstrap"],
                "duration": "Dec 2022 - May 2023",
                "description": [
                    "Developed an assessment portal with a variety of question types.",
                    "Enhanced user experience with dynamic dashboards, advanced data filtering, and security.",
                    "Improved system performance, reducing response time by 25%."
                ]
            }
        ],
        "technical_skills": {
            "languages": ["Python", "TypeScript"],
            "frameworks": ["Flask", "FastAPI", "Angular", "Azure Functions"],
            "libraries": ["Pandas", "NumPy", "Matplotlib", "SQL-Alchemy"],
            "databases": ["MS-SQL", "PostgreSQL", "MongoDB", "Pinecone Vector DB", "ChromaDB", "Azure Table"],
            "developer_tools": ["Git", "Docker", "Azure DevOps", "Postman", "VS Code", "Visual Studio"]
        },
        "education": [
            {
                "degree": "Master of Computer Application",
                "institution": "K. B. C. North Maharashtra University",
                "location": "Jalgaon",
                "duration": "2021 – 2023"
            },
            {
                "degree": "Bachelor of Science in Computer Science",
                "institution": "Sant Gadge Baba Amravati University",
                "location": "Amravati",
                "duration": "2018 – 2021"
            }
        ],
        "participations": [
            {
                "event": "PINNACLE - National Level Competition",
                "date": "05 Sept 2022",
                "institution": "School of Computer Science, NMU",
                "location": "Jalgaon, Maharashtra, India",
                "details": [
                    "Demonstrated strong Java programming skills by solving multiple algorithmic and problem-solving challenges.",
                    "Showcased adaptability and technical proficiency in diverse problem sets under competitive conditions."
                ]
            }
        ],
        "extra_details": [
            {
                "joining_date": "26-Dec-2022",
                "birthdate": "16-Mar-2000",
                "permanent_address": {
                    "street": "Pehalvan Nagar, Near Ganpati Temple",
                    "village": "Mhaisawadi",
                    "taluka": "Malkapur",
                    "district": "Buldhana",
                    "state": "Maharashtra",
                    "pincode": "443112"
                },
                "current_address": {
                    "flat_number": "A1-404",
                    "society": "Pyramid County",
                    "area": "Pansare Colony, Bhukum",
                    "pincode": "412115",
                    "city": "Pune",
                    "state": "Maharashtra"
                }
            }
        ]
    }
    
    return user_profile_data


