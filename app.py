import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain,LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

st.set_page_config(page_title="Maths Problem Solver",page_icon="🧮")
st.title("Maths Problem Solver Using Gemma2")

groq_api_key=st.sidebar.text_input("Enter your Groq api key",type="password")
if not groq_api_key:
    st.info("Enter your Groq api key to continue")
    st.stop()

llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

wikipedia_wrapper=WikipediaAPIWrapper()
wikipedia_tool=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching wikipedia to find information on the topics mentioned"
)

math_chain=LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering maths questions. Only mathematical expressions need to be provided"
)

template="""
You're an agent tasked for solving the user's mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(input_variables=["question"],template=template)

chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions"
)

agent=initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"Hi, I'm a Math chatbot who can answer all your maths questions"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

question=st.text_area("Enter your question")

if st.button("Answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)

            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=agent.run(st.session_state.messages,callbacks=[st_cb])

            st.session_state.messages.append({"role":"assistant","content":response})
            st.write("Response:")
            st.success(response)
    else:
        st.warning("Enter a question")
