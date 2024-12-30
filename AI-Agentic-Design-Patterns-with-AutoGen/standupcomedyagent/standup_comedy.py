import streamlit as st
from autogen import ConversableAgent
import openai
import pprint

# Streamlit Sidebar for API Key Input
st.sidebar.title("Settings")
OPENAI_API_KEY = st.sidebar.text_input(
    "Enter your OpenAI API Key", 
    type="password", 
    help="Your OpenAI API key will be used to access GPT models."
)

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    st.sidebar.warning("Please enter your OpenAI API key to proceed.")

# Function to initialize agents
def create_agent(name, system_message, termination_condition=None):
    return ConversableAgent(
        name=name,
        system_message=system_message,
        llm_config={"model": "gpt-3.5-turbo"},
        human_input_mode="NEVER",
        is_termination_msg=termination_condition,
    )

# Streamlit UI
st.title("Multi-Agent Stand-Up Comedy Chatbot")
st.sidebar.header("Chat Settings")
max_turns = st.sidebar.slider("Maximum Turns", 1, 10, 5)

# Agent setup
cathy_message = (
    "Your name is Cathy and you are a stand-up comedian. "
    "When you're ready to end the conversation, say 'I gotta go'."
)
joe_message = (
    "Your name is Joe and you are a stand-up comedian. "
    "When you're ready to end the conversation, say 'I gotta go'."
)

if OPENAI_API_KEY:
    cathy = create_agent("Cathy", cathy_message, lambda msg: "I gotta go" in msg["content"])
    joe = create_agent("Joe", joe_message, lambda msg: "I gotta go" in msg["content"])

    # Start the chat
    if st.button("Start Chat"):
        st.write("### Conversation")
        chat_result = joe.initiate_chat(
            recipient=cathy,
            message="I'm Joe. Cathy, let's keep the jokes rolling.",
            max_turns=max_turns,
        )

        # Display chat history
        for turn in chat_result.chat_history:
            if turn["role"] == "assistant":
                st.markdown(f"**{joe.name}:** {turn['content']}")
            elif turn["role"] == "user":
                st.markdown(f"**{cathy.name}:** {turn['content']}")

        # Display conversation summary
        st.write("### Summary")
        st.write(chat_result.summary)

        # Display cost details
        st.write("### Cost Details")
        st.json(chat_result.cost)

        # Save chat history option
        if st.button("Download Chat History"):
            chat_history_str = pprint.pformat(chat_result.chat_history)
            st.download_button(
                label="Download Chat History",
                data=chat_history_str,
                file_name="chat_history.txt",
                mime="text/plain",
            )
else:
    st.warning("Please enter your OpenAI API key in the sidebar to start the chatbot.")
