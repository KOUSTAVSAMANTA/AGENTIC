import streamlit as st
from langchain.memory import ConversationBufferMemory
from agents import agent1
import re


import time
from pymongo import MongoClient
import os

from langchain.prompts import PromptTemplate

from langchain_groq import ChatGroq



os.environ["GROQ_API_KEY"] = "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"

st.set_page_config(initial_sidebar_state="expanded")
# llm2 = Ollama(model="mistral")
llm2 = ChatGroq(temperature=0, model_name="llama3-70b-8192")

try:
    st.session_state.username = st.session_state.signin_username
except Exception as e:
    try:
        if st.session_state.username == "":
            st.session_state.username = ""
        else:
            pass
    except Exception as e2:
        st.switch_page("./pages/Login & Register.py")

try:
    get = st.session_state.signin_username
except Exception as e:
    if st.session_state.username != "":
        pass
    else:
        st.switch_page("./pages/Login & Register.py")


def clear_memory():
    z = delete_entry(user_id=st.session_state.username, collection2=collection2)
    st.session_state.messages = []
    st.toast("Memory cleared !! " + z, icon='🤖')


def on_copy_click(text):
    st.session_state.copied.append(text)


if "copied" not in st.session_state:
    st.session_state.copied = []

os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"


def read_and_save_file():
    try:
        st.session_state.phi = ""
    except Exception as e:
        st.session_state.phi = ""
        pass


client = MongoClient("mongodb+srv://koustav:koustav2003@cluster0.wmnc2.mongodb.net/?retryWrites=true&w=majority",
                     ssl=True)
db = client['AUTH']
collection = db['Authentication']
collection2 = db['Memory']


def load_or_create_memory(user_id, collection2):
    if collection2.find_one({"user_id": user_id}) is not None:
        _mem = ConversationBufferMemory(memory_key=memory_key)
        memory = collection2.find_one({"user_id": user_id})["memory_key"]
        for i in memory:
            try:
                _mem.chat_memory.add_user_message(i['Human'])
                _mem.chat_memory.add_ai_message(i['AI'])
            except Exception as e:
                pass
        return _mem
    else:
        memory = ConversationBufferMemory(memory_key=memory_key)
        return memory


def check_user(user_id, memory_key, collection2):
    if collection2.find_one({"user_id": user_id}) is not None:
        collection2.update_one(
            {"user_id": user_id}, {"$set": {"memory_key": memory_key}})
        return "updated"
    else:
        collection2.insert_one({"user_id": user_id, "memory_key": memory_key})
        return "created"


def delete_entry(user_id, collection2):
    collection2.delete_one({"user_id": user_id})
    return f"Deleted Previous Conversations with {user_id}"


st.sidebar.title("Chat with CODA")
st.title("Chat with CODA")

template = """You are a nice chatbot having a conversation with a human.your name is CODA powerd by mistral, llava and pgvector.
Make your Responses Brief.
Try to respond from previous Conversation if the answer is present.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""

if "messages" not in st.session_state:
    st.session_state.messages = []

memory_key = "chat_history"

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        try:
            st.image(message["img"], caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            pass
        st.markdown(message["content"])

uploaded_images = []
uploaded_files = []

# with st.sidebar:
#     genre = st.radio(
#         "Select the Type of File to Query with:- ",
#         ["***Image***", "***Documents***", "***Connect Internet***"],
#         index=None,
#     )
#
#     if genre == "***Image***":
#         st.session_state.searchmode = False
#         uploaded_images = st.sidebar.file_uploader("Upload Image for image query", type=["jpg", "png", "jpeg"],
#                                                    accept_multiple_files=True)
#     elif genre == "***Connect Internet***":
#         st.session_state.searchmode = True
#     else:
#         st.session_state.searchmode = False
#         uploaded_files = st.sidebar.file_uploader(
#             "Upload document for Document query",
#             key="file_uploader",
#             on_change=read_and_save_file,
#             accept_multiple_files=True,
#             type=["pdf"]
#         )

if prompt := st.chat_input("What is up?"):
    # if st.session_state.searchmode:
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)
    #     with st.chat_message("assistant"):
    #         message_placeholder = st.empty()
    #         memory = load_or_create_memory(user_id=st.session_state.username, collection2=collection2)
    #         full_response = ""
    #         z, memory = search(prompt, memory)
    #         assistant_response = z
    #         for chunk in assistant_response.split(" "):
    #             full_response += chunk + " "
    #             time.sleep(0.05)
    #             message_placeholder.markdown(full_response + "▌")
    #         message_placeholder.markdown(full_response)
    #     z = memory.chat_memory.messages
    #     mem = [{"Human": z[i].content, "AI": z[i + 1].content} for i in range(0, len(z), 2)]
    #     ch = check_user(user_id=st.session_state.username, memory_key=mem, collection2=collection2)
    #     st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    #     st.button("clear memory", on_click=clear_memory, args=())
    #
    # elif uploaded_files:
    #     st.session_state.messages.append(
    #         {"role": "user", "content": prompt, "files": [file.name for file in uploaded_files]})
    #     knowledge_bases = []
    #     for uploaded_file in uploaded_files:
    #         save_folder = './files'
    #         save_path = Path(save_folder, uploaded_file.name)
    #         with open(save_path, mode='wb') as w:
    #             w.write(uploaded_file.getvalue())
    #         pdf_knowledge_base = create_Kb(path=str(save_path))
    #         pdf_knowledge_base.load(recreate=False)
    #         knowledge_bases.append(pdf_knowledge_base)
    #
    #     combined_knowledge_base = CombinedKnowledgeBase(
    #         sources=knowledge_bases,
    #         vector_db=PgVector2(
    #             embedder=OllamaEmbedder(model="nomic-embed-text", dimensions=768),
    #             collection="combined_documents",
    #             db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    #         ),
    #     )
    #
    #     assistant = Assistant(
    #         llm=Groq(model="llama3-70b-8192"),
    #         # llm = chatOpenai
    #         knowledge_base=combined_knowledge_base,
    #         add_references_to_prompt=True,
    #         debug_mode=True
    #     )
    #
    #     st.session_state.phi = assistant
    #
    #     with st.chat_message("user"):
    #         st.markdown("Files: " + ", ".join([file.name for file in uploaded_files]))
    #         st.markdown(prompt)
    #
    #     with st.chat_message("assistant"):
    #         message_placeholder = st.empty()
    #         memory = load_or_create_memory(user_id=st.session_state.username, collection2=collection2)
    #         full_response = ""
    #         response = st.session_state.phi.run(prompt, stream=False)
    #         assistant_response = str(response)
    #         memory.save_context({"input": prompt + 'Files: ' + ', '.join([file.name for file in uploaded_files])},
    #                             {"output": assistant_response})
    #         for chunk in assistant_response.split(" "):
    #             full_response += chunk + " "
    #             time.sleep(0.05)
    #             message_placeholder.markdown(full_response + "▌")
    #         message_placeholder.markdown(full_response)
    #     z = memory.chat_memory.messages
    #     mem = [{"Human": z[i].content, "AI": z[i + 1].content} for i in range(0, len(z), 2)]
    #     ch = check_user(user_id=st.session_state.username, memory_key=mem, collection2=collection2)
    #     st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    #     st.button("clear memory", on_click=clear_memory, args=())
    #
    # elif uploaded_images:
    #     st.session_state.messages.append(
    #         {"role": "user", "content": prompt, "imgs": [img.name for img in uploaded_images]})
    #     with st.chat_message("user"):
    #         for img in uploaded_images:
    #             st.image(img, caption="Uploaded Image", use_column_width=True)
    #         st.markdown(prompt)
    #
    #     image_paths = []
    #     for uploaded_image in uploaded_images:
    #         image = PIL.Image.open(uploaded_image)
    #         image_path = f'images/{uploaded_image.name}'
    #         image.save(image_path)
    #         image_paths.append(image_path)
    #
    #     with st.chat_message("assistant"):
    #         message_placeholder = st.empty()
    #         memory = load_or_create_memory(user_id=st.session_state.username, collection2=collection2)
    #         full_response = ""
    #         llm = ola.generate(model='llava', prompt=prompt, images=image_paths)
    #         assistant_response = llm['response']
    #         for chunk in assistant_response.split(" "):
    #             full_response += chunk + " "
    #             time.sleep(0.05)
    #             message_placeholder.markdown(full_response + "▌")
    #         message_placeholder.markdown(full_response)
    #
    #     memory.save_context({"input": prompt + 'Images: ' + ', '.join([img.name for img in uploaded_images])},
    #                         {"output": assistant_response})
    #     z = memory.chat_memory.messages
    #     mem = [{"Human": z[i].content, "AI": z[i + 1].content} for i in range(0, len(z), 2)]
    #     ch = check_user(user_id=st.session_state.username, memory_key=mem, collection2=collection2)
    #     st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    #     st.button("clear memory", on_click=clear_memory, args=())
    #
    # else:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        memory = load_or_create_memory(user_id=st.session_state.username, collection2=collection2)
        full_response = ""
        prompt_t = PromptTemplate.from_template(template)
        rest = agent1(prompt)
        _printed = set()
        max_length = 1500

        responses_agent = []
        # for i in rest:
        message = rest.get("messages")
        # print(message)

        # Regular expression to match AIMessage content
        pattern = r"AIMessage\(content=(.+?),\s*response_metadata="

        # Find all matches
        matches = re.findall(pattern, str(message), re.DOTALL)

        # Clean up and print the matches
        for match in matches:
            # Remove leading and trailing spaces and newlines
            cleaned_match = match.strip().replace('\n', ' ')
            # Print or use the cleaned content
            # print(cleaned_match)
            try:
                if eval(cleaned_match)[0]["text"] in responses_agent:
                    pass
                else:
                    responses_agent.append(eval(cleaned_match)[0]["text"])
            except Exception as e:
                responses_agent.append(cleaned_match)

        for i in responses_agent:
            print("--------------------------")
            print(i)
        # print(responses_agent)
        # conversation = LLMChain(
        #     llm=llm2,
        #     prompt=prompt_t,
        #     verbose=False,
        #     memory=memory,
        #     output_parser=StrOutputParser()
        # )
        printing = []
        for response in responses_agent[-2:]:
            # Check if the response is already printed
            if response in printing:
                continue  # Skip if already printed

            # Add the response to the printed list
            printing.append(response)

            # Create a new placeholder for each response
            message_placeholder = st.empty()

            # Prepare the response
            assistant_response = response

            # Initialize full response for typing effect
            full_response = ""

            # Simulate typing effect
            for chunk in assistant_response.split(" "):
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")

            # Final display of the response without the typing cursor
            message_placeholder.markdown(full_response.strip(), unsafe_allow_html=True)

            # Append the response to the session state messages
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            # z = memory.chat_memory.messages
            #
            # mem = [{"Human": z[i].content, "AI": z[i + 1].content} for i in range(0, len(z), 2)]
            # ch = check_user(user_id=st.session_state.username, memory_key=mem, collection2=collection2)
        st.button("clear memory", on_click=clear_memory, args=())


# conda create --name GEMINI2  python=3.10
# pip install -u langgraph langchain-community langchain-anthropic tavily-python pandas streamlit st_pages passlib ollama pymongo clipboard phidata duckduckgo-search langchainhub langchain pgvector openai groq langchain_groq langchain_openai