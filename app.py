import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_huggingface import HuggingFaceEndpoint

# ----------------- Streamlit Page Config -------------------
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize Content from a URL')

# ----------------- Sidebar: Hugging Face API Key -------------------
with st.sidebar:
    hf_api_key = st.text_input("Huggingface API Token", value="", type="password")
    st.markdown("🔑 You can get a free token from [hf.co/settings/tokens](https://huggingface.co/settings/tokens)")

# ----------------- Main Input: URL -------------------
generic_url = st.text_input("Enter a YouTube or Web URL", placeholder="https://www.youtube.com/watch?v=example")

# ----------------- Prompt Template -------------------
prompt_template = """
You are a helpful summarizer. Provide a well-written and concise summary (around 300 words) of the following content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# ----------------- LLM Initialization (FIXED) -------------------
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

try:
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        huggingfacehub_api_token=hf_api_key,
        max_new_tokens=600,      # ✅ explicitly passed
        temperature=0.7          # ✅ explicitly passed
    )
except Exception as e:
    st.error("❌ Failed to initialize the HuggingFace model. Check your token or model name.")
    st.exception(e)
    st.stop()

# ----------------- Summarization Trigger -------------------
if st.button("Summarize the Content from YT or Website"):
    if not hf_api_key.strip() or not generic_url.strip():
        st.error("Please provide both the HuggingFace token and a valid URL.")
    elif not validators.url(generic_url):
        st.error("Please enter a valid URL (YouTube or website).")
    else:
        try:
            with st.spinner("🔄 Loading and summarizing content..."):

                # ✅ Load YouTube or Web content
                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                          "(KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"
                        }
                    )
                docs = loader.load()

                # ✅ Summarization Chain
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                # ✅ Display Output
                st.success("✅ Summary Generated Below")
                st.write(output_summary)

        except Exception as e:
            st.error("❌ An error occurred while processing the request.")
            st.exception(e)
