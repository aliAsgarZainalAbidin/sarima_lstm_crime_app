import streamlit as st

# @st.cache_resource
# def run_gradio():
#     p = subprocess.Popen(["python", "gradio_ui.py"])
#     time.sleep(5)  # Give the server time to start
#     return p

# # Start the Gradio process
# gradio_process = run_gradio()

gradio_url = "http://127.0.0.1:7860"
st.write(
    f'<iframe src="{gradio_url}" width="100%" height="100%"></iframe>',
    unsafe_allow_html=True,
)
