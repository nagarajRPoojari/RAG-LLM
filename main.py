import gradio as gr
import random
import time


from RagLLM.pipeline import RagBot

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    llm=RagBot()
    
    def respond(message, chat_history):
        bot_message = llm.bot(message)['result']
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()

