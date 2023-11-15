import gradio as gr
import random
import time


from RagLLM.pipeline import RagBot

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=1000)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    bot=RagBot()
    
    def respond(message, chat_history):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()

