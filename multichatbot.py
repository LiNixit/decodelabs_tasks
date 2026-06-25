# -*- coding: utf-8 -*-
import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the Hugging Face Client
# WARNING: Keep your API key secret! Do not share it in public forums.
client = InferenceClient(
    api_key="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"
)

# Define the available chatbots and their model IDs
AVAILABLE_MODELS = {
    "Meta Llama 3.1 (8B)": "meta-llama/Llama-3.1-8B-Instruct",
    "Minimax AI (7B)": "MiniMaxAI/MiniMax-M3" # Replaced with a widely supported chat model
}

def predict(message, history, bot_choice):
    """
    Handles the chat completion requests.
    'history' is now correctly managed by Gradio's internal state per user session.
    """
    # Get the correct model ID based on the user's choice
    model_id = AVAILABLE_MODELS.get(bot_choice, "meta-llama/Llama-3.1-8B-Instruct")

    # Format the history for the InferenceClient
    current_chat_history = [{"role": "system", "content": "Your helpful AI assistant"}]
    for msg in history:
        current_chat_history.append(msg)

    current_chat_history.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=current_chat_history,
            max_tokens=300
        )
        ai_response = response.choices[0].message.content
    except Exception as e:
        print(f"Error during AI response generation: {e}")
        gr.Warning(f"An error occurred: {e}")
        ai_response = f"Sorry, I encountered an error: {e}. Please try again or clear the chat."

    # Update the Gradio history block
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ai_response})

    # Return an empty string to clear the input box, and the updated history
    return "", history

# --------------------------
# GRADIO UI BUILDER
# --------------------------
with gr.Blocks(theme='soft') as demo:

    # --- SCREEN 1: Welcome Screen ---
    with gr.Column(visible=True) as welcome_screen:
        gr.Markdown(
            """
            <div style="text-align: center; padding: 50px;">
                <h1>👋 Welcome to the Multi-Bot Platform</h1>
                <p>Select an AI assistant to begin your conversation.</p>
            </div>
            """
        )

        # User selects the bot here
        bot_selection = gr.Radio(
            choices=list(AVAILABLE_MODELS.keys()),
            value="Meta Llama 3.1 (8B)",
            label="Choose your Chatbot",
            info="Select the underlying AI model you wish to interact with."
        )

        start_btn = gr.Button("Start Chatting 🚀", variant="primary")

    # --- SCREEN 2: Chat Interface ---
    with gr.Column(visible=False) as chat_screen:
        # Header displaying which bot is active, and a back button
        with gr.Row():
            back_btn = gr.Button("⬅️ Change Bot", size="sm")
            active_bot_display = gr.Markdown("### 🤖 Loading...")

        chatbot = gr.Chatbot(height=400, type='messages')
        msg_input = gr.Textbox(placeholder="Type your message here...", label="Your Message")

        with gr.Row():
            clear_btn = gr.ClearButton([msg_input, chatbot], value="🗑️ Clear Chat")

    # --------------------------
    # EVENT LISTENERS
    # --------------------------

    # 1. When "Start Chatting" is clicked:
    # Hide welcome screen, show chat screen, update the title to reflect chosen bot
    def open_chat(choice):
        return (
            gr.update(visible=False),              # Hide welcome_screen
            gr.update(visible=True),               # Show chat_screen
            f"### 🤖 Currently chatting with: **{choice}**" # Update active bot display
        )

    start_btn.click(
        fn=open_chat,
        inputs=[bot_selection],
        outputs=[welcome_screen, chat_screen, active_bot_display]
    )

    # 2. When user submits a message:
    msg_input.submit(
        fn=predict,
        inputs=[msg_input, chatbot, bot_selection],
        outputs=[msg_input, chatbot]
    )

    # 3. When "Change Bot" is clicked:
    # Hide chat screen, show welcome screen, and reset the chat history
    def go_back():
        return (
            gr.update(visible=True),  # Show welcome_screen
            gr.update(visible=False), # Hide chat_screen
            []                        # Clear chat history
        )

    back_btn.click(
        fn=go_back,
        inputs=None,
        outputs=[welcome_screen, chat_screen, chatbot]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)