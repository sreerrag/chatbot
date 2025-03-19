import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load the BlenderBot model and tokenizer
model_name = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_response(question):
    try:
        # Tokenize the input question
        inputs = tokenizer([question], return_tensors="pt", padding=True)
        # Generate a response
        reply_ids = model.generate(**inputs, max_length=60, pad_token_id=tokenizer.pad_token_id)
        # Decode the response
        response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

# Initialize Dash app
app = dash.Dash(__name__)

# Define custom CSS for a modern look
app.css.append_css({
    "external_url": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
})

# Layout with improved UI
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("ChatBot Pro", style={
            'textAlign': 'center',
            'color': '#ffffff',
            'padding': '15px',
            'backgroundColor': '#2c3e50',
            'margin': '0',
            'fontFamily': 'Arial, sans-serif'
        }),
        html.Button("Clear Chat", id="clear-button", style={
            'position': 'absolute',
            'right': '20px',
            'top': '15px',
            'backgroundColor': '#e74c3c',
            'color': 'white',
            'border': 'none',
            'padding': '8px 15px',
            'borderRadius': '5px',
            'cursor': 'pointer',
            'fontFamily': 'Arial, sans-serif'
        })
    ]),

    # Chat area
    html.Div(id='chatbot-output', style={
        'flex': '1',
        'padding': '20px',
        'maxWidth': '800px',
        'margin': '0 auto',
        'height': '70vh',
        'overflowY': 'auto',
        'backgroundColor': '#f5f6fa',
        'borderRadius': '10px',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)',
        'display': 'flex',
        'flexDirection': 'column'
    }),

    # Input area
    html.Div([
        dcc.Textarea(
            id='user-input',
            placeholder='Type your message...',
            style={
                'width': 'calc(100% - 120px)',
                'height': '50px',
                'margin': '10px 10px 10px 20px',
                'padding': '10px',
                'border': '1px solid #dcdcdc',
                'borderRadius': '5px',
                'resize': 'none',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '16px'
            }
        ),
        html.Button(
            'Send',
            id='submit-button',
            n_clicks=0,
            style={
                'width': '80px',
                'height': '50px',
                'margin': '10px 20px 10px 0',
                'backgroundColor': '#3498db',
                'color': 'white',
                'border': 'none',
                'borderRadius': '5px',
                'cursor': 'pointer',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '16px'
            }
        )
    ], style={
        'display': 'flex',
        'maxWidth': '800px',
        'margin': '0 auto',
        'backgroundColor': '#ffffff',
        'borderRadius': '10px',
        'boxShadow': '0 2px 10px rgba(0,0,0,0.1)'
    })
], style={
    'height': '100vh',
    'display': 'flex',
    'flexDirection': 'column',
    'backgroundColor': '#ecf0f1',
    'margin': '0',
    'padding': '0'
})

# Store chat history in a list
chat_history = []

@app.callback(
    [Output('chatbot-output', 'children'),
     Output('user-input', 'value')],
    [Input('submit-button', 'n_clicks'),
     Input('clear-button', 'n_clicks'),
     Input('user-input', 'n_submit')],
    [State('user-input', 'value')]
)
def update_output(submit_clicks, clear_clicks, n_submit, user_input):
    global chat_history
    ctx = dash.callback_context

    # Handle clear chat
    if ctx.triggered and 'clear-button' in ctx.triggered[0]['prop_id']:
        chat_history = []
        return html.Div("Start chatting!", style={
            'textAlign': 'center',
            'color': '#7f8c8d',
            'marginTop': '20px'
        }), ""

    # Handle user input
    if (submit_clicks > 0 or n_submit is not None) and user_input and user_input != 'Type your message...':
        # Get response
        response = get_response(user_input)
        # Add to chat history
        chat_history.append((user_input, response))
        # Display chat history with chat bubbles
        chat_elements = []
        for user_msg, bot_msg in chat_history:
            chat_elements.extend([
                # User message (right-aligned, blue bubble)
                html.Div(
                    f"{user_msg}",
                    style={
                        'alignSelf': 'flex-end',
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'padding': '10px 15px',
                        'borderRadius': '15px',
                        'margin': '5px 10px',
                        'maxWidth': '70%',
                        'wordWrap': 'break-word',
                        'fontFamily': 'Arial, sans-serif'
                    }
                ),
                # Bot message (left-aligned, gray bubble)
                html.Div(
                    f"{bot_msg}",
                    style={
                        'alignSelf': 'flex-start',
                        'backgroundColor': '#dcdcdc',
                        'color': '#2c3e50',
                        'padding': '10px 15px',
                        'borderRadius': '15px',
                        'margin': '5px 10px',
                        'maxWidth': '70%',
                        'wordWrap': 'break-word',
                        'fontFamily': 'Arial, sans-serif'
                    }
                )
            ])
        return chat_elements, ""  # Clear the input field after submission

    return html.Div("Start chatting!", style={
        'textAlign': 'center',
        'color': '#7f8c8d',
        'marginTop': '20px'
    }), user_input

if __name__ == '__main__':
    app.run(debug=True)