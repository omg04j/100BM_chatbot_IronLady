import streamlit as st
from typing import List, Dict, Any
from datetime import datetime
import base64
from pathlib import Path
import sys
import os

# Add current directory to path if needed
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Minimal page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="100 BM Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Import after streamlit config to avoid issues
try:
    # Import the RAG system from utils module
    import utils
    # Get the class from the module
    ProfileAwareRAGSystem = getattr(utils, 'ProfileAwareRAGSystem', None)
    
    if ProfileAwareRAGSystem is None:
        st.error("❌ Could not find ProfileAwareRAGSystem in utils.py")
        st.info("Available classes in utils.py:")
        for name in dir(utils):
            obj = getattr(utils, name)
            if isinstance(obj, type) and not name.startswith('_'):
                st.info(f"  - {name}")
        st.stop()
        
except Exception as e:
    st.error(f"❌ Error importing from utils.py: {str(e)}")
    st.info("""
    **Troubleshooting Tips:**
    1. Check if utils.py has any self-imports (like `from utils import ...`)
    2. Make sure all dependencies are installed (langchain, openai, etc.)
    3. Check if there are any syntax errors in utils.py
    4. Make sure .env file exists with OPENAI_API_KEY
    """)
    st.stop()

# Function to encode logo image
def get_logo_base64():
    """Convert logo to base64 for embedding"""
    try:
        logo_path = Path("iron_lady_logo.png")
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
    except:
        pass
    return None

logo_b64 = get_logo_base64()

# Compact CSS for embedded widget with RED theme
st.markdown(f"""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* Compact chat widget styling */
    .stApp {{
        max-width: 600px;
        margin: 0 auto;
    }}
    
    .chat-header {{
        background: linear-gradient(135deg, #DC143C 0%, #8B0000 100%);
        padding: 1.5rem;
        border-radius: 10px 10px 0 0;
        color: white;
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    .chat-header-logo {{
        width: 50px;
        height: 50px;
        background-color: white;
        border-radius: 8px;
        padding: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    
    .chat-header-logo img {{
        width: 100%;
        height: 100%;
        object-fit: contain;
    }}
    
    .chat-header-text {{
        flex: 1;
    }}
    
    .chat-header-title {{
        font-size: 1.3rem;
        font-weight: bold;
        margin: 0;
    }}
    
    .chat-header-subtitle {{
        font-size: 0.9rem;
        opacity: 0.9;
        margin: 0;
    }}
    
    .chat-container {{
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        max-height: 500px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }}
    
    .chat-message {{
        padding: 0.75rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: fadeIn 0.3s;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .user-message {{
        background-color: #ffebee;
        margin-left: 2rem;
        text-align: right;
        border-left: 4px solid #DC143C;
    }}
    
    .assistant-message {{
        background-color: white;
        margin-right: 2rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }}
    
    .welcome-message {{
        background: linear-gradient(135deg, #ffffff 0%, #fff5f5 100%);
        border: 2px solid #DC143C;
        margin: 0;
    }}
    
    .message-label {{
        font-size: 0.75rem;
        font-weight: bold;
        margin-bottom: 0.25rem;
        color: #DC143C;
    }}
    
    .message-content {{
        color: #333;
        line-height: 1.5;
    }}
    
    .input-container {{
        background: white;
        padding: 1rem;
        border-radius: 0 0 10px 10px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }}
    
    .source-badge {{
        display: inline-block;
        background-color: #ffebee;
        color: #DC143C;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        margin: 0.25rem 0.25rem 0 0;
        border: 1px solid #DC143C;
    }}
    
    .powered-by {{
        text-align: center;
        font-size: 0.7rem;
        color: #999;
        margin-top: 0.5rem;
        padding: 0.5rem 0;
    }}
    
    /* Hide extra Streamlit padding */
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    
    /* Compact form styling for Input */
    .stTextInput > div > div > input {{
        border-radius: 10px; /* Changed to match image style */
        border: 2px solid #DC143C;
        height: 2.75rem; 
        padding: 0.5rem 1rem; 
        font-size: 1rem; 
        margin-bottom: 0.5rem; /* Add some space before the next row */
    }}
    
    .stTextInput > div > div > input:focus {{
        border-color: #8B0000;
        box-shadow: 0 0 0 0.2rem rgba(220, 20, 60, 0.25);
    }}
    
    /* Enhanced button styling for desired look */
    .stButton > button {{
        /* Changed to square-ish rounded corners like the image */
        border-radius: 10px; 
        background: white; /* Changed to white background like the image */
        color: #DC143C; /* Icon/Text color */
        border: 1px solid #ccc; /* Subtle border like the image */
        font-weight: bold;
        padding: 0.5rem 0.25rem; 
        font-size: 0.85rem; 
        height: 3.5rem; /* Increased height to be taller than input */
        line-height: 1; 
        transition: all 0.3s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05); /* Subtle shadow */
        margin-top: 0; 
        display: flex;
        flex-direction: column; /* Stack icon and text */
        align-items: center;
        justify-content: center;
        gap: 0.25rem; 
    }}
    
    .stButton > button:hover {{
        background: #f8f8f8; /* Slight hover change */
        border-color: #DC143C; /* Red border on hover */
        box-shadow: 0 2px 4px rgba(220, 20, 60, 0.2);
        transform: translateY(-1px);
    }}
    
    /* Specific styling for the 'Clear Memory' button */
    #clear_memory_btn_second_row > button {{ 
        /* Match the look of the top buttons */
        background: white;
        border: 1px solid #ccc;
        color: #DC143C;
        /* Force full width in its column */
        width: 100%;
        height: 3.5rem;
    }}

    /* Target specific icons to remove default Streamlit button color tint */
    .stButton > button > div:first-child {{
        color: #DC143C !important;
    }}
    
    /* Ensure no extra padding in columns for tight alignment */
    .stTextInput > div {{
        padding-top: 0;
        padding-bottom: 0;
    }}
    .stForm > div > div {{
        padding-top: 0;
        padding-bottom: 0;
        margin-bottom: 0;
    }}
    .stForm > div {{
        margin-bottom: 0;
    }}
    
    /* Custom styling for the Clear Memory button icon */
    .memory-icon {{
        font-size: 1.5rem;
        line-height: 1;
        color: #E3526E; /* Pink/Red brain color like the image */
    }}
    
</style>
""", unsafe_allow_html=True)


# ============================================================================
# INITIALIZATION (Unchanged)
# ============================================================================

@st.cache_resource
def initialize_system():
    """Initialize RAG system (cached) - NO conversation memory stored here"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        
        # Initialize the ProfileAwareRAGSystem (stateless - no memory)
        rag_system = ProfileAwareRAGSystem(vector_store_path="./vector_store")
        return rag_system, None
        
    except FileNotFoundError as e:
        return None, f"Vector store not found: {str(e)}"
    except Exception as e:
        return None, f"Initialization error: {str(e)}"


def initialize_chat():
    """Initialize chat session state with SESSION-SPECIFIC memory"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = False
    
    # ✅ CRITICAL FIX: Session-specific conversation memory (NOT shared between users)
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []


# ============================================================================
# SUGGESTED QUESTIONS (Unchanged)
# ============================================================================

SUGGESTED_QUESTIONS = [
    "💊 I am a doctor, how can I apply 4T principles?",
    "👥 As an HR leader, what is the capability matrix?",
    "📋 How to create a board member persona?",
    "💻 As a tech executive, what is the success story framework?",
    "🗓️ What is my batch schedule?"
]


def render_suggestions():
    """Render suggested questions section"""
    if not st.session_state.messages:  # Only show when chat is empty
        st.markdown("""
        <div class="suggestions-section">
            <div class="suggestions-title">💡 Try these questions:</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Create clickable buttons for each suggestion
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            if st.button(question, key=f"suggest_{i}", use_container_width=True):
                # Set the question as if user typed it
                st.session_state.pending_question = question.split(" ", 1)[1]  # Remove emoji
                st.rerun()


# ============================================================================
# CHAT WIDGET WITH STREAMING AND SESSION-BASED MEMORY
# ============================================================================

def render_chat_widget():
    """Render compact chat widget with streaming support and session-based memory"""
    
    # Initialize
    initialize_chat()
    rag_system, error = initialize_system()
    
    if error or not rag_system:
        st.error(f"⚠️ System not available: {error}")
        st.info("""
        **Setup Instructions:**
        1. Create a `.env` file in your project directory
        2. Add: `OPENAI_API_KEY=your-key-here`
        3. Make sure `vector_store` folder exists
        4. Run: `python vector_store.py` to create the vector store if needed
        """)
        return
    
    # Header with Iron Lady logo
    logo_html = ""
    if logo_b64:
        logo_html = f'<div class="chat-header-logo"><img src="data:image/png;base64,{logo_b64}" alt="Iron Lady Logo"></div>'
    
    st.markdown(f"""
    <div class="chat-header">
        {logo_html}
        <div class="chat-header-text">
            <div class="chat-header-title">100BM AI Assistant</div>
            <div class="chat-header-subtitle">Iron Lady Leadership Program</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ✅ Memory status indicator (if there's conversation history)
    if len(st.session_state.conversation_history) > 0:
        st.markdown(f"""
        <div class="memory-status">
            🧠 <b>Memory Active:</b> Remembering {len(st.session_state.conversation_history)} conversation(s)
        </div>
        """, unsafe_allow_html=True)
    
    # Suggested questions section (only when chat is empty)
    render_suggestions()
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        if not st.session_state.messages:
            # ✅ Welcome message with memory indicator
            st.markdown(f"""
            <div class="chat-message assistant-message welcome-message">
                <div class="message-label">🤖 Assistant</div>
                <div class="message-content">
                    👋 <b>Welcome to 100BM AI Chat!</b>
                    <br><br>
                    I'm your intelligent assistant for the <b>Iron Lady Leadership Program</b>. 
                    Tell me your profession for personalized advice!
                    <br><br>
                    <span class="feature-badge">🎯 Profile-Aware</span>
                    <span class="feature-badge">📚 100BM Content</span>
                    <span class="feature-badge">⚡ Real-Time</span>
                    <span class="memory-badge">🧠 Memory Enabled</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-label">👤 You</div>
                    <div class="message-content">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div class="message-label">🤖 Assistant</div>
                    <div class="message-content">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Check if there's a pending question from suggestions
    if hasattr(st.session_state, 'pending_question'):
        user_input = st.session_state.pending_question
        delattr(st.session_state, 'pending_question')
        process_message(user_input, rag_system)
        st.rerun()
    
    # Input form with enhanced buttons
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    
    # ------------------------------------------------------------------
    # ✅ ROW 1: Input, Send, Clear Chat (Inside form)
    # ------------------------------------------------------------------
    with st.form(key="chat_form", clear_on_submit=True):
        # Columns to hold Input, Send, and Clear Chat
        # Note: We reserve space for the Clear Memory button by using a wide column for Input.
        col1, col2, col3 = st.columns([6, 1.5, 1.5])
        
        with col1:
            # Input Field
            user_input = st.text_input(
                "Message",
                placeholder="Ask a question...",
                label_visibility="collapsed",
                key="user_input_field"
            )
        
        with col2:
            # Send Button - Use custom HTML for the logo/text structure
            send_button = st.form_submit_button(
                "↗️ Send", 
                use_container_width=True,
                key="send_btn_form"
            )
        
        with col3:
            # Clear Chat Button - Use custom HTML for the logo/text structure
            clear_chat_button = st.form_submit_button(
                "🗑️ Clear Chat", 
                use_container_width=True,
                key="clear_chat_btn_form"
            )
            
        # Process input/actions within the form
        if send_button and user_input:
            process_message(user_input, rag_system)
            st.rerun()
        
        if clear_chat_button:
            st.session_state.messages = []
            st.rerun()

    # ------------------------------------------------------------------
    # ✅ ROW 2: Clear Memory (Outside form for simpler layout control)
    # ------------------------------------------------------------------
    # Align the empty space with the Input field column, and the button with 
    # the Send/Clear Chat columns combined.
    col_mem_empty, col_mem_btn = st.columns([6, 3]) 

    with col_mem_btn:
        # Clear Memory Button - Use markdown and custom class for the logo/text
        clear_memory_clicked = st.button(
            "🧠 Clear Memory", 
            key="clear_memory_btn_second_row", 
            use_container_width=True
        )

    # Handle Clear Memory action
    if clear_memory_clicked:
        st.session_state.conversation_history = []
        st.session_state.messages = [] # Clear chat history too for clean start
        st.rerun()


    # Footer
    st.markdown(
        '<div class="powered-by">Powered by OpenAI & LangChain</div>',
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)


def process_message(user_input: str, rag_system):
    """Process and display a message with streaming and session-based memory"""
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message immediately
    st.markdown(f"""
    <div class="chat-message user-message">
        <div class="message-label">👤 You</div>
        <div class="message-content">{user_input}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get AI response with STREAMING
    st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
    st.markdown('<div class="message-label">🤖 Assistant</div>', unsafe_allow_html=True)
    
    # Streaming response container
    response_container = st.empty()
    full_response = ""
    
    try:
        # ✅ Pass session-specific conversation history to RAG system
        for chunk in rag_system.ask_stream(user_input, st.session_state.conversation_history):
            # Check for history update marker
            if chunk.startswith("__HISTORY_UPDATE__:"):
                # History was updated, retrieve from session
                continue
            
            full_response += chunk
            # Update with blinking cursor while streaming
            response_container.markdown(
                f'<div class="message-content typing-cursor">{full_response}</div>',
                unsafe_allow_html=True
            )
        
        # Final display without cursor
        response_container.markdown(
            f'<div class="message-content">{full_response}</div>',
            unsafe_allow_html=True
        )
        
        # ✅ Update session conversation history
        st.session_state.conversation_history.append({
            'question': user_input,
            'answer': full_response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10 conversations in memory
        if len(st.session_state.conversation_history) > 10:
            st.session_state.conversation_history = st.session_state.conversation_history[-10:]
        
        # Add assistant message to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        error_msg = f"⚠️ Sorry, I encountered an error: {str(e)}"
        
        response_container.markdown(
            f'<div class="message-content">{error_msg}</div>',
            unsafe_allow_html=True
        )
        
        # Show detailed error in expander
        with st.expander("🔍 Error Details"):
            st.code(error_details)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": error_msg
        })
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main application"""
    render_chat_widget()


if __name__ == "__main__":
    main()
