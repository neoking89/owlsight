class Styling:
    main_page = """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton button {
            width: 100%;
            border-radius: 4px;
            padding: 0.5rem;
        }
        .stTextInput div[data-baseweb="input"] {
            border-radius: 4px;
        }
        .stSelectbox div[data-baseweb="select"] {
            border-radius: 4px;
        }
        .stDownloadButton button {
            width: auto;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        /* Chat interface styling */
        .chat-message {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .chat-message.user {
            background-color: #f0f7ff;
            border-left: 5px solid #4c8bf5;
        }
        .chat-message.assistant {
            background-color: #f8fdf8;
            border-left: 5px solid #34a853;
        }
        .chat-message.system {
            background-color: #fffde7;
            border-left: 5px solid #fbbc05;
            opacity: 0.85;
        }
        .message-content {
            display: flex;
            margin-top: 0.5rem;
        }
        .message-avatar {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            margin-right: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
        .user-avatar {
            background-color: #e3f2fd;
            color: #1976d2;
        }
        .assistant-avatar {
            background-color: #e8f5e9;
            color: #2e7d32;
        }
        .message-text {
            flex-grow: 1;
        }
        .message-citation {
            background-color: #f5f5f5;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 0.85em;
            color: #666;
            cursor: pointer;
            display: inline-block;
            margin: 2px;
        }
        .context-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 5px;
            background-color: #e3f2fd;
            color: #1976d2;
        }
        .source-info {
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
            margin-top: 10px;
            border-left: 3px solid #4c8bf5;
        }
        .upload-container {
            border: 2px dashed #ccc;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            background-color: #fafafa;
        }
        .upload-container:hover {
            border-color: #4c8bf5;
            background-color: #f5f9ff;
        }
        .document-card {
            background-color: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #4c8bf5;
        }
        .document-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .document-info {
            color: #666;
            font-size: 0.9em;
        }
        .document-preview {
            margin-top: 10px;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.9em;
            max-height: 100px;
            overflow-y: auto;
        }
        </style>
    """
    