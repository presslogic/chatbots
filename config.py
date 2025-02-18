class Config:
    PAGE_TITLE = "Deepseek Chatbot"

    OLLAMA_MODELS = ('deepseek-r1:14b', 'deepseek-r1:8b', 'deepseek-r1:7b')

    SYSTEM_PROMPT = f"""You are a helpful chatbot that has access to the following 
                    open-source models {OLLAMA_MODELS}.
                    You can can answer questions for users on any topic."""