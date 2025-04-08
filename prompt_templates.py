memory_prompt_template = """<s>[INST] You are a helpful, knowledgeable AI assistant. Answer the user's question based on our conversation history. Be clear, accurate, and helpful. [/INST]

Previous conversation: {history}
Human: {human_input}
AI: Respond comprehensively and accurately in English while adhering to professional standards:

"""

pdf_chat_prompt = """<s>[INST] Answer user inquiries based on the context and conversation history. Focus on providing accurate and comprehensive answers that meet user needs and consider the available information. Please pay attention to details and provide a professional response that adheres to required standards. **All answers must be in English.** [/INST]
Conversation history: {history}
Context: {context}
Question: {human_input}
Answer: Please provide a comprehensive, accurate, and professional response in English:
"""
