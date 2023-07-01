from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import AsyncCallbackManager
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

def get_chain(stream_handler) -> ConversationChain:
    manager = AsyncCallbackManager([])
    stream_manager = AsyncCallbackManager([stream_handler])

    streaming_llm = OpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0
    )

    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template='''You are an expert Chinese language tutor helping 
        a beginner learn Chinese. You are completely fluent in both 
        English and Mandarin Chinese, but you should always respond 
        in Chinese unless the student asks for English clarification.

        Current conversation:
        {history}
        Student: {input}
        Tutor:'''
    )

    chain = ConversationChain(
        llm=streaming_llm,
        prompt=prompt,
        memory=ConversationBufferMemory(ai_prefix="Tutor", human_prefix="Student"),
        callback_manager=manager
    )

    return chain