from langchain.embeddings.cohere import CohereEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
import os
from langchain.chat_models import ChatOpenAI
import telebot;
import pinecone
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Chroma
from tqdm.notebook import tqdm
import logging
import tg_logger
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

def main():
    os.environ["OPENAI_API_KEY"] = "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"
    os.environ["COHERE_API_KEY"] = "TTswjcHtySkELt8HyMaQNDcHhNIamyPaFLB9Gc8V"

    PINECONE_API_KEY = "d1f2e879-e4b6-44c2-9dc7-f7545ebb6b86"
    PINECONE_ENV = "asia-southeast1-gcp-free"

    bot = telebot.TeleBot('6315739175:AAFEQ4xwyjtEx4GT0zsyCrrFXjhuVlLjUfo')

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Telegram data
    token = "6144373052:AAEeyF5L56fMA9vOdVeZs7AHMFhresKhTeQ"
    users = [201621438]

    # Base logger
    logger = logging.getLogger('foo')
    logger.setLevel(logging.INFO)

    # Logging bridge setup
    tg_logger.setup(logger, token=token, users=users)

    """    
    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
        )

    index_name = "luz"
        
    print("here 1")
    
    urls = []

    with open("links.txt", 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line[:-1]
            urls.append(line)


    print(urls)

    loader = UnstructuredURLLoader(urls=urls)
    documents = loader.load()
    
        
    #split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

    print("here 2")

    

    """

    loader = TextLoader("./chunks.txt")
    documents = loader.load()

    llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0)

    #split into chunks
    text_splitter = CharacterTextSplitter(
            separator="\n<CHUNK>\n",
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )
    
    texts = text_splitter.split_documents(documents)

    llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=0)

    # Define the embeddings model
    embeddings = CohereEmbeddings(model = "multilingual-22-12")

    # For the new embeddings
    #docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)

    # For the existing embeddings
    # docsearch = Pinecone.from_existing_index(index_name, embeddings)

    docsearch = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    docsearch.persist()

    @bot.message_handler(content_types=["text"])
    def handle_text(message):

        if message.text:

            docs = docsearch.similarity_search(message.text)

            qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever = docsearch.as_retriever(), memory=memory)

            chat_history = [("", "")]

            result = qa({"question": message.text})

            sending_msg = "\n" + str(result["answer"]) + "\n"

            log_msg = sending_msg

            if len(log_msg) > 4096:
                for x in range(0, len(log_msg), 4096):
                    bot.send_message(message.chat.id, log_msg[x:x+4096])
            else:
                bot.send_message(message.chat.id, log_msg)

            logger.info(f'</code>@{message.from_user.username}<code> ({message.chat.id}) used echo:\n\n{message.text} \n\nAnswer: {log_msg}')
    
    bot.polling(none_stop=True, interval=0)

if __name__ == '__main__':
    main()