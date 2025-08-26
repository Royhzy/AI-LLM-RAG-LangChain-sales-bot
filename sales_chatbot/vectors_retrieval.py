import os.path

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings     #！！！魔塔社区embedding
from sympy.physics.units import temperature
from langchain_openai import AzureChatOpenAI

os.environ["OPENAI_API_KEY"] = "50a857aec1164241a3411b5e38e99982"  # 你的 API Key
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://genai-jp.openai.azure.com"  # 你的 Azure 实例域名
os.environ["OPENAI_API_TYPE"] = "azure"  # 指定使用 Azure 服务
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"  # API 版本


DB_DIR = 'faiss_db/'
EMBEDDINGS = ModelScopeEmbeddings(model_id='iic/nlp_gte_sentence-embedding_chinese-base')

def save_vectors_db():
    """构建向量数据，并保存到磁盘"""
    if os.path.exists(DB_DIR):
        print('向量数据库已经构建，直接读取就OK！')
    else:
        with open('sales_talks.txt', encoding='utf-8') as f:
            contents = f.read()
        #把文本内容，切割成一个个doc
        text_splitter = CharacterTextSplitter(
            separator=r'\d+\.',
            is_separator_regex=True,
            chunk_size=100,
            chunk_overlap=0,
            length_function=len
        )
        docs = text_splitter.create_documents([contents])
        print(len(docs))
        db = FAISS.from_documents(docs,EMBEDDINGS)
        db.save_local(DB_DIR)

        #result = db.similarity_search('小区里面有绿化吗？')    # 检验向量数据库是否加载（没接openai）
        #print(result)

def init_chain():
    """最终得到一个chain"""
    # 第一步：加载向量数据库
    db = FAISS.load_local(DB_DIR,EMBEDDINGS,allow_dangerous_deserialization=True)  #允许文件加载数据库会有风险，设置为true可以没有警告
    # 第二步：创建一个提示模版
    system_prompt = """
    你是一个专业、负责的问答助手。请确保你的回答符合公序良俗，不包含任何敏感、违法或不适当的内容。
    使用以下检索到的上下文来回答问题。如果你不知道答案，就说："我不知道呢，请您问问人工!"。最多使用三句话，保持答案简洁。\n
    {context}
    """
    prompt_template = ChatPromptTemplate.from_messages(  # 提问和回答的 历史记录  模板
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    # 第三步： 创建一个chain
    # 创建一个搜索器:similarity_score_threshold:根据相似度的分数来返回结果，分值>=0.7
    retriever = db.as_retriever(search_type='similarity_score_threshold', search_kwargs={"score_threshold": 0.7})

    model = AzureChatOpenAI(
        deployment_name="ln-gpt40",  # 你的模型部署名称
        temperature=0.2
    )
    chain1 = create_stuff_documents_chain(model, prompt_template)  #将检索到的结果（多个docs) 输入到提示模版中
    chain = create_retrieval_chain(retriever, chain1)
    return chain


if __name__ == '__main__':
    save_vectors_db()
    chain = init_chain()
    res = chain.invoke({'input':'房子附近有什么设施？'})
    print(res)   #大模型会得到相似度大于0.7的答案，比如会有x个答案，然后大模型会把x个答案最终汇总成answer
    print(res['answer'])