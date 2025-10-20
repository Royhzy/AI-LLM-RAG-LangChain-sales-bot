import os
import time
import json
from typing import List, Dict, Any, Tuple
from langchain.chains import RetrievalQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ModelScopeEmbeddings
from langchain.docstore.document import Document

# 设置环境变量
os.environ["OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://genai-jp.openai.azure.com"
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"

# 配置常量
DB_DIR = 'faiss_db/'
METADATA_PATH = os.path.join(DB_DIR, 'metadata.json')
EMBEDDINGS = ModelScopeEmbeddings(model_id='iic/nlp_gte_sentence-embedding_chinese-base')
CHUNK_SIZE = 300  # 增大分块大小
CHUNK_OVERLAP = 50  # 添加重叠以保持上下文连贯


def save_vectors_db():
    """构建向量数据，并保存到磁盘"""
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print('向量数据库已经存在，直接读取即可！')
        return

    # 创建数据库目录
    os.makedirs(DB_DIR, exist_ok=True)

    try:
        with open('sales_talks.txt', encoding='utf-8') as f:
            contents = f.read()

        # 改进的文本分割策略
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

        # 创建带有元数据的文档
        docs = text_splitter.create_documents([contents])
        docs_with_metadata = add_metadata_to_docs(docs)

        print(f"生成了 {len(docs_with_metadata)} 个文档块")

        # 创建并保存向量数据库
        start_time = time.time()
        db = FAISS.from_documents(docs_with_metadata, EMBEDDINGS)
        db.save_local(DB_DIR)
        save_metadata(docs_with_metadata)

        elapsed_time = time.time() - start_time
        print(f"向量数据库构建完成，耗时: {elapsed_time:.2f} 秒")

    except Exception as e:
        print(f"构建向量数据库时出错: {e}")


def add_metadata_to_docs(docs: List[Document]) -> List[Document]:
    """为文档添加元数据以增强检索能力"""
    docs_with_metadata = []

    for i, doc in enumerate(docs):
        # 提取文档的主题和类型
        metadata = {
            "chunk_id": i,
            "topic": extract_topic(doc.page_content),
            "doc_type": classify_document(doc.page_content)
        }

        docs_with_metadata.append(Document(
            page_content=doc.page_content,
            metadata=metadata
        ))

    return docs_with_metadata


def extract_topic(content: str) -> str:
    """简单的主题提取逻辑，可根据实际情况改进"""
    # 这里可以使用更复杂的NLP技术，如关键词提取或主题模型
    keywords = ["绿化", "设施", "交通", "价格", "户型", "面积"]
    for keyword in keywords:
        if keyword in content:
            return keyword
    return "其他"


def classify_document(content: str) -> str:
    """简单的文档分类逻辑"""
    if "小区" in content or "附近" in content:
        return "周边环境"
    elif "户型" in content or "面积" in content:
        return "房屋属性"
    elif "价格" in content or "首付" in content:
        return "价格信息"
    return "其他"


def save_metadata(docs: List[Document]):
    """保存文档元数据到文件"""
    metadata = [{
        "chunk_id": doc.metadata.get("chunk_id"),
        "topic": doc.metadata.get("topic"),
        "doc_type": doc.metadata.get("doc_type"),
        "content_length": len(doc.page_content)
    } for doc in docs]

    with open(METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def init_chain():
    """初始化检索链，优化检索策略"""
    try:
        # 加载向量数据库
        db = FAISS.load_local(DB_DIR, EMBEDDINGS, allow_dangerous_deserialization=True)

        # 优化的检索器配置
        retriever = db.as_retriever(
            search_type="mmr",  # 使用最大边际相关性检索
            search_kwargs={
                "k": 5,  # 返回更多文档用于MMR重排序
                "fetch_k": 20,  # 先获取更多文档进行筛选
                "lambda_mult": 0.7  # 平衡相关性和多样性
            }
        )

        # 创建提示模板
        system_prompt = """
        你是一个专业、负责的房产问答助手。请根据以下检索到的上下文简洁回答问题。
        如果你不知道答案，就说："我不知道呢，请您问问人工!"。
        请保持回答简洁，最多使用三句话。

        检索到的相关信息:
        {context}
        """

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # 初始化模型
        model = AzureChatOpenAI(
            deployment_name="ln-gpt40",
            temperature=0.2,
            max_tokens=200
        )

        # 创建文档组合链和检索链
        combine_documents_chain = create_stuff_documents_chain(model, prompt_template)
        chain = create_retrieval_chain(retriever, combine_documents_chain)

        return chain

    except Exception as e:
        print(f"初始化检索链时出错: {e}")
        return None


def evaluate_performance():
    """评估检索性能"""
    # 加载向量数据库
    db = FAISS.load_local(DB_DIR, EMBEDDINGS, allow_dangerous_deserialization=True)

    # 准备测试问题
    test_questions = [
        "小区里面有绿化吗？",
        "房子附近有什么设施？",
        "这个户型有多大面积？",
        "周边交通方便吗？",
        "房价是多少？"
    ]

    # 测试原始相似度检索
    print("\n=== 测试原始相似度检索 ===")
    original_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    test_retriever_performance(original_retriever, test_questions)

    # 测试优化后的MMR检索
    print("\n=== 测试优化后的MMR检索 ===")
    optimized_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 20, "lambda_mult": 0.7}
    )
    test_retriever_performance(optimized_retriever, test_questions)


def test_retriever_performance(retriever, questions: List[str]):
    """测试检索器性能"""
    for question in questions:
        start_time = time.time()
        results = retriever.get_relevant_documents(question)
        elapsed_time = time.time() - start_time

        print(f"问题: {question}")
        print(f"检索耗时: {elapsed_time:.4f} 秒")
        print(f"返回文档数: {len(results)}")
        print("---")


if __name__ == '__main__':
    save_vectors_db()

    # 评估性能
    evaluate_performance()

    # 初始化链并测试
    chain = init_chain()
    if chain:
        test_questions = [
            "房子附近有什么设施？",
            "小区绿化如何？",
            "周边有学校吗？"
        ]

        for question in test_questions:
            res = chain.invoke({'input': question})
            print(f"\n问题: {question}")

            print(f"回答: {res['answer']}")
