from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import os
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from basicUtils import basicUtils
import json
import ast
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import matplotlib.pyplot as plt
from langchain.chains import LLMChain
from chatPdfApi import chatPdfApi



def langchaincase():

    os.environ["OPENAI_API_KEY"] = "***************************"
    elasticsearch_url = "http://localhost:9200"

    # split pdf
    pdfPathList = basicUtils.split_pdf('Textbook of pathology1.pdf', 10)

    source_id_list = []

    for temp_pdf_path in pdfPathList:
        # Call the chatPdf API and save the sourceId using the temporary file
        source_id = chatPdfApi.upload_pdf(temp_pdf_path)
        source_id_list.append(source_id)
        os.remove(temp_pdf_path)

    # print JSON
    json_data = json.dumps(source_id_list)
    print(json_data)

    # !!!FOR TEST!!!
    json_data = '["src_3vKvaAWIbI4o1rfR3g81h", "src_Gy2LBD1KRiCjMmakZrOCy", "src_NgHHcj6SLitpO5MxHJy1L", "src_pXWmy0bNuHZbfE1RL4xus", "src_pUFDQVq0ZdgqqsERxkpRT", "src_mwttJX02zcGFjWGudt4f5", "src_hWf7Edqpy6DbdkNjAniNN", "src_Z47A7wWomHTcfnNjNXuDD", "src_FUzerFeGE5sbGTgh2bR79", "src_qURUzRuziJDz33ofxTeHx", "src_eyeB0iU9VjhDCT5GZTqn2", "src_xj7aCZRe5guNyYbLqffWp", "src_FqUVeOMlMfXyc9ZnQdRoF", "src_XTAgBkTnkmxHttpX4KGPc", "src_M77iWBJ0LGZJLPsCOOuSy", "src_PI2b9gXwg73cW9hWF1wzl", "src_PbN8lvya29GZecYIkH9Rb", "src_pCOAoKN1kPTTyMKfioRKN", "src_5xGfB81UBvvowI6BL481F", "src_PgTE7sYzvw9lQV02GCDlJ", "src_1AirGlo0RlvLIgaHr9YSO", "src_EqwJMVmqp7Nyaf1iujHy1", "src_eN20aGiRGqlywUUmetG5y", "src_sBY3yfTn4tLB6kk85C9OL", "src_IDzLBXE0M5gBRWaO7mLOh", "src_XDQ96Tpl5zhfI7RZYZguL", "src_RdkkT0r32jRqBo8dWQw5U", "src_Ng6gyNRycycMLF5QTocpL", "src_s14kEGDfhbXxW2vD59AC5", "src_E5iTVDGln2QUpoDrg80nf", "src_z4Sr8NWAx72Pk3K6NEKgE", "src_gaJUp4ER20Hq1Sg3cDzzw", "src_tN91Nh7pi4c5Knur8lnRb", "src_kaJYe2fHXfUeKkI9T0MTN", "src_h9zpG4RQTKNGAcnFhGkcK", "src_FJHH2J4O7HskGXjWIC04H", "src_evpHe0V8ewUYgwBh049LQ", "src_iYvcrZHXFPD3hOUj3zrdD", "src_QsFjWsSTx5w6zRQLBWaqO", "src_Tdj57YiKSbyqZpezjI6wI", "src_RGo4VCI4DSZQUFaKzwK1A", "src_LSRVzhWSYx6tkPTQpWRW5", "src_lls5bC7fAWxmzo7uQQ6RH", "src_2kI0biGAerEmVet0CmHsM", "src_hf4SQFjRYQAuS7566qFdv", "src_Ey4vZdlZvfflwxIthfnIT", "src_wtmCNM9LXJAGuosAzFfHR", "src_95rT33acjcPtmA9RCQO7L", "src_P2xOZN5ex08f1sC4vI45s", "src_cHZ7p8KOWQkzxjf3JmmsZ", "src_vwfHGH1shunnO8K2Yxlgc", "src_6PZSyqsRLJIuKMFCiLyIt", "src_3Vya3yjS9vzziNDgI04MV", "src_YBetk9ocq7KdJVH9R3WN8", "src_StRAHyfvokdCVeYhsKRfp", "src_3u9TtUdKbfbkO1BS7Ej8P", "src_nprfq0HK3ODycjR59Roa3", "src_cvABOQKpHkshrOIv2Tyiv", "src_QqyLkzDxkfpEFkPp0jsyN", "src_cdEBWHNPtsEEeySlPuMnS", "src_4kYmsAaqDTDS12aeBEmHM", "src_WjxbRrLFiOBeTr0MRRFCj", "src_oiV8JJONdsQsgMRE3msTZ", "src_5ol2IqQU5u7mFPWFezV5j", "src_mU4B1SDswM7TMjr4GxoTg", "src_yOs7ucdMVcurTrd7heI2Y", "src_SmPqNj7N4M30M5E0ccP3v", "src_ixFj2w7ZMPpAsB26rxEHV", "src_2S74VPB2ml0MXx6tpvGsc", "src_I1XRe8z4Stx9Hg5j6dScB", "src_PoOZ8GA17faXmb0kqnzgi", "src_3P4LqJ1EbI60A93RLIvo9", "src_Bwxz0eGlnI3XVnyccyV3q", "src_ZG52O6eyNmGCLw6yhL7ya", "src_gAtSzRy7pZV9gPfUDOyvQ", "src_LdLH56isQXpRyeabWJ8yg", "src_iGWThAq03Bem6gmZBYb69", "src_vdCCXUk290X5ZxiUHANhi", "src_XY5Koiilwl3GApO0GO7kV", "src_ETnNabLSXPGQDnrw5qO7V", "src_7l255zDKWUyevReQjYQjN", "src_JlU7dZXvp10hdLgnwEvFJ", "src_oYzkKVnzjdIcZhVM2Zkq6", "src_ATVvIqhZ5067iFoU3Efmm", "src_iIJ3U2NigH0lN8O9Duir8", "src_WjN7FBeLQrHvS8WHUk77q", "src_DXoFySZbUMTTMVgu1627p", "src_i4Ka9jCT5fmlyGc86OLo1", "src_n9J2ACCZ1cmzt9seO7vWo", "src_0A0Y28SSTnUOdfPiqIeqE", "src_TxTfStA0HirTkeQR9YmJf", "src_sveTYHAkkprczeYwKre5e", "src_D1iWABK0HJlQXUTmnyixH", "src_6eryDKDUxeySLqPYIHRM1", "src_MhS52KfGUxqYlv1cFMqUH", "src_lmeGJKmsMtqwF6QFZWt4i"]'
    source_id_list = json.loads(json_data)
    # !!!FOR TEST!!!


    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=400
    )

    # set the index
    elasticsearch_index = "test4000"

    # Since the direct storage of the entire pdf exceeds the length limit of openAI's embedding, so split traversal
    for pdf_number in range(1, 11):
        pdf_filename = f"Textbook of pathology{pdf_number}.pdf"
        loader = PyPDFLoader(pdf_filename)
        documents = loader.load()

        # Splitting Documents and Creating Elasticsearch Stores
        docs = text_splitter.split_documents(documents)
        ElasticsearchStore.from_documents(
            docs,
            embeddings,
            es_url=elasticsearch_url,
            index_name=elasticsearch_index,
            es_user="elastic",
            es_password="=rYu3-fhBhEe_O_NYkpE",
            strategy=ElasticsearchStore.ApproxRetrievalStrategy()
        )

    # Re-create the elastic_vector_search object used to generate retriever
    elastic_vector_search = ElasticsearchStore(
        es_url=elasticsearch_url,
        index_name=elasticsearch_index,
        embedding=embeddings,
        es_user="elastic",
        es_password="=rYu3-fhBhEe_O_NYkpE",
        distance_strategy="COSINE",
        strategy=ElasticsearchStore.ApproxRetrievalStrategy(),
    )

    # retriever = elastic_vector_search.as_retriever()
    llm = ChatOpenAI(temperature=0.1, model='gpt-3.5-turbo-16k')

    # Prompt
    template = """Please answer the final question in the context below. If you don't know the answer,
                say you don't know; don't try to make up an answer. Answer the question in the form of a 
                definition.Keep your answer as short as possible.
                        
                        {context}
                        
                    Question: {question}"""


    similarity_list = list()
    rouge_score_list = list()
    meteor_score_list = list()

    # Test without running the api that generates the problem, directly read the formatted qa.txt
    # qa_dict = chatPdfApi.generate_questions(source_id_list)

    with open('qa_dict.txt', 'r') as file:
        file_contents = file.read()

    # Parses the contents of a file into dictionary format
    qa_dict = ast.literal_eval(file_contents)

    # Iterate over the answers to the questions and store the results
    for q, a in qa_dict.items():

        docs = elastic_vector_search.similarity_search(q)
        contents = []
        for doc in docs:
            content = doc.page_content
            contents.append(content)

        llm_chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate.from_template(template)
        )

        local_result = llm_chain.predict(context=contents, question=q)

        print("------------------------Question---------------------------------")
        print(q)
        print("---------------------------------------------------------")

        similarity = basicUtils.calculate_cosine_similarity(a, local_result)
        rouge_score = basicUtils.get_rouge_scores(local_result, a)
        meteor_score = basicUtils.get_meteor_score(local_result, a)

        print(f"llm answer: {a}")
        print(f"local answer: {local_result}")
        print(f"Cosine Similarity: {similarity}")
        print(f"ROUGE score: {rouge_score}")
        print(f"METEOR score: {meteor_score}")

        similarity_list.append(similarity)
        rouge_score_list.append(rouge_score)
        meteor_score_list.append(meteor_score)

    r_values_list = list()
    p_values_list = list()
    f1_values_list = list()

    for item in rouge_score_list:
        r_value = item['rouge-1']['r']
        r_values_list.append(r_value)
        r_value = item['rouge-1']['p']
        p_values_list.append(r_value)
        r_value = item['rouge-1']['f']
        f1_values_list.append(r_value)

    similarity_average = basicUtils.calculate_average(numbers=similarity_list)
    r_score_average = basicUtils.calculate_average(numbers=r_values_list)
    p_score_average = basicUtils.calculate_average(numbers=p_values_list)
    f1_score_average = basicUtils.calculate_average(numbers=f1_values_list)
    meteor_score_average = basicUtils.calculate_average(numbers=meteor_score_list)

    print("---------------------------------------------------------")
    print("Total number of pdfs:", len(source_id_list), "Total number of questions：", len(similarity_list),
          "similarity average：", similarity_average,
          "METEOR average：", meteor_score_average, "ROGUE r average：", r_score_average,
          "ROGUE p average：", p_score_average, "ROGUE f1 average：", f1_score_average)
    print("---------------------------------------------------------")

    fig = plt.figure(figsize=(20, 8))
    # Using range(len(y1)) as the x-axis value
    x = list(range(1, len(similarity_list) + 1))

    # Use plt.plot() to draw lines for each list
    plt.plot(x, similarity_list, label='cosine', marker='o', linewidth=2)
    plt.plot(x, f1_values_list, label='rogue-F1', marker='o', linestyle=':', linewidth=2)
    plt.plot(x, meteor_score_list, label='meteor', marker='o', linestyle='-', linewidth=2)

    # Add title and axis labels
    plt.title('Linear Chart of Evaluation')
    plt.xlabel('Number of Questions & Answers')
    plt.ylabel('Metrics')

    # Setting the x-axis scale
    xticks = list(range(0, 281, 30))
    plt.xticks(xticks)

    plt.legend()
    plt.savefig("test.svg", dpi=600, format="svg")
    plt.show()



if __name__ == '__main__':
    langchaincase()
