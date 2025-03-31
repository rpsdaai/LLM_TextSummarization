import urllib

from config import llm_local as llm
from pathlib import Path

from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader

load_dotenv()

# Using local LLM. Remember to start up LLM model in LLM Studio
# https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/use-cases/document-summarization/summarization_large_documents_langchain.ipynb
# https://github.com/gkamradt/langchain-tutorials/blob/main/data_generation/5%20Levels%20Of%20Summarization%20-%20Novice%20To%20Expert.ipynb

def doProcessDocuments(pdf_url, no_of_pages):
    # https://stackoverflow.com/questions/50110800/python-pathlib-make-directories-if-they-don-t-exist
    data_folder = Path.cwd() / "data"
    try:
        data_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Folder is already there")
    else:
        print("Folder was created")

    # pdf_url = "https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf"
    pdf_file = str(Path(data_folder, pdf_url.split("/")[-1]))

    urllib.request.urlretrieve(pdf_url, pdf_file)

    pdf_loader = PyPDFLoader(pdf_file)
    pages = pdf_loader.load_and_split()
    # print(pages[3].page_content)

    return pages[:no_of_pages]
    # three_pages = pages[:3]

def doStuffing(document):
    prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
                Return your response in bullet points which covers the key points of the text.
                ```{text}```
                BULLET POINT SUMMARY:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    stuff_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)      

    result = stuff_chain.invoke(document)

    # Keys: ['input_documents', 'output_text']
    print("Keys:", list(result.keys()))
    # [[Document(metadata={'source': 'D:\\Users\\ng_a\\My NP SDGAI\\PDC-2\\LLM\\Assignment\\app\\data\\practitioners_guide_to_mlops_whitepaper.pdf', 
    # 'page': 0, 'page_label': '1'}, page_content='Practitioners guide to MLOps:  \nA framework for continuous \ndelivery and automation of  \nmachine learning.White paper\nMay 2021\nAuthors:  \nKhalid Salama,  \nJarek Kazmierczak,  
    # \nDonna Shut'), ....
    print("Values:", list(result.values()))
    # print(result["output_text"])  

    return result['output_text']

def doMapReduce(document):
    # https://python.langchain.com/docs/versions/migrating_chains/map_reduce_chain/
    map_prompt_template = """
                        Write a summary of this chunk of text that includes the main points and any important details.
                        {text}
                        """

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
                        Write a concise summary of the following text delimited by triple backquotes.
                        Return your response in bullet points which covers the key points of the text.
                        ```{text}```
                        BULLET POINT SUMMARY:
                        """

    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text"]
    )

    map_reduce_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=True,
    )

    result = map_reduce_chain.invoke(document)

    return(result["output_text"])    

def doRefine(document):
    question_prompt_template = """
                    Please provide a summary of the following text.
                    TEXT: {text}
                    SUMMARY:
                    """

    question_prompt = PromptTemplate(
        template=question_prompt_template, input_variables=["text"]
    )

    refine_prompt_template = """
                Write a concise summary of the following text delimited by triple backquotes.
                Return your response in bullet points which covers the key points of the text.
                ```{text}```
                BULLET POINT SUMMARY:
                """

    refine_prompt = PromptTemplate(
        template=refine_prompt_template, input_variables=["text"]
    )

    refine_chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
    )

    result = refine_chain.invoke(document)

    return(result["output_text"])

if __name__ == "__main__":
    pages = doProcessDocuments("https://services.google.com/fh/files/misc/practitioners_guide_to_mlops_whitepaper.pdf", 3)  
    print ("STUFFING.....\n")
    print (doStuffing(pages))  

    print ("MAP REDUCE ....\n")
    print (doMapReduce(pages))

    print ("REFINE ....\n")
    print (doRefine(pages))