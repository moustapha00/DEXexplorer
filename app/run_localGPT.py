import os
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline


from prompt_template_utils import get_prompt_template

from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from variables import (
    MAX_NEW_TOKENS,
    MODELS_PATH,
)


def load_model(device_type, model_id, model_basename=None):
    """
    Loads a model for text generation from HuggingFace's model hub.

    Args:
        device_type (str): The type of device to use.
        model_id (str): The identifier of the model to load.
        model_basename (str): The basename of the model for quantized models.

    Returns:
        HuggingFacePipeline: A pipeline for text generation with the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    print(f"Loading Model: {model_id}, on: {device_type}")
    print("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    print("Local LLM Loaded")

    return local_llm


def retrieval_qa_pipline(embeddings, persist_directory, llm, k, promptTemplate_type=None):
    """
    Sets up a retrieval-based Question Answering (QA) system.

    Args:
        embeddings: The embeddings model.
        persist_directory: The directory to persist the vector store.
        llm: The language model.
        k: The number of chunks to retrieve.
        promptTemplate_type (optional): The type of prompt template. Defaults to None.

    Returns:
        RetrievalQA: An initialized retrieval-based QA system.
    """

    # load the vectorstore
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
    retriever = db.as_retriever(search_kwargs={"k": k})

    # get the prompt template and memory if set by the user.
    prompt = get_prompt_template(promptTemplate_type=promptTemplate_type)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
        retriever=retriever,
        return_source_documents=True,  # verbose=True,
        #callbacks=callback_manager,
        chain_type_kwargs={
            "prompt": prompt,
        },
        )

    return qa


def main(llm, embeddings, k, persist_directory, query, promptTemplate_type=None):
    """
    Executes the main information retrieval.

    Args:
        llm: The language model.
        embeddings: The embeddings model.
        k: The number of chunks to retrieve.
        persist_directory: The directory to persist the vector store.
        query: The query to ask the QA system.
        promptTemplate_type (optional): The type of prompt template. Defaults to None.

    Returns:
        tuple: The answer to the query and the source documents used to derive the answer.
    """

    #print(f"Running on: {device_type}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    qa = retrieval_qa_pipline(embeddings, persist_directory, llm, k, promptTemplate_type=promptTemplate_type)
    res = qa(query)
    answer, docs = res["result"], res["source_documents"]
    return answer, docs
