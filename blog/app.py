from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import base64
import whisper
import json
from pytube import YouTube
import whisper

model = whisper.load_model("base")

logo_url = "image/wiznetlogo.png"
def read_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    return prompt

def generate_response(client, model, prompt, user_input):
    if not user_input:
        return "Not found text. Please input text."

    try:
        completion = client.chat.completions.create(
            model=model,  
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1,  
            max_tokens=4096
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        raise Exception("프롬프트 처리 중 오류가 발생했습니다.")

def create_dalle_image(client, prompt):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url  
    except Exception as e:
        return str(e)

def gpt4_vision(client,base64_image):

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {client.api_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "What’s in this image?"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)


st.title("GPT Prompt 자동 생성기")
st.markdown('<p style="font-size: small;">Made by Simon</p>', unsafe_allow_html=True)
st.markdown("""
    <style>
    button.st-emotion-cache-7ym5gk {
        display: block;
        width: 100%;   
        margin: 0 auto;   
        padding: 8px 0;   
        font-size: 16px;  
        background: linear-gradient(to bottom, #a8c461, #36cd9c); /* 그라데이션 적용 */
        border: none;
        box-shadow: 3px 3px 10px rgba(0, 0, 0, 0.5);
        color: white;
        font-weight: bold;
        text-align: center;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }

    button.st-emotion-cache-7ym5gk::before {
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to bottom, rgba(255, 255, 255, 0.2), transparent);
        transform: translateY(-100%);
        transition: transform 0.3s ease;
    }

    button.st-emotion-cache-7ym5gk:hover::before {
        transform: translateY(0);
    }

    button.st-emotion-cache-7ym5gk:hover {
        background: linear-gradient(to bottom, #36cd9c, #a8c461); /* 호버 효과 */
    }
    </style>
    """, unsafe_allow_html=True)



with st.sidebar:
    model_choice = st.selectbox("모델 선택", ["gpt-3.5-turbo-16k", "gpt-4","gpt-4-1106-preview"], key="model_select")

    user_api_key = st.text_input("OpenAI API Key", type="password")

    url = st.text_input("크롤링 URL입력(GPT-4-1106 권장)", key="url_input")
   
    user_input = st.text_area("여기에 글을 입력하세요", height=1000, key="input_text")

    char_count = len(user_input)

    st.caption(f"현재 글자 수: {char_count}")

    uploaded_image = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"], key="image_uploader")

if user_api_key:
    print(user_api_key)
    client = OpenAI(api_key=user_api_key)
else:
    st.error("API 키가 필요합니다.")

#크롤링 함수
#playweight로 수정예정
def crawl_url(url):
    if url:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
   
            return soup.get_text()
        except Exception as e:
            return str(e)
    return "URL이 입력되지 않았습니다."

def download_youtube(url):
    yt = YouTube(url)
    yt.streams.filter(only_audio=True).first().download(
    output_path='.', filename='input.mp3')
    result = model.transcribe("input.mp3")
    print(len(result))
    return result

def text_spliter(result):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, overlap_siz=50, length_function=len)
    docs = [Document(page_content=x) for x in text_spliter.split_text(result["text"])]
    split_docs = text_splitter.split_documetns(docs)

def langchain_templates(client,split_docs):
    llm=ChatOpenAI(temperature=0.1, openai_api_key=client.api_key)
    map_template = read_prompt("prompts/map_template.txt")
    reduce_template = read_prompt("prompts/reduce_template.txt")
    map_prompt = PromptTemplate(map_template)
    reduce_prompt = PromptTemplate(reduce_template)
    #map reduce chains
    #1. reduce chain
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    
    combine_documents_chain = StuffDocumentsChain(llm_chain=reduce_chain, document_veriable_name="doc_summaries")
    
    reduce_documents_chain = ReduceDocumentsChain(combine_documents_chain=combine_documents_chain, collapse_documents_chain=collapse_documents_chain, token_max=4096)

    #2. map chain
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    map_reduce_chain = MapReduceChain(llm_chain=map_chain, reduce_documents_chain=reduce_documents_chain, document_variable_name="docs",return_intermediate_steps=False)
    
    sum_result = map_reduce_chain.run(spilit_docs)

    return sum_results


st.write("")
st.image(logo_url)
st.header("WIZnet ChatGPT 글쓰기 도우미")
st.write("")
buttons = ["SEO 블로그 글로 변환하기",
             "문어체로 작성하기",
             "그룹웨어에 작성할 글 요약하기",
             "영어로 번역 작성하기",
             "일본어로 번역 작성하기",
             "이메일 전체공지 작성하기",
             "프로젝트 기획 작성하기",
             "크롤링 데이터 파싱하기",
             "DALL-E 이미지 생성하기",
             "GPT-4-Vision 이미지 해석하기"
             ]

result_containers = [st.empty() for _ in range(10)]

col1, col2 = st.columns(2)
with col1:
    if st.button("SEO 최적화 블로그 글 작성하기"):
        try:
            file_path = "prompts/0_prompt.txt"
            prompt = read_prompt(file_path)
            response = generate_response(client, model_choice, prompt, user_input)
            result_containers[0].text_area("변환결과", response, height=600)
        except Exception as e:
            print(e)
            st.error("오류가 발생했습니다. Prompt를 확인해주세요.")
with col2:
    if st.button("문어체로 작성하기"):
        try:
            file_path = "prompts/1_prompt.txt"
            prompt = read_prompt(file_path)
            response = generate_response(client, model_choice, prompt, user_input)
            result_containers[1].text_area("변환결과", response, height=600)
        except Exception as e:
            st.error("오류가 발생했습니다. Prompt를 확인해주세요.")

col1, col2 = st.columns(2)
with col1:
    if st.button("그룹웨어에 작성할 글 요약하기"):
        try:
            file_path = "prompts/2_prompt.txt"
            prompt = read_prompt(file_path)
            response = generate_response(client, model_choice, prompt, user_input)
            result_containers[2].text_area("변환결과", response, height=600)
        except Exception as e:
            st.error("오류가 발생했습니다. Prompt를 확인해주세요.")
with col2:
    if st.button("영어로 번역 작성하기"):
        try:
            file_path = "prompts/3_prompt.txt"
            prompt = read_prompt(file_path)
            response = generate_response(client, model_choice, prompt, user_input)
            result_containers[3].text_area("변환결과", response, height=600)
        except Exception as e:
            st.error("오류가 발생했습니다. Prompt를 확인해주세요.")

col1, col2 = st.columns(2)
with col1:
    if st.button("일본어로 번역 작성하기"):
        try:
            file_path = "prompts/4_prompt.txt"
            prompt = read_prompt(file_path)
            response = generate_response(client, model_choice, prompt, user_input)
            result_containers[4].text_area("변환결과", response, height=600)
        except Exception as e:
            st.error("오류가 발생했습니다. Prompt를 확인해주세요.")
with col2:
    if st.button("이메일 전체공지 작성하기"):
        try:
            file_path = "prompts/5_prompt.txt"
            prompt = read_prompt(file_path)
            response = generate_response(client, model_choice, prompt, user_input)
            result_containers[5].text_area("변환결과", response, height=600)
        except Exception as e:
            st.error("오류가 발생했습니다. Prompt를 확인해주세요.")

col1, col2 = st.columns(2)
with col1:
    if st.button("프로젝트 기획 작성하기"):
        try:
            file_path = "prompts/6_prompt.txt"
            prompt = read_prompt(file_path)
            response = generate_response(client, model_choice, prompt, user_input)
            result_containers[6].text_area("변환결과", response, height=600)
        except Exception as e:  
            st.error("오류가 발생했습니다. Prompt를 확인해주세요.")

with col2:
    if st.button("크롤링 데이터 파싱하기"):
        try:
            file_path = "prompts/7_prompt.txt"
            prompt = read_prompt(file_path)
            crawl_data = crawl_url(url)
            print("프린터확인:",crawl_data)
            response = generate_response(client, model_choice, prompt, crawl_data)
            result_containers[7].text_area("변환결과", response, height=600)
        except Exception as e:
            st.error("오류가 발생했습니다. Prompt를 확인해주세요.")

col1, col2 = st.columns(2)
with col1:
    if st.button("DALL-E 이미지 생성하기"):
        try:
            prompt = user_input 
            image_url = create_dalle_image(client, prompt)
            if image_url:
                result_containers[8].image(image_url, caption=prompt)
            else:
                result_containers[8].write("이미지 생성 실패")
        except Exception as e: 
            st.error("오류가 발생했습니다. Prompt를 확인해주세요.")
with col2:
    if st.button("GPT-4-Vision 이미지 해석하기"):
        try:
            text = gpt4_vision(client,base64_image)
            result_containers[9].text_area("변환결과", text, height=600)
        except Exception as e:
            st.error("오류가 발생했습니다. Prompt를 확인해주세요.")
# #'추후 도입예정
# #col1, col2 = st.columns(2)
# #with col1:
# #    if st.button("youtube 영상 요약하기"):
#         try:
#             if url:
#                 result_url = download_youtube(url)
#                 split_docs = text_spliter(result_url)
#                 final_result = langchain_templates(client,split_docs)
#                 result_containers[10].text_area("변환결과", final_result, height=600)
#             else:
#                 st.error("URL이 입력되지 않았습니다.")
#         except Exception as e:
#             print(e)
#             st.error("오류가 발생했습니다. Prompt를 확인해주세요.")
            
# with col2:
#     if st.button("GPT-4-Vision 이미지 해석하기2"):
#         try:
#             text = gpt4_vision(client,base64_image)
#             result_containers[9].text_area("변환결과", text, height=600)
#         except Exception as e:
#             print(e)
#             st.error("오류가 발생했습니다. Prompt를 확인해주세요.")
# '''
