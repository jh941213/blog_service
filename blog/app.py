import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_api_key)

def read_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    return prompt

def generate_response(client, model, prompt, user_input):
    full_text = prompt + "\n\n" + user_input
    completion = client.chat.completions.create(
        model=model,  
        messages=[
            {"role": "system", "content": full_text},
            {"role": "user", "content": user_input}
        ]
        ,temperature=0.1,  
        max_tokens=4096
    )
    return completion.choices[0].message.content.strip()

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


st.title("GPT 기반 글쓰기 보조 도구")
st.markdown('<p style="font-size: small;">Made by Simon</p>', unsafe_allow_html=True)
st.markdown("""
    <style>
    button.st-emotion-cache-7ym5gk {
        display: block;
        width: 100%;   
        margin: 0 auto;   
        padding: 8px 0;   
        font-size: 16px;  
        background-color:lightgreen;     
    }
    </style>
    """, unsafe_allow_html=True)


with st.sidebar:
    model_choice = st.selectbox("모델 선택", ["gpt-3.5-turbo-16k", "gpt-4","gpt-4-1106-preview"], key="model_select")
    url = st.text_input("크롤링 URL입력(GPT-4-1106 권장)", key="url_input")
   
    user_input = st.text_area("여기에 글을 입력하세요", height=1100, key="input_text")

    char_count = len(user_input)

    st.caption(f"현재 글자 수: {char_count}")
#크롤링 함수
def crawl_url(url):
    if url:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
   
            return soup.get_text()
        except Exception as e:
            return str(e)
    return "URL이 입력되지 않았습니다."

st.header("Blog Assistant")
buttons = ["SEO 최적화 블로그 글 작성하기",
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
        file_path = "../prompts/0_prompt.txt"
        prompt = read_prompt(file_path)
        response = generate_response(client, model_choice, prompt, user_input)
        result_containers[0].text_area("변환결과", response, height=600)
with col2:
    if st.button("문어체로 작성하기"):
        file_path = "../prompts/1_prompt.txt"
        prompt = read_prompt(file_path)
        response = generate_response(client, model_choice, prompt, user_input)
        result_containers[1].text_area("변환결과", response, height=600)


col1, col2 = st.columns(2)
with col1:
    if st.button("그룹웨어에 작성할 글 요약하기"):
        file_path = "../prompts/2_prompt.txt"
        prompt = read_prompt(file_path)
        response = generate_response(client, model_choice, prompt, user_input)
        result_containers[2].text_area("변환결과", response, height=600)
with col2:
    if st.button("영어로 번역 작성하기"):
        file_path = "../prompts/3_prompt.txt"
        prompt = read_prompt(file_path)
        response = generate_response(client, model_choice, prompt, user_input)
        result_containers[3].text_area("변환결과", response, height=600)

col1, col2 = st.columns(2)
with col1:
    if st.button("일본어로 번역 작성하기"):
        file_path = "../prompts/4_prompt.txt"
        prompt = read_prompt(file_path)
        response = generate_response(client, model_choice, prompt, user_input)
        result_containers[4].text_area("변환결과", response, height=600)
with col2:
    if st.button("이메일 전체공지 작성하기"):
        file_path = "../prompts/5_prompt.txt"
        prompt = read_prompt(file_path)
        response = generate_response(client, model_choice, prompt, user_input)
        result_containers[5].text_area("변환결과", response, height=600)

col1, col2 = st.columns(2)
with col1:
    if st.button("프로젝트 기획 작성하기"):
        file_path = "../prompts/6_prompt.txt"
        prompt = read_prompt(file_path)
        response = generate_response(client, model_choice, prompt, user_input)
        result_containers[6].text_area("변환결과", response, height=600)
with col2:
    if st.button("크롤링 데이터 파싱하기"):
        file_path = "../prompts/7_prompt.txt"
        prompt = read_prompt(file_path)
        crawl_data = crawl_url(url)
        print("프린터확인:",crawl_data)
        response = generate_response(client, model_choice, prompt, crawl_data)
        result_containers[7].text_area("변환결과", response, height=600)

col1, col2 = st.columns(2)
with col1:
    if st.button("DALL-E 이미지 생성하기"):
        prompt = user_input 
        image_url = create_dalle_image(client, prompt)
        if image_url:
            result_containers[8].image(image_url, caption=prompt)
        else:
            result_containers[8].write("이미지 생성 실패")
with col2:
    if st.button("GPT-4-Vision 이미지 해석하기"):
        file_path = "../prompts/9_prompt.txt"
        prompt = read_prompt(file_path)
        response = generate_response(client, model_choice, prompt, user_input)
        result_containers[9].text_area("변환결과", response, height=600)
