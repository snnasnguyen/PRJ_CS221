import streamlit as st
import os
from openai import OpenAI
import google.generativeai as genai
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

client = OpenAI(api_key="Your API")
os.environ["GOOGLE_API_KEY"] = "Your API"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro")

model_id = "haidangnguyen467/finetune-xlm-r-base-uit-visquad-v1_2"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=True, trust_remote_code=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_id, token=True, trust_remote_code=True)
question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)

def gemini(prompt, model=gemini_model):
    response = model.generate_content(prompt).text.strip()
    return response

def chatgpt(prompt, client=client):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    response = completion.choices[0].message.content
    return response

def xlm_r(context, question, question_answerer=question_answerer):
    return question_answerer(question=question, context=context)['answer']

# Các hàm tạo prompt
#zero-shot prompt
def zero_shot_prompt(context, question):
    prompt = f"""
    Read the following text and answer the question below. The answer must be within the paragraph and be one continuous phrase or keyword, not separated, without explanation or additional information outside the paragraph.   
    Văn bản:
    {context}
    
    Câu hỏi:
    {question}
    
    Trả lời:
  """
    return(prompt)

#one-shot prompt
def one_shot_prompt(context, question):
    prompt = f"""
    Read the following text and answer the question below. The answer must be within the paragraph and be one continuous phrase or keyword, not separated, without explanation or additional information outside the paragraph.   

    Ví dụ:
    Văn bản: "Albert Einstein là nhà vật lý nổi tiếng với thuyết tương đối."
    Câu hỏi: "Ai là người phát triển thuyết tương đối?"
    Trả lời: "Albert Einstein"
    
    Văn bản:
    {context}
    
    Câu hỏi:
    {question}
    
    Trả lời:
  """
    return(prompt)
    
#few_shot prompt
def few_shot_prompt(context, question):
    prompt = f"""
    Read the following text and answer the question below. The answer must be within the paragraph and be one continuous phrase or keyword, not separated, without explanation or additional information outside the paragraph.   
    
    Ví dụ:
    Văn bản: "Albert Einstein là nhà vật lý nổi tiếng với thuyết tương đối."
    Câu hỏi: "Ai là người phát triển thuyết tương đối?"
    Trả lời: "Albert Einstein"
    
    Văn bản: "Kháng sinh được dùng để điều trị các bệnh nhiễm trùng do vi khuẩn gây ra."
    Câu hỏi: "Kháng sinh dùng để điều trị cái gì?"
    Trả lời: "nhiễm trùng do vi khuẩn"
    
    Văn bản: "Isaac Newton đã phát hiện ra định luật vạn vật hấp dẫn vào thế kỷ 17."
    Câu hỏi: "Định luật vạn vật hấp dẫn được phát hiện vào thời kỳ nào?"
    Trả lời: "thế kỷ 17"
    
    Văn bản: "Paris là thủ đô của Pháp và là một trung tâm văn hóa lớn."
    Câu hỏi: "Paris là thủ đô của quốc gia nào?"
    Trả lời: "Pháp"
    
    Văn bản: "Con người khám phá ra lửa có thể được tạo ra bằng cách đánh đá vào nhau vì nó tạo ra tia lửa."
    Câu hỏi: "Tại sao lửa có thể được tạo ra khi đánh đá vào nhau?"
    Trả lời: "tạo ra tia lửa"
    
    Văn bản: "Việc xây dựng cây cầu mất 5 năm do địa hình phức tạp và thời tiết xấu."
    Câu hỏi: "Việc xây dựng cây cầu mất bao lâu?"
    Trả lời: "5 năm"
    
    Văn bản:
    {context}
    
    Câu hỏi:
    {question}
    
    Trả lời:
  """
    return(prompt)

#chain_of_thought prompt
def chain_of_thought_prompt(context, question):
    prompt = f"""
    Read the following context and answer the question below. Follow these steps to ensure accuracy:

    1. Read and fully understand the provided context.
    2. Identify the sentence or sentences in the context that may contain the answer to the question.
    3. Analyze the sentence(s) to confirm the exact information that answers the question.
    4. Provide the exact phrase or continuous keyword from the sentence(s) that answers the question, ensuring the answer is unbroken and directly addresses the question.
    5. Ensure no additional explanation or words are included outside of the exact phrase.
    Văn bản:
    {context}
    
    Câu hỏi:
    {question}
    
    Trả lời:
  """
    return(prompt)

def infer(source, technique_prompt, context, question):
    if technique_prompt == "zero-shot":
        prompt = zero_shot_prompt(context, question)
    elif technique_prompt == "one-shot":
        prompt = one_shot_prompt(context, question)
    elif technique_prompt == "few-shot":
        prompt = few_shot_prompt(context, question)
    elif technique_prompt == "chain-of-thought":
        prompt = chain_of_thought_prompt(context, question)

    if source == "gemini-1.5-pro":
        answer = gemini(prompt)
    elif source == "gpt-4.o-mini":
        answer = chatgpt(prompt)
    elif source == "xlm-r-base":
        answer = xlm_r(context,question,question_answerer)

    return answer

st.title("Hệ Thống Hỏi Đáp Từ Đoạn Văn")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

context = st.text_area("Văn bản:", height=150)
question = st.text_input("Câu hỏi:")
source = st.selectbox("Chọn mô hình:", ["gemini-1.5-pro", "gpt-4.o-mini", "xlm-r-base"])
technique_prompt = st.selectbox("Chọn phương pháp suy luận:", ["zero-shot", "one-shot", "few-shot", "chain-of-thought"])

if st.button("Gửi"):
    if context and question:
        with st.spinner("Đang suy luận..."):
            try:
                answer = infer(source, technique_prompt, context, question)
                st.session_state.chat_history.append((source, question, answer))
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {str(e)}")
    else:
        st.warning("Vui lòng nhập đầy đủ đoạn văn bản và câu hỏi.")

st.header("Lịch sử hội thoại")
for i, (source, q, a) in enumerate(st.session_state.chat_history):
    st.write(f"**Câu hỏi {i+1}:** {q}")
    st.write(f"**Trả lời:** {a}")
    st.write(f"**Mô hình:** {source}")

