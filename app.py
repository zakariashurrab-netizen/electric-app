import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ⚠️ تأكد من كتابة مفتاح جوجل السري الفعلي الخاص بك هنا بدقة
MY_SECRET_KEY = "AQ.Ab8RN6KeiG-s_bRTEyDWGImUF_D-vUBfVY8ftvS-0V8v2FEF3w"

# إعدادات شاشة العرض لمتصفحات اللابتوب والجوال
st.set_page_config(page_title="مستشار الهندسة الكهربائية", page_icon="⚡", layout="centered")

# التنسيق العربي العلوي الفخم
st.markdown("<h2 style='text-align: center; color: #1e3d59;'>💬 مستشار الهندسة الكهربائية الذكي</h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #666;'>الطاقة الشمسية • المحولات • المولدات</h5>", unsafe_allow_html=True)
st.write("---")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_retriever():
    if not os.path.exists("./") or not any(f.endswith('.pdf') for f in os.listdir("./")):
        return None
    loader = DirectoryLoader('./', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # استخدام النموذج الرسمي والمدعوم حالياً في التحديثات
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=MY_SECRET_KEY)
    
    # استخدام مكتبة FAISS الخفيفة داخل الذاكرة لمنع أخطاء السيرفرات السحابية
    db = FAISS.from_documents(texts, embeddings)
    return db.as_retriever(search_kwargs={"k": 4})

# تشغيل الفهرسة الفورية الحية
try:
    retriever = get_retriever()
except Exception as e:
    st.error(f"❌ خطأ داخلي في النظام السحابي: {e}")
    retriever = None

if retriever is None:
    st.error("❌ لم يتم العثور على أي ملفات مراجع هندسية PDF بجانب المجلد السحابي! يرجى رفع الكتب أولاً.")
else:
    # عرض رسائل الشات السابقة بتنسيق محادثة منسق ومقروء
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message.content)

    # صندوق الكتابة السفلي الذكي لمتصفحات اللابتوب والجوال
    if user_query := st.chat_input("اكتب سؤالك الهندسي هنا باللغة العربية..."):
        with st.chat_message("user"):
            st.write(user_query)
            
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("⏳ جاري فحص المراجع وصياغة الرد الاستشاري المنمق..."):
                try:
                    docs = retriever.invoke(user_query)
                    context = "\n\n".join(doc.page_content for doc in docs)

                    system_instruction = (
                        "أنت مهندس استشاري أول وخبير رائد في مجالات الطاقة الشمسية، المحولات، والمولدات الكهربائية. "
                        "مهمتك هي صياغة إجابة فنية منمقة للغاية، بليغة، ومنظمة تمنح المستخدم الثقة المطلقة في جودة التطبيق.\n\n"
                        "⚠️ قواعد التنسيق والصياغة الإلزامية:\n"
                        "1. استخدم لغة عربية فصحى قوية، رصينة، ومهنية جداً بأسلوب استشاري رفيع.\n"
                        "2. رتّب الإجابة بصرياً بشكل رائع باستخدام الرموز التعبيرية الوظيفية (مثل: 🔹 للعناصر، ⚡ للتنبيهات، 📊 للحسابات، 🛠️ للصيانة).\n"
                        "3. قسّم النصوص الطويلة إلى فقرات قصيرة، واجعل العناوين الفرعية واضحة وبخط منمق.\n"
                        "4. ادخل في التفاصيل الفنية الدقيقة المستخرجة من المراجع المتاحة، واذكر الأرقام والمعادلات بوضوح تام.\n"
                        "5. وثّق إجابتك بذكر المصدر أو اسم الدليل المرفق في نهاية الإجابة إن وجد لتعزيز المصداقية.\n"
                        "6. إذا كانت البيانات مفقودة، قل بوضوح: 'نود الإفادة بأن هذه الجزئية الفنية تقع خارج نطاق المراجع المعتمدة لدينا حالياً'.\n\n"
                        "المراجع الهندسية المتاحة للاستنباط:\n{context}"
                    )
                    
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_instruction),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{question}")
                    ])

                    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.3, google_api_key=MY_SECRET_KEY)
                    chain = prompt | llm | StrOutputParser()
                    
                    response = chain.invoke({
                        "context": context,
                        "history": st.session_state.chat_history,
                        "question": user_query
                    })
                    
                    st.write(response)
                    st.session_state.chat_history.append(HumanMessage(content=user_query))
                    st.session_state.chat_history.append(AIMessage(content=response))
                    
                except Exception as e:
                    st.error(f"❌ خطأ أثناء توليد الصياغة: {e}")
